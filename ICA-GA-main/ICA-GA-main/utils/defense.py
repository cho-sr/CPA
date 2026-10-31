# --------------------------------------------------
# 防御算法
# --------------------------------------------------

import torch
from typing import List

from .parameter_list_dict import ParameterListDict
from utils import logger


def dp_noise(gradients: ParameterListDict, noise_multiplier: float):
    """
    向梯度列表添加用于差分隐私的高斯噪声。

    Args:
        gradients (ParameterListDict): 模型各层梯度的ParameterListDict。
        noise_multiplier (float): 噪声强度系数。该值越大，添加的噪声越多
    """
    for i, grad in enumerate(gradients):
        noise = torch.randn_like(grad) * noise_multiplier
        gradients.data[i] += noise


def quantile(tensor, q, dim=None, keepdim=False):
    """
    https://github.com/pytorch/pytorch/issues/64947#issuecomment-2810054982
    Computes the quantile of the input tensor along the specified dimension.

    Parameters:
    tensor (torch.Tensor): The input tensor.
    q (float): The quantile to compute, should be a float between 0 and 1.
    dim (int): The dimension to reduce. If None, the tensor is flattened.
    keepdim (bool): Whether to keep the reduced dimension in the output.
    Returns:
    torch.Tensor: The quantile value(s) along the specified dimension.
    """
    assert 0 <= q <= 1, "\n\nquantile value should be a float between 0 and 1.\n\n"

    if dim is None:
        tensor = tensor.flatten()
        dim = 0

    sorted_tensor, _ = torch.sort(tensor, dim=dim)
    num_elements = sorted_tensor.size(dim)
    index = q * (num_elements - 1)
    lower_index = int(index)
    upper_index = min(lower_index + 1, num_elements - 1)
    lower_value = sorted_tensor.select(dim, lower_index)
    upper_value = sorted_tensor.select(dim, upper_index)
    # linear interpolation
    weight = index - lower_index
    quantile_value = (1 - weight) * lower_value + weight * upper_value

    return quantile_value.unsqueeze(dim) if keepdim else quantile_value


def compress_gradient(gradients: ParameterListDict, compression_rate: float):
    """
    梯度压缩，将最小的部分元素置为0

    Args:
        gradients (ParameterListDict): 模型各层梯度的ParameterListDict。
        compression_rate (float): 压缩率，取值范围 (0, 1)，表示保留的比例
    """
    if compression_rate == 1:
        return
    if not (0 < compression_rate < 1):
        raise ValueError("Compression rate must be between 0 and 1.")

    # 1. 将所有梯度压平并拼接成一个向量
    flat_gradients = torch.cat([g.flatten() for g in gradients])
    # 2. 计算所有梯度元素的绝对值
    all_magnitudes = torch.abs(flat_gradients)
    # 3. 计算全局阈值，找到对应压缩率的分位数
    # threshold = torch.quantile(all_magnitudes, compression_rate) # RuntimeError: quantile() input tensor is too large
    threshold = quantile(all_magnitudes, compression_rate)

    # --- 创建并应用掩码 ---
    total_elements = 0
    zero_elements = 0

    for i, grad in enumerate(gradients):
        # 4. 根据阈值创建掩码 (mask)
        # 绝对值大于等于阈值的元素保留，其余的置零
        mask = (torch.abs(grad) >= threshold).to(grad.dtype)

        # 5. 应用掩码得到压缩后的梯度
        compressed_grad = grad * mask
        gradients.data[i] = compressed_grad

        # 统计稀疏度用于验证
        total_elements += grad.numel()
        zero_elements += torch.count_nonzero(mask == 0).item()

    # 计算实际稀疏度
    actual_sparsity = zero_elements / total_elements if total_elements > 0 else 0.0
    logger.info(f"梯度压缩：实际稀疏率{actual_sparsity}")


def compress_gradient_layer(
    gradients: ParameterListDict, compression_rate: float, layer_name: str
):
    """
    压缩指定层的梯度

    Args:
        gradients (ParameterListDict): 模型各层梯度的ParameterListDict。
        compression_rate (float): 压缩率，取值范围 (0, 1)，表示保留的比例
        layer_name (str): 指定的层名称

    返回：
    torch.Tensor: 压缩后的梯度张量
    """
    if compression_rate == 1:
        return
    if not (0 < compression_rate < 1):
        raise ValueError("Compression rate must be between 0 and 1.")

    gradient_tensor = gradients[layer_name]

    # 计算保留的元素数量
    num_elements = gradient_tensor.numel()
    num_to_keep = int(num_elements * compression_rate)

    # 获取绝对值最大的元素的索引
    _, indices = torch.topk(torch.abs(gradient_tensor.flatten()), num_to_keep)

    # 创建一个全零的张量用于保存压缩后的梯度
    compressed_gradient = torch.zeros_like(gradient_tensor)

    # 将保留的元素放回到压缩后的张量中
    compressed_gradient.flatten()[indices] = gradient_tensor.flatten()[indices]

    gradients.data[gradients.names_index[layer_name]] = compressed_gradient
