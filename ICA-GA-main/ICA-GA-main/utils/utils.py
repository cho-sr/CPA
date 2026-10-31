# --------------------------------------------------
# 通用的一些零碎函数
# --------------------------------------------------

import os
import math
import time
import random
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from typing import List, Union, Tuple, Dict, Any
from datetime import datetime
from collections import Counter

from .log import logger


def set_seed(seed: int = 0):
    """设置随机种子

    Args:
        seed (int): 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保PyTorch的计算是可重复的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def slice_to_str(s: slice):
    """切片实例转字符串

    Args:
        s (slice): 切片

    Returns:
        例如 "[1:3]"
    """
    if not isinstance(s, slice):
        raise TypeError("Expected a slice object")
    start = str(s.start) if s.start is not None else ""
    stop = str(s.stop) if s.stop is not None else ""
    step = str(s.step) if s.step is not None else ""
    parts = [start, stop]
    if step:
        parts.append(step)
    return f'[{":".join(parts)}]'


def format_time(timestamp: float = 0):
    """将时间戳转换为年/月/日/时字符串

    Args:
        timestamp (float): 时间戳。如果不传入，则使用当前时间
    """
    if timestamp <= 0:
        timestamp = time.time()
    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime(r"%Y年%m月%d日 %H:%M")


def format_duration_time(duration: float):
    """将时间（秒）转为天/时/分/秒字符串

    Args:
        duration (float): 时间
    """
    days, rem = divmod(duration, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    parts = []
    if days >= 1:
        parts.append(f"{int(days)}天")
    if hours >= 1:
        parts.append(f"{int(hours)}小时")
    if minutes >= 1:
        parts.append(f"{int(minutes)}分钟")
    if (hours < 1 and seconds >= 1) or not parts:
        parts.append(f"{int(seconds)}秒")

    return "".join(parts)


def visualize(
    batch_tensor: Union[torch.Tensor, List],  # 图像张量或图像张量的列表
    label="Sample",  # 展示标签前缀
    max_num=10,  # 展示的数量上限
    save_path="",  # 图像保存位置，空为不保存
    is_show=True,  # 是否展示图像
    is_label=True,  # 是否显示标签
    columns=8,  # 列数
):
    """可视化一组图像张量或特征向量
    - batch_tensor (torch.Tensor): 包含图像数据或特征向量的张量，形状 (batch_size, ...)。
    - label (str): 展示标签前缀。
    - max_num (int): 展示的图像数量上限。
    - save_path (str): 图像保存的路径。如果为空，则不保存图像。
    - is_show (bool): 是否展示图像。如果为 True，则展示图像。
    """
    if isinstance(batch_tensor, list):  # 如果输入是列表，转换为张量
        batch_tensor = torch.stack(batch_tensor)
    batch_tensor = batch_tensor.cpu()
    n = batch_tensor.shape[0]
    image_sizes_dim1 = {
        512: (32, 16),
        1024: (32, 32),
        2048: (64, 32),
        100352: (512, 196),
    }
    image_size = None
    if batch_tensor.dim() == 2:  # 1维张量
        image_size = image_sizes_dim1.get(batch_tensor.shape[1], None)
    elif 3 <= batch_tensor.dim() <= 4:  # 3维张量
        image_size = "img"
        tensor2image = transforms.ToPILImage()
        if batch_tensor.dim() == 3:
            batch_tensor = batch_tensor.unsqueeze(0)
            n = 1
        batch_tensor = batch_tensor.clamp(0, 1)
    assert image_size, f"Unsupported input shape {batch_tensor. shape}"
    if n > max_num:
        n = max_num
    rows = math.ceil(n / columns)
    rows_n = 2 if is_label else 1.8
    if columns == 10:
        rows_n *= 0.75
    fig, axes = plt.subplots(rows, columns, figsize=(14, rows_n * rows))
    for i in range(n):
        ax = axes[i // columns, i % columns] if rows > 1 else axes[i % columns]
        if is_label:
            ax.set_title(f"{label} {i+1}")
        ax.axis("off")  # 关闭坐标轴显示
        if image_size == "img":
            img = tensor2image(batch_tensor[i].squeeze())
            ax.imshow(img, cmap="gray")
        else:
            # 将 1D 张量重整形为 2D 张量
            tensor_reshaped = batch_tensor[i].view(*image_size)
            # 归一化到 0-1 范围
            tensor_normalized = tensor_reshaped - tensor_reshaped.min()
            tensor_normalized /= tensor_reshaped.max() - tensor_reshaped.min()
            tensor_numpy = tensor_normalized.numpy()
            ax.imshow(tensor_numpy, cmap="viridis")
    # 如果样本数不是列数的倍数，隐藏多余的子图
    if n % columns != 0:
        for j in range(n, rows * columns):
            fig.delaxes(
                axes[j // columns, j % columns] if rows > 1 else axes[j % columns]
            )
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    if save_path:
        # 如果父目录路径非空且不存在，则递归创建目录
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            logger.debug(f"创建图片目录：{directory}")
        plt.savefig(save_path, bbox_inches="tight")
        # plt.savefig(save_path)
    if is_show:
        plt.show()
    return plt


# 字典转字符串
def dict_to_str(d: dict):
    parts = []
    for key, value in d.items():
        formatted_value_str = ""
        if isinstance(value, float):
            formatted_value_str = f"{value:.4f}"
        elif isinstance(value, int):
            formatted_value_str = f"{value:>4}"
        elif isinstance(value, str):
            formatted_value_str = value
        else:
            formatted_value_str = str(value)
        parts.append(f"{key}: {formatted_value_str}")
    log_str = ", ".join(parts)
    return log_str


# 返回当前年月日字符串（6位）
def get_current_date_str():
    return datetime.now().strftime("%y%m%d")


# 展平字典
def flatten_dict(nested_dict, parent_key="", sep="."):
    """
    将嵌套字典展平为单层字典。

    参数:
        nested_dict (dict): 需要展平的嵌套字典。
        parent_key (str): 内部使用，用于递归时传递父键。
        sep (str): 用于连接多层键的分隔符。

    返回:
        dict: 展平后的单层字典。
    """
    items = []
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# 分析标签重复率
def analyze_y(y: torch.Tensor) -> dict:
    """
    分析一个批次中标签张量的信息。

    Args:
        y: 从 DataLoader 输出的标签张量。

    Returns:
        一个包含标签信息的字典，包括：
        - sample_count (样本数量)
        - unique_sample_count (独立的样本数量)
        - uniqueness_rate (样本独立率)
        - repetition_rate (样本重复率)
    """
    # 确保输入是 tensor 并展平
    if not isinstance(y, torch.Tensor):
        try:
            y = torch.tensor(y)
        except Exception as e:
            raise TypeError(f"输入 y 无法转换为 torch.Tensor: {e}")

    # 将 tensor 转换为 python list 以便使用 Counter
    labels_list = y.view(-1).tolist()

    # 样本总数
    sample_count = len(labels_list)
    if sample_count == 0:
        return {
            "sample_count": 0,
            "unique_sample_count": 0,
            "uniqueness_rate": 0.0,
            "repetition_rate": 0.0,
        }

    # 使用 Counter 高效计算每个标签的出现次数
    label_counts = Counter(labels_list)

    # 独立的样本数量 (只出现一次的标签)
    unique_sample_count = sum(1 for count in label_counts.values() if count == 1)

    # 样本独立率
    uniqueness_rate = unique_sample_count / sample_count

    # 样本重复率
    repetition_rate = 1.0 - uniqueness_rate

    return {
        "sample_count": sample_count,
        "unique_sample_count": unique_sample_count,
        "uniqueness_rate": uniqueness_rate,
        "repetition_rate": repetition_rate,
    }


# 从高维特征张量的中心提取 s×s 区域并展平为一维
def sample_and_flatten_center(tensor, s):
    """
    从高维特征张量的中心提取 s×s 区域并展平为一维

    参数:
        tensor (torch.Tensor): 输入张量，形状为 (b, m, n, n)
        s (int): 采样尺寸，需满足 0 < s <= n

    返回:
        torch.Tensor: 展平后的一维张量，形状为 (m * s * s,)
    """
    assert len(tensor.shape) == 4, "输入张量必须是四维 (b, m, n, n)"
    b, m, n, _ = tensor.shape
    start = (n - s) // 2  # 计算中心区域起始索引

    # 切片提取中心 s×s 区域
    center_patch = tensor[:, :, start : start + s, start : start + s]

    # 展平成一维张量 (b, m*s*s)
    flattened = center_patch.reshape(center_patch.size(0), -1)
    return flattened


def find_and_sort_files(
    directory: str, pattern_str: str
) -> List[Tuple[Union[int, str], str]]:
    """
    在指定目录中搜索匹配特定模式的文件，并返回排序后的结果列表。

    模式中的 '*' 是一个通配符，用于匹配文件名中的任意字符。

    Args:
        directory (str): 要搜索的目录路径。
        pattern_str (str): 文件名匹配字符串，必须包含一个 '*'。
                           例如: "ResNet50_batchs=*.pth"

    Returns:
        List[Tuple[Union[int, str], str]]:
            一个元组列表。每个元组包含 (匹配值, 完整文件名)。
            - 如果匹配值是数字，则转换为 int 类型。
            - 列表首先按数字升序排列，然后是按字母顺序排列的字符串。

    Raises:
        ValueError: 如果目录不存在或模式字符串不含 '*'。
    """
    if not os.path.isdir(directory):
        raise ValueError(f"错误: 目录 '{directory}' 不存在或不是一个有效的目录。")

    if "*" not in pattern_str:
        raise ValueError("错误: 模式字符串 'pattern_str' 必须包含 '*' 通配符。")

    # 1. 将模式分割为前缀和后缀
    prefix, suffix = pattern_str.split("*", 1)

    numeric_results = []
    string_results = []

    # 2. 遍历目录中的所有条目
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            continue  # 跳过子目录

        # 3. 检查文件名是否符合前缀和后缀
        if filename.startswith(prefix) and filename.endswith(suffix):
            # 4. 提取 '*' 匹配的部分
            # 计算切片的结束位置，如果后缀为空则为None
            end_slice = -len(suffix) if suffix else None
            value_str = filename[len(prefix) : end_slice]

            if not value_str:
                continue  # 如果通配符匹配部分为空，则跳过

            # 5. 尝试将匹配值转换为数字，否则保留为字符串
            if value_str.isdigit():
                numeric_results.append((int(value_str), filename))
            else:
                string_results.append((value_str, filename))

    # 6. 对两类结果分别排序
    numeric_results.sort()  # 元组默认按第一个元素（数字）升序排序
    string_results.sort()  # 字符串结果按字母顺序排序

    # 7. 合并结果并返回
    return numeric_results + string_results


def add_prefix_to_dict_keys(d: dict, prefix: str):
    """为字典d的每个key增加前缀prefix

    Args:
        d (dict): 目标字典
        prefix (str): 前缀字符串
    Returns:
        dict: 修改后的字典
    """
    return {prefix + key: value for key, value in d.items()}


def merge_dicts(dicts_list: List[Dict[str, Any]], separator: str) -> Dict[str, str]:
    """
    合并多个字典中相同键的值，将它们转换为字符串并用指定分隔符连接。

    该函数处理一个字典列表，其中每个字典可能包含部分不同的键。它会收集所有字典中相同键的值，
    将这些值转换为字符串，并用指定的分隔符将它们连接起来。返回的字典键顺序基于输入字典中键的首次出现顺序。

    参数:
    dicts_list -- 包含多个字典的列表。每个字典可以有部分相同的键，值可以是任意类型
    separator -- 用于连接值的分隔符字符串

    返回:
    一个字典，其中：
      - 键是输入字典中出现的所有键（按首次出现顺序）
      - 值是将所有字典中对应键的值转换为字符串后，用分隔符连接形成的字符串

    示例:
    >>> data = [{"a": 1, "b": "1"}, {"a": 2, "c": True}]
    >>> merge_dicts(data, "|")
    {'a': '1|2', 'b': '1', 'c': 'True'}
    """
    # 收集所有键并保持首次出现顺序
    all_keys = []
    for d in dicts_list:
        for key in d:
            if key not in all_keys:
                all_keys.append(key)

    def format_number(value: Union[int, float]) -> str:
        if isinstance(value, float) and value != int(value):
            if value > 1 or value < -1:
                return f"{value:.1f}"
            else:
                return f"{value:.4f}"
        # 处理整数或小数部分为0的浮点数
        return (
            str(int(value))
            if isinstance(value, float) and value.is_integer()
            else str(value)
        )

    # 为每个键收集所有字典中对应的值
    merged = {}
    for key in all_keys:
        values = []
        for d in dicts_list:
            if key in d:
                value = d[key]
                # 数字特殊处理
                if isinstance(value, (int, float)):
                    formatted = format_number(value)
                    values.append(formatted)
                else:
                    values.append(str(value))
        # 用分隔符连接所有值
        merged[key] = separator.join(values)

    return merged
