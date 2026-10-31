import math
import torch
from copy import deepcopy
from torch import nn, optim
from typing import Iterator, Dict, Any

from models.fl.base_cnn_model import BaseCnnModel
from utils.datasets import get_dataloader
from utils.bn_statistics_hook import BNStatisticsHook


def data_iter_fedsgd(
    model: BaseCnnModel,
    dataloader=None,
    dataset_name="cifar100",
    batch_size=10,
    shuffle=False,
    image_size=224,
    train=False,
    skip_index=0,
    need_bn_statistics=False,
    need_dY=False,
    model_mode_train=True,
    dataloader_seed=None,
    device="cpu",
) -> Iterator[Dict[str, Any]]:
    """
    模拟 FedSGD (Federated Stochastic Gradient Descent) 的数据迭代器。
    按批次处理数据，计算每个批次的梯度，并将其与相关信息一同返回。

    Args:
        model (BaseCnnModel): 用于计算梯度的 PyTorch 模型实例。
        dataloader (Optional[DataLoader], optional): 预先配置的数据加载器。
            如果提供此参数，则忽略 `dataset_name`, `batch_size`, `shuffle`,
            `image_size`, `train` 参数。
        dataset_name (str, optional): 数据集的名称 (如果 `dataloader` 未提供)。
        batch_size (int, optional): 批处理大小 (如果 `dataloader` 未提供)。
        shuffle (bool, optional): 是否打乱数据 (如果 `dataloader` 未提供)。
        image_size (int, optional): 图像尺寸 (如果 `dataloader` 未提供)。
        train (bool, optional): 是否使用训练集 (如果 `dataloader` 未提供)。
        skip_index (int, optional): 跳过数据加载器中前 `skip_index` 个批次。
        need_bn_statistics (bool): 是否返回BN层数据。
        need_dY (bool): 是否返回推理结果Y的梯度。
        model_mode_train (bool): T时模型设为train()，F时设为eval()。
        dataloader_seed (int, optional): 指定数据加载器的种子。
        device (str, optional): 计算设备 ("cpu" 或 "cuda")。

    Yields:
        Iterator[Dict[str, Any]]: 一个字典，包含以下键：
            - "gradient" (ParameterListDict): 计算得到的模型梯度。
            - "x" (torch.Tensor): 当前批次的输入数据样本 (已克隆并分离)。
            - "y" (torch.Tensor): 当前批次的真实标签 (已克隆并分离)。
            - "feature_vector" (Optional[torch.Tensor]): 从模型获取的特征向量。
            - "bn_statistics" (List[torch.Tensor]): need_bn_statistics=True时存在，BN层统计信息。
    """
    if dataloader is None:
        data_gen = None
        if dataloader_seed is not None:
            data_gen = torch.Generator().manual_seed(dataloader_seed)
        dataloader = get_dataloader(
            dataset_name, batch_size, shuffle, image_size, train, generator=data_gen
        )
    model = model.to(device)
    if model_mode_train:
        model.train()
    else:
        model.eval()
    criterion = nn.CrossEntropyLoss()
    if need_bn_statistics:
        bn_hook = BNStatisticsHook(model=model)
    else:
        bn_hook = None

    for index, (x, y) in enumerate(dataloader):
        if index < skip_index:
            continue
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        if bn_hook is not None:
            bn_hook.clear()
            bn_hook.register()
        y_pred = model(x)
        if need_dY:
            y_pred.retain_grad()
        loss = criterion(y_pred, y)
        loss.backward()
        data = {
            "gradient": model.get_gradient(device="cpu"),  # 梯度
            "x": x.clone().detach().cpu(),  # 样本
            "y": y.clone().detach().cpu(),  # 标签
            "feature_vector": model.get_feature_vector().cpu(),  # 前向传播时得到的特征向量
        }
        if bn_hook is not None:
            data["bn_statistics"] = bn_hook.mean_var_list
            bn_hook.unregister()
        if need_dY:
            dY = y_pred.grad.clone().cpu()
            data["dY"] = dY
        yield data


def data_iter_fedavg(
    model: BaseCnnModel,
    # 数据集设置
    dataloader=None,
    dataset_name="cifar100",
    batch_size=10,
    shuffle=False,
    image_size=224,
    train=False,
    skip_index=0,
    dataloader_seed=None,
    # 客户端设置
    local_steps=1,
    local_epochs=2,
    local_lr=0.1,
    need_dY=False,
    device="cpu",
) -> Iterator[Dict[str, Any]]:
    """
    模拟 FedAvg (Federated Averaging) 中客户端行为的数据迭代器。
    该迭代器模拟多个客户端的本地训练过程。每个客户端使用一部分数据进行
    指定轮次的本地训练，然后返回本地模型参数的更新量。

    Args:
        model (BaseCnnModel): 初始的全局 PyTorch 模型实例。
        dataloader (Optional[DataLoader], optional): 预先配置的数据加载器。
            如果提供，将忽略 `dataset_name`, `batch_size`, `shuffle`,
            `image_size`, `train`。
        dataset_name (str, optional): 数据集名称 (如果 `dataloader` 未提供)。
        batch_size (int, optional): 每个客户端内单次训练的批大小 (如果 `dataloader` 未提供)。
        shuffle (bool, optional): 是否打乱整个数据集 (如果 `dataloader` 未提供)。
        image_size (int, optional): 图像尺寸 (如果 `dataloader` 未提供)。
        train (bool, optional): 是否使用训练集 (如果 `dataloader` 未提供)。
        skip_index (int, optional): 跳过数据加载器中前 `skip_index` 个批次。
        dataloader_seed (int, optional): 指定数据加载器的种子。
        local_steps (int, optional): 每个模拟客户端在一个 epochs 中使用多少批样本进行训练。
        local_epochs (int, optional): 每个客户端本地训练的轮次数。
        local_lr (float, optional): 客户端本地训练期间使用的学习率。
        need_dY (bool): 是否返回推理结果Y的梯度。
        device (str, optional): 计算设备 ("cpu" 或 "cuda")。

    Yields:
        Iterator[Dict[str, Any]]: 一个字典，包含一个客户端的上传信息：
            - "gradient" (ParameterListDict): 聚合梯度，即本地模型参数更新。
            - "x" (torch.Tensor): 该客户端在其本地所有轮次中处理过的所有输入样本的拼接。
            - "y" (torch.Tensor): 该客户端在其本地所有轮次中处理过的所有真实标签的拼接。
            - "feature_vector" (torch.Tensor): 该客户端在其本地所有轮次、所有批次后的特征向量的拼接。
            - "x_list" (List[torch.Tensor]): 包含每个本地批次输入数据的列表。
            - "y_list" (List[torch.Tensor]): 包含每个本地批次真实标签的列表。
            - "feature_vector_list" (List[Optional[torch.Tensor]]): 包含每个数据的特征向量的列表。
            - "input_x_num" (int): 输入过模型的样本总数
            - "dY_list" (List[torch.Tensor]): 输入过模型的样本产生的推理梯度
    """
    if dataloader is None:
        data_gen = None
        if dataloader_seed is not None:
            data_gen = torch.Generator().manual_seed(dataloader_seed)
        dataloader = get_dataloader(
            dataset_name, batch_size, shuffle, image_size, train, generator=data_gen
        )
    dataloader = iter(dataloader)
    model = deepcopy(model).cpu()  # 节省显存
    criterion = nn.CrossEntropyLoss()
    ds_len = len(dataloader)
    local_num = math.floor(ds_len / local_steps)  # 客户端数量
    global_params = model.get_parameter()  # 全局模型参数，自定义ParameterListDict类型
    for _ in range(skip_index):
        next(dataloader)
    # 遍历每个客户端
    for local_i in range(local_num):
        local_model = deepcopy(model)  # 本地模型
        local_model = local_model.to(device)
        local_model.train()
        local_opt = optim.SGD(local_model.parameters(), lr=local_lr)  # 本地优化器
        # 存放每轮的特征向量、样本、标签、推理梯度
        x_list = []
        y_list = []
        feature_vector_list = []
        dY_list = []
        # 缓存客户端的本地数据
        batch_datas = []
        for _ in range(local_steps):
            x, y = next(dataloader)
            batch_datas.append((x, y))
            x_list.append(x.clone().detach())
            y_list.append(y.clone().detach())
        # 执行本地轮次
        for ep in range(local_epochs):
            for x, y in batch_datas:
                x, y = x.to(device), y.to(device)
                local_model.zero_grad()
                local_opt.zero_grad()
                y_pred = local_model(x)
                if need_dY:
                    y_pred.retain_grad()
                loss = criterion(y_pred, y)
                loss.backward()
                local_opt.step()
                feature_vector_list.append(local_model.get_feature_vector().cpu())
                if need_dY:
                    dY = y_pred.grad.clone().cpu()
                    dY_list.append(dY)
        local_params = local_model.get_parameter()  # 更新后的参数
        # 计算参数更新。ParameterListDict已重载减法操作
        local_update = (global_params - local_params).to("cpu")

        data = {
            "gradient": local_update,  # 聚合梯度，即参数更新
            "x": torch.cat(x_list, dim=0),
            "y": torch.cat(y_list, dim=0),
            "feature_vector": torch.cat(feature_vector_list, dim=0),
            "x_list": x_list,
            "y_list": y_list,
            "feature_vector_list": feature_vector_list,
            # 输入过模型的样本总数
            "input_x_num": local_steps * batch_size * local_epochs,
        }
        if dY_list:
            data["dY_list"] = dY_list
            data["dY"] = torch.cat(dY_list, dim=0)
        del x, y, y_pred, loss, local_params, local_model, local_opt, batch_datas
        yield data
