import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms, datasets
import random
from typing import List, Dict, Any, Tuple, Optional, Union


"""
imagenet
caltech256
cifar10
cifar100
mnist
"""

from utils import logger

DATASETS_DIR = "./data/datasets"


def get_dataset_info(name: str) -> Dict[str, Any]:
    """
    获取指定数据集的信息字典

    Args:
        name (str): 数据集的名称。

    Returns:
        Optional[Dict[str, Any]]: 包含数据集信息的字典，如果数据集名称不受支持则返回 None。
                                  例如：{"num_classes": 10}。
    """
    infos = {
        "imagenet": {
            "num_classes": 1000,
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        },
        "caltech256": {
            "num_classes": 257,  # 实际类别
            "mean": (0.542, 0.517, 0.492),
            "std": (0.266, 0.262, 0.274),
        },
        "cifar10": {
            "num_classes": 10,
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2470, 0.2435, 0.2616),
        },
        "cifar100": {
            "num_classes": 100,
            "mean": (0.5071, 0.4867, 0.4408),
            "std": (0.2675, 0.2565, 0.2761),
        },
        "mnist": {
            "num_classes": 10,
            "mean": (0.1307, 0.1307, 0.1307),
            "std": (0.3081, 0.3081, 0.3081),
        },
    }
    return infos[name]


def get_dataset(
    name: str,
    image_size: int,
    train: bool,
    normalize=False,
    data_enhancement=False,
    transform=None,
) -> Dataset:
    """
    获取指定名称和配置的数据集。

    Args:
        name (str): 数据集的名称。支持的名称有："imagenet", "caltech256", "cifar10", "cifar100", "mnist"。
        image_size (int): 图像的目标尺寸（正方形）。
        train (bool): 如果为 True，则加载训练集；否则加载测试集/验证集。
        normalize (bool): 如果为 True，则标准化数据
        data_enhancement (bool): 如果为True，启用数据增强
        transform (bool): 自定义变换处理

    Returns:
        Dataset: 加载并经过预处理的 PyTorch 数据集。

    Raises:
        ValueError: 如果指定的数据集名称不受支持。
    """
    if transform is not None:
        _transform = transform
    else:
        dataset_info = get_dataset_info(name)

        if data_enhancement:
            transform_list = [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                # transforms.Resize([image_size, image_size]),  # 被RandomResizedCrop替代
                transforms.ToTensor(),
            ]
        else:
            transform_list = [
                transforms.Resize([image_size, image_size]),
                transforms.ToTensor(),
            ]
        if normalize:
            transform_list.append(
                transforms.Normalize(dataset_info["mean"], dataset_info["std"])
            )
        _transform = transforms.Compose(transform_list)
    if name == "imagenet":
        split = "train" if train else "val"
        dataset = datasets.ImageNet(
            f"{DATASETS_DIR}/imagenet/",
            transform=_transform,
            split=split,
        )
    elif name == "caltech256":
        if transform is None:
            transform_list.insert(
                1,
                transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
            )
            _transform = transforms.Compose(transform_list)
        dataset = datasets.Caltech256(
            f"{DATASETS_DIR}/caltech256",
            transform=_transform,
            download=True,
        )
        # 排除不存在文件的index
        dataset.index = (
            dataset.index[:6307] + dataset.index[6308:22619] + dataset.index[22620:]
        )
        dataset.y = dataset.y[:6307] + dataset.y[6308:22619] + dataset.y[22620:]
    elif name == "cifar10":
        dataset = datasets.CIFAR10(
            f"{DATASETS_DIR}/cifar10",
            transform=_transform,
            download=True,
            train=train,
        )
    elif name == "cifar100":
        dataset = datasets.CIFAR100(
            f"{DATASETS_DIR}/cifar100",
            transform=_transform,
            download=True,
            train=train,
        )
    elif name == "mnist":
        if transform is None:
            # 转换为三通道
            transform_list.insert(1, transforms.Lambda(lambda x: x.convert("RGB")))
            _transform = transforms.Compose(transform_list)
        dataset = datasets.MNIST(
            f"{DATASETS_DIR}/mnist",
            transform=_transform,
            download=True,
            train=train,
        )
    else:
        raise ValueError(f"Dataset '{name}' is not supported.")
    return dataset


def get_dataloader(
    name: str,
    batch_size: int,
    shuffle=False,
    image_size=224,
    train=False,
    normalize=False,
    data_enhancement=False,
    transform=None,
    generator=None,
) -> DataLoader:
    """
    获取指定数据集的数据加载器。

    Args:
        name (str): 数据集的名称。
        batch_size (int): 每个批次中的样本数量。
        shuffle (bool, optional): 是否在每个 epoch 开始时打乱数据。默认为 False。
        image_size (int, optional): 图像的目标尺寸。默认为 224。
        train (bool, optional): 是否加载训练集。默认为 False。
        normalize (bool): 如果为 True，则标准化数据
        data_enhancement (bool): 如果为True，启用数据增强
        transform (bool): 自定义变换处理
        generator (bool): 自定义种子生成器

    Returns:
        DataLoader: PyTorch 数据加载器。
    """
    dataset = get_dataset(
        name, image_size, train, normalize, data_enhancement, transform
    )
    data_loader = DataLoader(
        dataset, batch_size, shuffle=shuffle, drop_last=True, generator=generator
    )
    return data_loader


def get_repetition_dataloader(
    name: str,
    batch_size: int,
    shuffle=False,  # 如果为 True ，则主元素随机
    image_size=224,
    train=False,
    rep_rate=0.1,  # 标签重复率
    **kwargs,
):
    """
    获取一个数据加载器，其中每个批次内包含指定比例的重复标签样本。

    Args:
        name (str): 数据集的名称。
        batch_size (int): 每个批次中的样本数量。
        shuffle (bool, optional): 如果为 True，则主导重复标签的选择和非重复标签的选择会随机化。
                                 原始数据集的样本顺序也会被打乱。默认为 False。
        image_size (int, optional): 图像的目标尺寸。默认为 224。
        train (bool, optional): 是否加载训练集。默认为 False。
        rep_rate (float, optional): 标签重复率。取值范围 [0, 1]。
                                    表示每个批次中具有相同（主导）标签的样本所占的比例。
                                    默认为 0.1。

    Returns:
        DataLoader: 按照指定重复率构造的 PyTorch 数据加载器。
                    如果无法满足重复率要求（例如，某个类别样本不足），
                    加载器可能提前结束或批次大小可能不完全符合预期。
    """

    # 标签重复个数
    rep_num = round(batch_size * rep_rate)
    rep_num = max(0, min(rep_num, batch_size))
    no_rep_num = batch_size - rep_num

    # 取原数据集
    dataset = get_dataset(name, image_size, train)
    dataset_info = get_dataset_info(name)
    num_classes = dataset_info["num_classes"]  # 标签种类数

    # 将原数据集中的样本，打乱顺序后装进对应标签的标签桶
    buckets = [[] for _ in range(num_classes)]
    if shuffle:
        indices = torch.randperm(len(dataset))
    else:
        indices = torch.arange(len(dataset))
    for i in indices:
        d = dataset[i]
        buckets[d[1]].append(i)

    new_order = []  # 新的数据集的下标顺序

    while True:
        if rep_num > 0:
            # 1. 从标签桶中取一个剩余样本数 >= rep_num 的标签，作为主标签
            main_labels = [
                i for i, bucket in enumerate(buckets) if len(bucket) >= rep_num
            ]
            if not main_labels:
                # logger.warning(f"警告：不存在剩余样本数 >= {rep_num} 的标签")
                break
            if shuffle:
                main_label = random.choice(main_labels)
            else:
                main_label = main_labels[0]

            # 2. 将 rep_num 个主标签的样本的下标装进 new_order
            main_samples = buckets[main_label][:rep_num]
            new_order.extend(main_samples)
            buckets[main_label] = buckets[main_label][rep_num:]
        else:
            main_label = num_classes + 2

        # 3. 从标签桶中取 no_rep_num 个剩余样本数 >0 的标签，作为次标签
        secondary_labels = [i for i, bucket in enumerate(buckets) if len(bucket) > 0]
        if main_label in secondary_labels:
            secondary_labels.remove(main_label)
        if len(secondary_labels) < no_rep_num:
            break
        if shuffle:
            chosen_secondary_labels = random.sample(secondary_labels, no_rep_num)
        else:
            chosen_secondary_labels = secondary_labels[:no_rep_num]

        # 4. 每个次标签，各取1个样本的下标装进 new_order
        for label in chosen_secondary_labels:
            new_order.append(buckets[label].pop(0))

    # 将 new_order 转换为 new_dataset
    new_dataset = Subset(dataset, new_order)

    dataLoader = DataLoader(new_dataset, batch_size, shuffle=False, drop_last=True)
    return dataLoader


def get_filtered_dataloader(
    name: str,
    batch_size: int,
    shuffle=False,
    image_size=224,
    train=False,
    blacklist=[],  # 标签黑名单
    whitelist=[],  # 标签白名单
):
    """
    获取一个根据标签黑名单或白名单过滤后的数据加载器。

    Args:
        name (str): 数据集的名称。
        batch_size (int): 每个批次中的样本数量。
        shuffle (bool, optional): 是否在每个 epoch 开始时打乱数据。默认为 False。
        image_size (int, optional): 图像的目标尺寸。默认为 224。
        train (bool, optional): 是否加载训练集。默认为 False。
        blacklist (Optional[List[int]], optional): 不应包含在数据集中的标签列表。默认为 None。
        whitelist (Optional[List[int]], optional): 只应包含在数据集中的标签列表。默认为 None。
        num_workers (int, optional): 用于数据加载的子进程数量。默认为 0。
        pin_memory (bool, optional): 如果为 True，数据加载器会将张量复制到 CUDA 固定内存中再返回。默认为 False。

    Returns:
        DataLoader: 经过标签过滤的 PyTorch 数据加载器。
    """

    if not blacklist and not whitelist:
        return get_dataloader(name, batch_size, shuffle, image_size, train)

    dataset = get_dataset(name, image_size, train)
    # 过滤数据集
    filtered_indices = []
    for i in range(len(dataset)):
        label = dataset[i][1]  # dataset[i] 返回 (image, label) 的元组
        if (whitelist and label not in whitelist) or (label in blacklist):
            continue
        filtered_indices.append(i)

    # 创建过滤后的数据集
    new_dataset = Subset(dataset, filtered_indices)

    # 创建数据加载器
    dataLoader = DataLoader(new_dataset, batch_size, shuffle=shuffle, drop_last=True)
    return dataLoader
