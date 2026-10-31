import math
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union
from scipy.optimize import linear_sum_assignment
import warnings
import torch.nn.functional as F

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Importing `peak_signal_noise_ratio` from `torchmetrics.functional` was deprecated",
)

try:
    from torchmetrics.image import peak_signal_noise_ratio
except ImportError:
    from torchmetrics.functional import peak_signal_noise_ratio

from .pytorch_ssim import ssim as ssim_
from .pytorch_lpips import Lpips

_func_mse = nn.MSELoss()
_func_lpips = None


def init_lpips(device):
    """设置设备。必须先调用此函数，才能调用lpips相关的评估"""
    global _func_lpips
    if _func_lpips is None:
        _func_lpips = Lpips()
    _func_lpips.set_device(device)


def mse(A: torch.Tensor, B: torch.Tensor) -> float:
    """计算两个张量A和B之间的均方误差 (MSE)。

    Args:
        A (torch.Tensor): 第一个输入张量。形状为 (n, 3, W, H)
        B (torch.Tensor): 第二个输入张量，应与A具有相同的形状。

    Returns:
        float: A和B之间的MSE值。
    """
    return _func_mse(A, B).item()


def psnr(A: torch.Tensor, B: torch.Tensor) -> float:
    """计算两个张量A和B之间的峰值信噪比 (PSNR)。

    Args:
        A (torch.Tensor): 第一个输入张量。形状为 (n, 3, W, H)
        B (torch.Tensor): 第二个输入张量，应与A具有相同的形状。

    Returns:
        float: A和B之间的PSNR值。
    """
    return peak_signal_noise_ratio(A, B, dim=0, data_range=1.0).item()


def ssim(A: torch.Tensor, B: torch.Tensor) -> float:
    """计算两个张量A和B之间的结构相似性指数 (SSIM)。

    Args:
        A (torch.Tensor): 第一个输入张量。
        B (torch.Tensor): 第二个输入张量，应与A具有相同的形状。

    Returns:
        float: A和B之间的SSIM值。
    """
    return ssim_(A, B).item()


def lpips(A: torch.Tensor, B: torch.Tensor) -> float:
    """计算两个张量A和B之间的感知相似度 (LPIPS, Learned Perceptual Image Patch Similarity) 。

    Args:
        A (torch.Tensor): 第一个输入张量。
        B (torch.Tensor): 第二个输入张量，应与A具有相同的形状。

    Returns:
        float: A和B之间的SSIM值。
    """
    assert _func_lpips is not None, "请先调用 init_lpips(device) 初始化 LPIPS 模型"
    return _func_lpips(A, B).item()


def mse_psnr_ssim_lpips(
    A: torch.Tensor, B: torch.Tensor
) -> Tuple[float, float, float, float]:
    """计算两个张量A和B之间的MSE、PSNR、SSIM、LPIPS。

    Args:
        A (torch.Tensor): 第一个输入张量。
        B (torch.Tensor): 第二个输入张量，应与A具有相同的形状。

    Returns:
        Tuple[float, float, float, float]: 包含MSE、PSNR、SSIM、LPIPS值的元组。
    """
    return mse(A, B), psnr(A, B), ssim(A, B), lpips(A, B)


def cosine_similarity(A: torch.Tensor, B: torch.Tensor) -> Tuple[float, List]:
    """计算两个张量A和B的余弦相似度，考虑样本顺序可能不同。
    配对是唯一的，不允许A中样本被重复使用。
    Args:
        A (torch.Tensor): 形状为 [m, d] 的张量，表示m个样本，每个样本d个参数。
                        - 如果维度大于2，将被重塑为 [m, -1] 进行对比。
        B (torch.Tensor): 形状为 [n, d] 的张量，表示n个样本，每个样本d个参数。
                        - 如果维度大于2，将被重塑为 [n, -1] 进行对比。
    Returns:
        Tuple[avg_similarity, matches]:
            - avg_similarity (float): 基于最佳匹配的平均余弦相似度。
            - matches (List[Tuple[int, int, float]]): 每个样本的匹配结果列表，
              每个元组包含 (A中样本索引, B中样本索引, 对应的余弦相似度)。
              列表按B中样本索引升序排列。
    """
    # 如果不是二维，则转为二维
    if A.ndimension() > 2:
        A = A.view(A.shape[0], -1)
        B = B.view(B.shape[0], -1)
    # 计算余弦相似度矩阵
    cos_sim_matrix = torch.mm(A, B.t()) / (
        torch.norm(A, dim=1).unsqueeze(1) * torch.norm(B, dim=1).unsqueeze(0)
    )
    # 检查并处理 NaN 和 Inf 值
    if torch.isnan(cos_sim_matrix).any():
        cos_sim_matrix = torch.nan_to_num(cos_sim_matrix, nan=0.0)
    if torch.isinf(cos_sim_matrix).any():
        cos_sim_matrix = torch.nan_to_num(cos_sim_matrix, posinf=0.0, neginf=0.0)
    # 将相似度矩阵转换为CPU上的numpy数组
    cos_sim_matrix_np = cos_sim_matrix.cpu().numpy()
    # 使用匈牙利算法求解最优匹配
    row_ind, col_ind = linear_sum_assignment(-cos_sim_matrix_np)
    # 计算平均相似度
    avg_cos = float(cos_sim_matrix_np[row_ind, col_ind].sum() / len(A))
    # 构建匹配结果
    matches = [(i, j, cos_sim_matrix_np[i, j]) for i, j in zip(row_ind, col_ind)]
    # 整理为按 B 升序
    matches = sorted(matches, key=lambda x: x[1])
    return avg_cos, matches


def cosine_similarity_allow_repeat(
    A: torch.Tensor, B: torch.Tensor
) -> Tuple[float, List[Tuple[int, int, float]]]:
    """
    为B中的每一个样本，从A中寻找一个最匹配的样本（允许A中样本被重复使用）。
    返回这些最佳匹配的平均余弦相似度，以及具体的匹配列表。

    Args:
        A (torch.Tensor): 形状为 [m, d] 的张量，表示m个候选样本。
        B (torch.Tensor): 形状为 [n, d] 的张量，表示n个目标样本。

    Returns:
        Tuple[float, List]: 一个元组，包含：
            - avg_cos (float): B中每个样本与其最佳A样本配对的平均余弦相似度。
            - matches (List): 匹配详情列表。每个元素是 (a_idx, b_idx, similarity)，
                              表示B[b_idx]的最佳匹配是A[a_idx]，其相似度为similarity。
    """
    # 1. 数据预处理
    if A.ndimension() > 2:
        A = A.view(A.shape[0], -1)
    if B.ndimension() > 2:
        B = B.view(B.shape[0], -1)

    # 2. 计算余弦相似度矩阵
    epsilon = 1e-8
    norm_A = torch.norm(A, p=2, dim=1).unsqueeze(1)
    norm_B = torch.norm(B, p=2, dim=1).unsqueeze(0)
    dot_product = torch.mm(A, B.t())
    cos_sim_matrix = dot_product / (norm_A * norm_B + epsilon)

    # 3. 为B的每个样本寻找最佳匹配
    # torch.max会同时返回最大值和最大值所在的索引
    best_matches_for_b = torch.max(cos_sim_matrix, dim=0)

    # 分别获取相似度值 和 对应的A中的索引
    best_similarities = best_matches_for_b.values  # 形状: [n]
    best_A_indices = best_matches_for_b.indices  # 形状: [n]

    # 4. 计算平均相似度
    avg_cos = torch.mean(best_similarities)

    # 5. 构建匹配结果列表
    # B的索引就是从 0 到 n-1
    b_indices = range(B.shape[0])

    # 将PyTorch张量转换为Python列表以便打包
    best_A_indices_list = best_A_indices.cpu().tolist()
    best_similarities_list = best_similarities.cpu().tolist()

    # 使用zip将A的索引、B的索引和相似度分数打包在一起
    matches = list(zip(best_A_indices_list, b_indices, best_similarities_list))

    return avg_cos.item(), matches


def reorder_tensor(
    A: torch.Tensor,
    matches: List[Tuple[int, int, float]],
    B: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """根据 cosine_similarity 匹配结果调整张量A的顺序，并可选地调整B。
    如果提供了B，则函数会根据匹配同时调整A和B，并确保只保留匹配上的元素。

    Args:
        A (torch.Tensor): 形状为 [m, d] 的张量，表示m个样本，每个样本d个参数。
        B (torch.Tensor): 可选。形状为 [n, d] 的张量，表示n个样本，每个样本d个参数。
        - 如果传入 B ，且与 A 长度不一致，而删除二者的未匹配的元素。

    Returns:
        rA (torch.Tensor): 调整后的 A
        rB (torch.Tensor): 调整后的 B
    """

    # 提取匹配的索引
    a_indices, b_indices, _ = zip(*matches)
    # 无需调整形状
    if a_indices == b_indices:
        if B is None:
            print("==", a_indices, b_indices)
            return A
        if A.shape == B.shape:
            return A, B

    # 根据匹配结果调整 A 的顺序
    reordered_A = A[list(a_indices)]
    if B is None:
        return reordered_A
    # 调整 B 的顺序
    reordered_B = B[list(b_indices)]
    return reordered_A, reordered_B


def similarity_reorder(
    A: torch.Tensor, B: torch.Tensor, allow_repeat=False
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """计算两个张量A和B的余弦相似度，并调整其顺序。

    Args:
        A (torch.Tensor): 形状为 [m, d] 的张量，表示m个样本，每个样本d个参数。
        B (torch.Tensor): 形状为 [n, d] 的张量，表示n个样本，每个样本d个参数。
        allow_repeat (bool): 是否允许重复选择A中的样本来给B配对。

    Returns:
        rA (torch.Tensor): 调整后的 A
        rB (torch.Tensor): 调整后的 B
        avg_cos (float): 计算得到的平均余弦相似度。
    """
    if allow_repeat:
        avg_cos, matches = cosine_similarity_allow_repeat(A, B)
    else:
        avg_cos, matches = cosine_similarity(A, B)
    rA, rB = reorder_tensor(A, matches, B)
    return rA, rB, avg_cos


def reorder_mse_psnr_ssim_lpips(
    A: torch.Tensor, B: torch.Tensor, allow_repeat=False
) -> Tuple[float, float, float, float, float, torch.Tensor, torch.Tensor]:
    """计算两个张量A和B的余弦相似度，调整其顺序以最大化匹配，
    然后计算调整后张量的MSE、PSNR、SSIM、LPIPS。

    Args:
        A (torch.Tensor): 形状为 [m, d] 的张量，表示m个样本，每个样本d个参数。
        B (torch.Tensor): 可选。形状为 [n, d] 的张量，表示n个样本，每个样本d个参数。
        allow_repeat (bool): 是否允许重复选择A中的样本来给B配对。

    Returns:
        Tuple[float, float, float, float, torch.Tensor, torch.Tensor]:
            - mse (float): 调整后张量之间的均方误差。
            - psnr (float): 调整后张量之间的峰值信噪比。
            - ssim (float): 调整后张量之间的结构相似性指数。
            - lpips (float): 调整后张量之间的感知相似度指数。
            - avg_cos (float): 用于重排的平均余弦相似度。
            - rA (torch.Tensor): 调整顺序并筛选后的A。
            - rB (torch.Tensor): 调整顺序并筛选后的B。
    """
    rA, rB, avg_cos = similarity_reorder(A, B, allow_repeat)
    mse, psnr, ssim, lpips = mse_psnr_ssim_lpips(rA, rB)
    return mse, psnr, ssim, lpips, avg_cos, rA, rB


def analyze_experiment_data(data_list: list[dict], calculate_std: bool = True) -> dict:
    """
    分析实验数据列表，计算每个数据项的均值，并可选择性地计算标准差。

    Args:
        data_list (list[dict]): 包含实验数据的列表。
        calculate_std (bool): 是否计算标准差。

    Returns:
        dict: 一个包含结果的有序字典。
    """
    if not data_list:
        return {}

    # 使用第一个字典的键来保证顺序
    all_keys = list(data_list[0].keys())
    result = {}

    for key in all_keys:
        valid_nums = []
        invalid_vals = []

        # 1. 分离合法与非法数据
        for item in data_list:
            value = item.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                valid_nums.append(value)
            else:
                invalid_vals.append(value)

        # 2. 根据分离结果进行统计
        # 情况一：存在合法数值
        if valid_nums:
            # 计算均值
            mean = sum(valid_nums) / len(valid_nums)
            result[key] = mean
            # 如果需要，计算标准差
            if calculate_std:
                if len(valid_nums) > 1:
                    variance = sum([(x - mean) ** 2 for x in valid_nums]) / len(
                        valid_nums
                    )
                    std_dev = math.sqrt(variance)
                else:
                    std_dev = 0.0  # 只有一个合法数值时，标准差为0
                result[f"{key}_std"] = std_dev

        # 情况二：全部为非法数值
        else:
            std_dev = 0.0
            # 2.1 如果所有非法值都相同
            if len(set(str(v) for v in invalid_vals)) == 1:
                mean = invalid_vals[0]
            # 2.2 如果存在不同的非法值
            else:
                # 获取所有非法值的类型或字符串表示
                error_types = set()
                for v in invalid_vals:
                    if isinstance(v, str):
                        error_types.add(v)
                    elif v is None:
                        error_types.add("None")
                    else:  # inf, -inf, nan
                        error_types.add(str(v))
                mean = ",".join(sorted(list(error_types)))
                mean += f"({type(invalid_vals[0])})"

            result[key] = mean
            if calculate_std:
                result[f"{key}_std"] = std_dev

    return dict(result)


def compute_label_acc(y_true, y_fake):
    """度量标签的匹配数目
    https://github.com/zhaohuali/E2EGI/blob/e99d601fa610c5f5b089ba900f93161c6955f562/kernels/utils.py#L91
    """

    y_true_sort = y_true.view(
        -1,
    ).sort()[0]
    y_fake_sort = y_fake.view(
        -1,
    ).sort()[0]

    i = 0
    j = 0
    n_true = len(y_true_sort)
    n_fake = len(y_fake_sort)
    n_correct = 0

    while i < n_true and j < n_fake:

        if y_true_sort[i] == y_fake_sort[j]:
            n_correct += 1
            i += 1
            j += 1
        elif y_true_sort[i] > y_fake_sort[j]:
            j += 1
        elif y_true_sort[i] < y_fake_sort[j]:
            i += 1
    return n_correct, n_correct / n_true


def calculate_batch_f1_score(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int
):
    """
    计算一个批次中预测标签和真实标签（可重复）的F1分数。
    该方法将标签视为多重集（multiset）进行比较。

    Args:
        y_true (torch.Tensor): 真实标签的一维张量。
        y_pred (torch.Tensor): 预测标签的一维张量。
        num_classes (int): 数据集中的类别总数。

    Returns:
        dict: 包含 'f1', 'precision', 'recall' 的字典。
    """
    # 使用bincount高效地统计每个类别的出现次数
    true_counts = torch.bincount(y_true, minlength=num_classes)
    pred_counts = torch.bincount(y_pred, minlength=num_classes)

    # TP (真阳性) 是两个多重集的交集大小
    # 对于每个类别，交集大小是真实数量和预测数量中的较小值
    tp_counts = torch.min(true_counts, pred_counts)
    tp = tp_counts.sum().item()

    # FP (假阳性) = 总预测数 - TP
    fp = y_pred.numel() - tp
    # FN (假阴性) = 总真实数 - TP
    fn = y_true.numel() - tp

    # 计算 Precision, Recall, F1
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    return f1, precision, recall
    # return {"f1": f1, "precision": precision, "recall": recall}


def analyze_batch_properties(
    y_pred_logits: torch.Tensor, dW: torch.Tensor, y_true: torch.Tensor
):
    """
    分析一个批次的模型预测和梯度相关的统计性质。

    Args:
        y_pred_logits (torch.Tensor): 模型的原始输出 (logits)，形状为 (B, N)。
        dW (torch.Tensor): 最后一层FC层的权重梯度，形状为 (N, K) 或 (K, N)。
        y_true (torch.Tensor): 真实标签的一维张量，形状为 (B,)。

    Returns:
        dict: 包含 'entropy', 'separation_gap', 'top_k_hit_rate' 的字典。
    """
    num_classes = y_pred_logits.shape[1]
    device = y_pred_logits.device

    # ---- 指标 1: 模型预测的熵 ----
    # 使用softmax将logits转为概率
    probs = F.softmax(y_pred_logits, dim=1)
    # 计算每个样本的熵，然后求批次平均值
    eps = 1e-9
    entropy_per_sample = -torch.sum(probs * torch.log2(probs + eps), dim=1)
    avg_entropy = entropy_per_sample.mean().item()

    # ---- 聚合梯度 ----
    # dW 的形状是 (num_classes, features)，按特征维度求和
    if dW.shape[0] == num_classes:
        g_b = dW.sum(dim=1)
    else:
        g_b = dW.sum(dim=0)

    # ---- 提取批次内的唯一真实标签 ----
    unique_true_labels = torch.unique(y_true)

    # ---- 指标 2: 批次梯度分离差距 ----
    # 创建一个布尔掩码，标记哪些是真实类别
    true_class_mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
    true_class_mask[unique_true_labels] = True

    # 检查是否存在非真实标签类别
    if (~true_class_mask).any():
        # 真实标签（应该为负梯度）的最大值
        g_true_worst = g_b[true_class_mask].max().item()
        # 错误标签（应该为正梯度）的最小值
        g_false_best = g_b[~true_class_mask].min().item()
        separation_gap = g_false_best - g_true_worst
    else:
        # 如果批次中包含了所有类别，则分离差距无意义
        separation_gap = float("nan")

    # ---- 指标 3: Top-K 命中率 ----
    k = len(unique_true_labels)
    if k > 0:
        # 找到梯度最小（负）的 k 个类别的索引
        pred_top_k_indices = torch.argsort(g_b)[:k]

        # 计算预测的Top-K集合与真实标签集合的交集大小
        # 为了高效计算交集，转为set操作
        set_true = set(unique_true_labels.tolist())
        set_pred_top_k = set(pred_top_k_indices.tolist())
        hits = len(set_true.intersection(set_pred_top_k))

        top_k_hit_rate = hits / k
    else:
        # 如果批次为空，则命中率无意义
        top_k_hit_rate = float("nan")

    return {
        "entropy": avg_entropy,
        "separation_gap": separation_gap,
        "top_k_hit_rate": top_k_hit_rate,
    }


def calculate_threshold_matching_rate(
    A: torch.Tensor, B: torch.Tensor, main_metric: str, threshold: float
):
    """
    计算样本数据(A)对目标数据(B)的阈值匹配率。

    对于B中的每一个目标，此函数会在A中寻找一个最佳匹配的样本。
    如果这个最佳匹配的评估值满足阈值条件，则认为该目标被成功匹配。
    函数随后会计算所有成功匹配对的各项评估指标，并返回详细的统计结果。

    Args:
        A (torch.Tensor): 样本数据。形状为 (n, 3, W, H)，代表n个样本。
        B (torch.Tensor): 目标数据。形状为 (m, 3, W, H)，代表m个目标。
        main_metric (str): 用于寻找最佳匹配的主评估函数名称。
                           可选值为 'mse', 'psnr', 'ssim', 'lpips'。
        threshold (float): 用于判断是否匹配的阈值。

    Returns:
        一个字典，包含以下英文键：
        - 'match_count' (int): 成功匹配的目标数量。
        - 'match_rate' (float): 匹配数 / 总目标数。
        - 'avg_mse' (float): 所有匹配对的平均MSE。
        - 'avg_psnr' (float): 所有匹配对的平均PSNR。
        - 'avg_ssim' (float): 所有匹配对的平均SSIM。
        - 'avg_lpips' (float): 所有匹配对的平均LPIPS。
        - 'matched_samples' (torch.Tensor): 匹配的样本张量，形状 (k, 3, W, H)，k为匹配数。
        - 'matched_targets' (torch.Tensor): 对应的目标张量，形状 (k, 3, W, H)，顺序与样本一致。
    """
    n, _, W_a, H_a = A.shape
    m, _, W_b, H_b = B.shape

    assert (W_a, H_a) == (W_b, H_b), "样本和目标的图像尺寸必须一致。"
    if n == 0 or m == 0:
        print("警告: 输入张量 A 或 B 为空。")
        return {
            "match_count": 0,
            "match_rate": 0.0,
            "avg_mse": 0.0,
            "avg_psnr": 0.0,
            "avg_ssim": 0.0,
            "avg_lpips": 0.0,
            "matched_samples": torch.empty(0, 3, W_a, H_a, device=A.device),
            "matched_targets": torch.empty(0, 3, W_b, H_b, device=B.device),
        }

    # 定义评估指标的属性
    metrics_config = {
        "mse": {
            "func": mse,
            "compare": lambda s, best: s < best,
            "initial": float("inf"),
            "check": lambda s, t: s <= t,
        },
        "lpips": {
            "func": lpips,
            "compare": lambda s, best: s < best,
            "initial": float("inf"),
            "check": lambda s, t: s <= t,
        },
        "psnr": {
            "func": psnr,
            "compare": lambda s, best: s > best,
            "initial": float("-inf"),
            "check": lambda s, t: s >= t,
        },
        "ssim": {
            "func": ssim,
            "compare": lambda s, best: s > best,
            "initial": float("-inf"),
            "check": lambda s, t: s >= t,
        },
    }

    assert (
        main_metric in metrics_config
    ), f"无效的主评估指标: {main_metric}。请从 {list(metrics_config.keys())} 中选择。"

    # 确保 LPIPS 已初始化 (如果需要)
    try:
        assert _func_lpips is not None
    except (NameError, AssertionError):
        print("警告: LPIPS 模型未初始化。如果需要计算LPIPS，请先调用 init_lpips()。")

    # 1. 为B中的每个目标找到A中的最佳匹配
    best_matches_info = []
    metric_info = metrics_config[main_metric]
    main_eval_func = metric_info["func"]
    is_better = metric_info["compare"]

    for j in range(m):
        target_b = B[j : j + 1]
        best_score = metric_info["initial"]
        best_sample_idx = -1

        for i in range(n):
            sample_a = A[i : i + 1]
            score = main_eval_func(sample_a, target_b)
            if is_better(score, best_score):
                best_score = score
                best_sample_idx = i

        if best_sample_idx != -1:
            best_matches_info.append(
                {"target_idx": j, "sample_idx": best_sample_idx, "score": best_score}
            )

    # 2. 根据阈值筛选出成功的匹配
    threshold_check = metric_info["check"]
    final_matches = [
        m_info
        for m_info in best_matches_info
        if threshold_check(m_info["score"], threshold)
    ]

    # 3. 计算统计结果
    match_count = len(final_matches)
    match_rate = match_count / m if m > 0 else 0.0

    if match_count == 0:
        return {
            "match_count": 0,
            "match_rate": match_rate,
            "avg_mse": 0.0,
            "avg_psnr": 0.0,
            "avg_ssim": 0.0,
            "avg_lpips": 0.0,
            "matched_samples": torch.empty(0, 3, W_a, H_a, device=A.device),
            "matched_targets": torch.empty(0, 3, W_b, H_b, device=B.device),
        }

    # 4. 收集匹配的张量
    matched_sample_indices = [match["sample_idx"] for match in final_matches]
    matched_target_indices = [match["target_idx"] for match in final_matches]

    matched_samples = A[matched_sample_indices]
    matched_targets = B[matched_target_indices]

    # 5. 计算所有匹配对的各项指标均值
    all_funcs = {"mse": mse, "psnr": psnr, "ssim": ssim, "lpips": lpips}
    # avg_metrics = {}
    # for name, func in all_funcs.items():
    #     avg_metrics[f"avg_{name}"] = func(matched_samples, matched_targets)

    totals = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    for i in range(match_count):
        sample = matched_samples[i : i + 1]
        target = matched_targets[i : i + 1]
        for name, func in all_funcs.items():
            try:
                totals[name] += func(sample, target)
            except Exception as e:
                # 如果某个指标（如LPIPS）未初始化，则跳过
                print(f"计算 '{name}' 时出错: {e}")
    avg_metrics = {f"avg_{name}": val / match_count for name, val in totals.items()}

    # 6. 准备并返回最终结果
    result = {
        "match_count": match_count,
        "match_rate": match_rate,
        "matched_samples": matched_samples,
        "matched_targets": matched_targets,
    }
    result.update(avg_metrics)

    return result
