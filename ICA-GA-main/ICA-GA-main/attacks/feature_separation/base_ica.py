import torch
from typing import Union, Optional
from torch import Tensor
from abc import abstractmethod

from .base_separator import BaseSeparator
from utils import logger


class BaseICA(BaseSeparator):
    def __init__(
        self,
        w_init_type: str = "eye",
        device: torch.device = torch.device("cpu"),
    ):
        self.W = None  # 缓存解混矩阵
        self.device = device
        self.w_init_type = w_init_type
        self.eps = torch.tensor(1e-10, device=self.device)
        self.w_hook_func = None  # 在迭代过程中获取解混矩阵w的函数
        self.epochs = 0  # 缓存实际迭代次数

    @abstractmethod
    def ica(
        self,
        X: torch.Tensor,  # 混合信号矩阵
        S_size: int,  # 源信号个数
        S: torch.Tensor,  # 源信号（测试用）
        w_init=Union[str, Tensor],  # 初始解混矩阵
    ) -> torch.Tensor:  # 返回重建信号
        pass

    def run(
        self,
        grad_w: torch.Tensor,
        grad_b: torch.Tensor,
        y: torch.Tensor,
        real_feature: torch.Tensor,
        S_size: int = 0,
    ) -> torch.Tensor:
        if S_size <= 0:
            S_size = y.shape[0]
        return self.ica(grad_w, S_size, real_feature)

    def _get_w_init(self, w_init: Optional[Union[str, Tensor]], n_components: int):
        """处理并返回初始解混矩阵。

        Args:
            w_init: 允许使用字符串指定类型，或直接传入Tensor作为初始矩阵，或None。
                - str: "eye" 单位矩阵， "random" 随机， "random_q" 随机正交。
                - Tensor: 形状为 [n_components,n_components] 的方阵，直接作为初始解混矩阵
                - None: 使用 self.w_init_type
            n_components (int): 独立成分个数。
        """
        if w_init is None:
            w_init = self.w_init_type
        if isinstance(w_init, str):
            if w_init == "eye":
                # 单位矩阵，对角线为1，其余为0
                return torch.eye(n_components, device=self.device)
            elif w_init == "random" or w_init == "random_q":
                # 随机的高斯矩阵
                random_w = torch.randn(n_components, n_components, device=self.device)
                if w_init == "random_q":
                    # 进行QR分解，取其Q矩阵部分。Q矩阵保证是正交的，符合白化后的理论要求。
                    q_w, _ = torch.linalg.qr(random_w)
                    return q_w
                else:
                    return random_w
            else:
                raise ValueError(
                    f'w_init 只能指定类型为 "eye", "random", "random_q" ，不允许 "{w_init}"'
                )
        elif isinstance(w_init, Tensor):
            if w_init.shape != (n_components, n_components):
                raise ValueError(
                    f"w_init 若为Tensor，其形状必须为 [{n_components}, {n_components}]，"
                    f"但传入的形状为 {w_init.shape}"
                )
            W = w_init.to(self.device)
            return W
        else:
            raise TypeError(
                f"w_init 只能使用str指定类型，或传入Tensor。不允许 {type(w_init)}"
            )

    def _preprocessing(self, X: Tensor, n_components: int):
        """对混合信号矩阵进行中心化、白化预处理。

        Args:
            X (Tensor): 混合信号矩阵，形状为 (n_features, n_samples)，即 (混合信号数量, 单个信号长度)
            n_components (int): 独立成分个数。不能大于 n_samples 及 n_features

        Returns:
            X_centered (Tensor): 中心化混合信号矩阵，形状为 (n_features, n_samples)。
            K (Tensor): 白化矩阵，形状为 (n_components, n_features)。
            X_w (Tensor): 白化后的数据矩阵，形状为 (n_components, n_samples)。
                          其行向量近似正交且具有单位方差。 X_w = K @ X_centered_T
            X_mean (Tensor): 每个特征的均值
            n_components (int): 允许的独立成分个数
        """
        if X.device != self.device:
            X = X.to(self.device)
        # 获取原始数据的形状，检查 n_components
        n_features, n_samples = X.shape
        if n_components > min(n_samples, n_features):
            n_components = min(n_samples, n_features)
            logger.warning(
                f"n_components is too large: it will be set to {n_components}"
            )

        X_mean = torch.mean(X, dim=1, keepdim=True)  # 均值
        X_centered = X - X_mean  # 中心化

        # 计算协方差矩阵 C
        C = torch.matmul(X_centered, X_centered.T) / (X_centered.shape[1] - 1)
        # 对 C 特征值分解
        eig_vals, eig_vecs = torch.linalg.eig(C)
        # 提取前 n_components 个最大的特征值，获取特征向量矩阵 U 和对应的特征值 lamb
        topk_indices = torch.topk(eig_vals.float().abs(), n_components)[1]
        U = eig_vecs.float()
        lamb = eig_vals.float()[topk_indices].abs()
        # 特征值的逆平方根，并构造对角矩阵 lamb_inv_sqrt
        lamb_inv_sqrt = torch.diag(1 / (torch.sqrt(lamb) + self.eps)).float()
        # 计算白化矩阵 K
        K = (lamb_inv_sqrt @ U.T[topk_indices]).float()
        # 对输入数据进行白化处理
        X_w = K @ X_centered
        return X_centered, K, X_w, X_mean, n_components

    def _preprocessing_cpu(self, X: Tensor, n_components: int):
        """对混合信号矩阵进行中心化、白化预处理。在CPU上进行以确保精度。

        Args:
            X (Tensor): 混合信号矩阵，形状为 (n_features, n_samples)，即 (混合信号数量, 单个信号长度)
            n_components (int): 独立成分个数。不能大于 n_samples 及 n_features

        Returns:
            X_centered (Tensor): 中心化混合信号矩阵，形状为 (n_features, n_samples)。
            K (Tensor): 白化矩阵，形状为 (n_components, n_features)。
            X_w (Tensor): 白化后的数据矩阵，形状为 (n_components, n_samples)。
                          其行向量近似正交且具有单位方差。 X_w = K @ X_centered_T
            X_mean (Tensor): 每个特征的均值
            n_components (int): 允许的独立成分个数
        """
        X = X.cpu()
        eps = self.eps.cpu()
        # 获取原始数据的形状，检查 n_components
        n_features, n_samples = X.shape
        if n_components > min(n_samples, n_features):
            n_components = min(n_samples, n_features)
            logger.warning(
                f"n_components is too large: it will be set to {n_components}"
            )

        X_mean = torch.mean(X, dim=1, keepdim=True)  # 均值
        X_centered = X - X_mean  # 中心化

        # 计算协方差矩阵 C
        C = torch.matmul(X_centered, X_centered.T) / (X_centered.shape[1] - 1)
        # 对 C 特征值分解
        eig_vals, eig_vecs = torch.linalg.eig(C)
        # 提取前 n_components 个最大的特征值，获取特征向量矩阵 U 和对应的特征值 lamb
        topk_indices = torch.topk(eig_vals.float().abs(), n_components)[1]
        U = eig_vecs.float()
        lamb = eig_vals.float()[topk_indices].abs()
        # 特征值的逆平方根，并构造对角矩阵 lamb_inv_sqrt
        lamb_inv_sqrt = torch.diag(1 / (torch.sqrt(lamb) + eps)).float()
        # 计算白化矩阵 K
        K = (lamb_inv_sqrt @ U.T[topk_indices]).float()
        # 对输入数据进行白化处理
        X_w = K @ X_centered
        return (
            X_centered.to(self.device),
            K.to(self.device),
            X_w.to(self.device),
            X_mean.to(self.device),
            n_components,
        )

    def _get_recover_S(
        self,
        W: torch.Tensor,
        n_components: int,
        X_w: torch.Tensor,
        X_mean: torch.Tensor,
        K: torch.Tensor,
    ):
        """获取还原信号。

        Args:
            W (Tensor): 解混矩阵
            n_components (int): 独立成分个数
            X_w (Tensor): 白化混合信号
            X_mean (Tensor): 混合信号均值
            V (Tensor): 白化矩阵

        Returns:
            Tensor: 还原信号
        """
        with torch.no_grad():
            # 1. 归一化解混矩阵
            W_norm = W / (W.norm(dim=-1, keepdim=True) + self.eps)
            # 2. 初步计算解混信号
            estimated_components = W_norm @ X_w
            # 3. 逆中心化处理：调整信号的均值，使其与源信号的均值一致
            mean_adjustment = W_norm @ K @ X_mean
            re_S = estimated_components + mean_adjustment
            # 4. 恢复输入形状
            re_S = re_S.detach().view([n_components, -1])
            # 5. 取绝对值
            re_S = re_S.abs()
        return re_S

    def register_w_hook(self, w_hook_func):
        """注册一个钩子，用于在迭代过程中获取解混矩阵w"""
        self.w_hook_func = w_hook_func

    def assess(self, *args, **kwargs):
        result = super().assess(*args, **kwargs)
        result["epochs"] = self.epochs
        return result
