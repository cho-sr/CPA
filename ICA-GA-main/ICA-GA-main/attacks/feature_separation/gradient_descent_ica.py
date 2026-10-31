"""
基于梯度下降的ICA
"""

import time
import warnings
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD

from utils.log import logger
from utils.evaluation import (
    cosine_similarity,
    reorder_tensor,
    cosine_similarity_allow_repeat,
)
from .base_ica import BaseICA

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Casting complex values to real discards the imaginary part",
)


class GradientDescentICA(BaseICA):
    def __init__(
        self,
        lr=0.001,  # 学习率
        num_epochs=4000,  # 优化次数
        g_func="logcosh",
        # 最优超参
        ne=0.0935827857091549,  # 负熵权重
        decor=0.2513674104161625,  # 去相关权重
        nv=0.05811961446881231,  # 负值惩罚权重
        l1=17.94138146247377,  # L1正则权重
        is_log=True,  # 是否输出日志
        loss_inf_break=True,
        log_num=10,  # 日志记录次数
        *args,
        **kwargs,
    ):
        """
        基于梯度下降的ICA算法。
        用于从混合信号中分离出独立的源信号。
        先验：已知源信号是非负、稀疏的。

        Args:
            lr (float): 学习率
            num_epochs (float): 优化次数
            g_func (str): 非高斯性估计类型，可选：
                - "logcosh"
                - "exp"
                - "quartic"
            ne (float): 负熵权重
            decor (float): 去相关权重
            nv (float): 负值惩罚权重
            l1 (float): L1正则权重
            is_log (bool): 是否输出日志
            loss_inf_break (bool): 负熵损失爆炸时，True结束迭代，False屏蔽该损失继续迭代
            log_num (int): 日志记录次数
            w_init_type (str): 默认的初始解混矩阵类型，"eye" 单位矩阵， "random" 随机， "random_q" 随机正交。
            device (torch.device): 计算所使用的设备
        """
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.num_epochs = num_epochs
        self.ne = ne
        self.decor = decor
        self.nv = nv
        self.l1 = l1
        self.psnr = 0.0
        self.avg_similarity = 0.0
        self.is_log = is_log
        self.log_num = log_num
        self.loss_inf = False  # 标记是否损失爆炸
        self.loss_inf_break = loss_inf_break
        if g_func == "logcosh":
            self.g_func = self._g_func_logcosh
        elif g_func == "exp":
            self.g_func = self._g_func_exp
        elif g_func == "quartic":
            self.g_func = self._g_func_quartic
        else:
            raise ValueError(f"Unknown function type: {g_func}")

    def _g_func_logcosh(self, estimated_components: torch.Tensor):
        """
        双曲余弦对数函数 G(x) = log(cosh(x))
        适用于亚高斯或接近高斯分布的源信号。
        """
        return torch.log(torch.cosh(estimated_components) + self.eps)

    def _g_func_exp(self, estimated_components: torch.Tensor):
        """
        指数函数 G(x) = -exp(-x^2/2)
        对异常值敏感，适合处理超高斯分布（尖峰厚尾）的源信号。
        """
        return -torch.exp(-(estimated_components**2) / 2.0)

    def _g_func_quartic(self, estimated_components: torch.Tensor):
        """
        四次函数 G(x) = (1/4)x^4
        适用于明显超高斯信号分布（高峭度）的源信号。
        """
        return (estimated_components**4) / 4.0

    # 负熵损失
    def get_loss_ne(self, estimated_components: torch.Tensor):
        g_values = self.g_func(estimated_components)

        negentropy = g_values.mean(dim=-1) ** 2
        loss_ne = -negentropy.mean()

        return loss_ne

    # 去相关损失
    def get_loss_decor(self, unmixing_mat_norm: torch.Tensor):
        # 得到一个对称矩阵，其中每个元素表示解混矩阵的两行之间的余弦相似度。
        cos_matrix = torch.matmul(unmixing_mat_norm, unmixing_mat_norm.T).abs()

        # 计算去相关损失（旧，废弃）
        # loss_decor = (torch.exp(cos_matrix * T) - 1).mean()
        # 去相关损失， Frobenius 范数的平方
        off_diagonal_matrix = cos_matrix - torch.eye(
            self.n_components, device=self.device
        )
        loss_decor = (off_diagonal_matrix**2).sum()
        return loss_decor

    # 负值惩罚：解混矩阵和混合信号（梯度）中都含负值，互相抵消，促进源信号中不含负值。
    def get_loss_nv(self, independent_components: torch.Tensor):
        loss_nv = torch.minimum(  # 每个信号中，取正值或负值中较小一方的均值
            F.relu(-independent_components).norm(dim=-1),
            F.relu(independent_components).norm(dim=-1),
        ).mean()
        return loss_nv

    # L1正则：鼓励稀疏性
    def get_loss_l1(self, independent_components: torch.Tensor):
        loss_l1 = torch.abs(independent_components).mean()
        return loss_l1

    def ica(
        self,
        X: torch.Tensor,  # 混合信号矩阵，形状为 (n_features, n_samples)，即 (混合信号数量, 单个信号长度)
        n_components: int,  # 源信号个数
        S=None,  # 源信号（测试用）。在当前场景中，我们已知它是【非负的】。
        w_init=None,  # 初始解混矩阵
    ):
        self.loss_inf = False  # 标记是否损失爆炸
        if S is not None:
            S = S.cpu()
        X_centered, K, X_w, X_mean, n_components = self._preprocessing_cpu(
            X, n_components
        )
        self.n_components = n_components

        # 解混矩阵，待优化。
        W = self._get_w_init(w_init, n_components)
        W.requires_grad = True

        param_list = [W]
        opt = SGD(param_list, lr=self.lr, weight_decay=0, momentum=0)
        t0 = time.time()
        log_eps = self.num_epochs // self.log_num
        if self.is_log:
            logger.info(
                f"Start GD-ICA. n_components: {n_components}, ne: {self.ne:<4}, decor: {self.decor:<4}, \
nv: {self.nv:<4}, l1: {self.l1:<4}"
            )
        self.epochs = 0
        for i in range(self.num_epochs):
            self.epochs += 1
            loss_ne = loss_decor = loss_nv = loss_l1 = torch.tensor(
                0.0, device=self.device
            )
            # 解混矩阵归一化
            unmixing_mat_norm = W / (W.norm(dim=-1, keepdim=True) + self.eps)
            # 估计的独立成分（解混信号）
            estimated_components = torch.matmul(unmixing_mat_norm, X_w)
            # 还原的独立成分：经过逆中心化处理，使其与源信号的均值一致
            independent_components = estimated_components + torch.matmul(
                torch.matmul(unmixing_mat_norm, K), X_mean
            )
            # =============================
            # 负熵损失
            if self.ne > 0:
                loss_ne = self.get_loss_ne(estimated_components)
                if torch.isinf(loss_ne) or torch.isnan(loss_ne):  # 损失爆炸
                    self.loss_inf = True
                    if self.loss_inf_break:  # 继续迭代
                        logger.debug(f"GD-ICA:{i+1:>5}, loss_ne={loss_ne}, break")
                        break
                    else:  # 屏蔽 loss_ne 继续迭代
                        loss_ne = torch.tensor(0.0, device=self.device)
            # 去相关损失
            if self.decor > 0:
                loss_decor = self.get_loss_decor(unmixing_mat_norm)
            # 负值惩罚
            if self.nv > 0:
                loss_nv = self.get_loss_nv(independent_components)
            # L1正则
            if self.l1 > 0:
                loss_l1 = self.get_loss_l1(independent_components)
            # 总损失
            loss = (
                loss_ne * self.ne
                + loss_decor * self.decor
                + loss_nv * self.nv
                + loss_l1 * self.l1
            )
            loss.backward()
            opt.step()
            recover_S = None
            if i == 0 or i % log_eps == log_eps - 1 or i == self.num_epochs - 1:
                if self.w_hook_func is not None:
                    self.w_hook_func(W.detach().cpu())
                if self.is_log:
                    t1 = time.time()
                    if S is None:
                        logger.info(f"{i:<4}, loss: {loss:.4f}, time: {t1-t0:.4f}s")
                    else:
                        # 恢复源信号
                        recover_S = self._get_recover_S(W, n_components, X_w, X_mean, K)
                        recover_S = recover_S.cpu()
                        self.avg_similarity, matches = cosine_similarity(recover_S, S)
                        # self.avg_similarity, matches = cosine_similarity_allow_repeat(
                        #     recover_S, S
                        # )
                        logger.info(
                            f"GD-ICA:{i+1:>5}, loss: {loss:.4f}, cos_similarity: {self.avg_similarity:.4f}, time: {t1-t0:.4f}s"
                            + f", loss_ne: {loss_ne.item():.4f}, loss_decor: {loss_decor.item():.4f}, loss_nv: {loss_nv.item():.4f}, loss_l1: {loss_l1.item():.4f}"
                        )
        # 返回恢复的源信号
        recover_S = self._get_recover_S(W, n_components, X_w, X_mean, K)
        recover_S = recover_S.cpu()
        if S is not None:  # 调整顺序
            self.avg_similarity, matches = cosine_similarity(recover_S, S)
            # self.avg_similarity, matches = cosine_similarity_allow_repeat(recover_S, S)
            recover_S, _ = reorder_tensor(recover_S, matches, S)  # 重新排列顺序

        self.W = W
        return recover_S
