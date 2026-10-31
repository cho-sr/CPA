"""
n_samples: 数据点的数量，即单个信号的长度。
n_features: 每个样本的特征数量，即观测到的混合信号（传感器或通道）的数量。
n_components: 估计的独立源信号的数量，即待分离的成分数。
"""

import math
import time
import torch
from torch import Tensor

from utils import logger
from utils.evaluation import cosine_similarity, cosine_similarity_allow_repeat
from .base_ica import BaseICA


class FastICA(BaseICA):

    def __init__(
        self,
        whiten_solver="eigh",
        g_func="logcosh",
        algorithm="parallel",
        max_iter=200,
        is_log=True,
        *args,
        **kwargs,
    ):
        """
        FastICA (Fast Independent Component Analysis) 算法实现。
        用于从混合信号中分离出独立的源信号。
        先验：已知源信号是非负、稀疏的。

        Args:
            whiten_solver (str): 白化求解器类型，可选：
                - "eigh": 通过对协方差矩阵 (X.T @ X) 进行特征值分解来实现。适用于特征数量（n_features）较少的情况。
                - "svd": 通过对中心化后的数据矩阵 X_centered_T 进行奇异值分解来实现。
            g_func (str): 非高斯性估计类型，可选：
                - "logcosh"
                - "exp"
                - "quartic"
            algorithm (str): FastICA使用的算法，可选：
                - "parallel": 并行算法，同时更新所有独立成分的解混向量。
                - "deflation": 缩减型算法，逐个提取独立成分。
            max_iter (int): 迭代次数: FastICA 迭代过程中的最大迭代次数。
            is_log (bool): 是否打印循环日志
            w_init_type (str): 默认的初始解混矩阵类型，"eye" 单位矩阵， "random" 随机， "random_q" 随机正交。
            device (torch.device): 计算所使用的设备
        """
        super().__init__(*args, **kwargs)
        self.whiten_solver = whiten_solver
        self.max_iter = max_iter
        self.is_log = is_log
        if g_func == "logcosh":
            self.g_func = self._g_func_logcosh
        elif g_func == "exp":
            self.g_func = self._g_func_exp
        elif g_func == "quartic":
            self.g_func = self._g_func_quartic
        else:
            raise ValueError(f"Unknown function type: {g_func}")
        if algorithm == "parallel":
            self.algorithm_func = self._ica_par
        elif algorithm == "deflation":
            self.algorithm_func = self._ica_def
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm}")

    def _g_func_logcosh(self, x: Tensor, alpha=1.0):
        """双曲余弦对数函数 G(x) = (1/α) log cosh (αx)
        鲁棒性强，适用于亚高斯或接近高斯分布的混合信号。

        Args:
            x (Tensor): 要评估的信号
            alpha (float): 控制平滑性，值越大对边缘值越敏感。

        Returns:
            gx: 一阶导数
            g_x: 二阶导数的均值
        """
        x_scaled = x * alpha
        gx = torch.tanh(x_scaled)
        g_x = alpha * (1 - gx**2).mean(dim=-1)
        return gx, g_x

    def _g_func_exp(self, x: Tensor):
        """指数型函数 G(x) = -e ^ ((-x^2) / 2)
        对异常值敏感，适合处理超高斯分布（尖峰厚尾）的源信号
        """
        exp_term = torch.exp(-(x**2) / 2)
        gx = x * exp_term
        g_x = (1 - x**2) * exp_term
        return gx, g_x.mean(dim=-1)

    def _g_func_quartic(self, x: Tensor):
        """四次函数 G(x) = (1/4) x^4
        适用于明显超高斯信号，计算高效，对噪声敏感
        """
        gx = x**3
        g_x = 3 * x**2
        return gx, g_x.mean(dim=-1)

    def _ica_par(self, X_w, w_init, K, n_components, X_mean, S=None):
        """并行ICA (Parallel FastICA)
        通过迭代优化解混矩阵 W，同时估计所有独立成分，最大化其非高斯性。
        该算法旨在找到一个解混矩阵 W ，使得 Y = W @ X_w 中的 Y 的行向量（即估计的独立成分）尽可能地非高斯和相互独立。

        Args:
            X_w (Tensor): 白化后的混合信号矩阵，形状为 (n_components, n_samples)。
            w_init (Tensor): 初始的解混矩阵 W ，形状为 (n_components, n_components)。
            K (Tensor): 白化矩阵，形状为 (n_components, n_features)。
            n_components (int): 独立成分个数。
            X_mean (Tensor): 原始混合信号的均值 (n_features, 1) 。
            S (Tensor, optional): 真实的源信号，用于在迭代过程中监控还原效果。

        Returns:
            W (Tensor): 估计得到的解混矩阵，作用于白化数据 X_w 。形状为 (n_components, n_components)。
        """
        if S is not None:
            S = S.cpu()

        def _sym_decorrelation(W):
            """对称去相关。确保解混矩阵 W 的行向量（在白化空间中）是正交的。
            W <- (W * W.T) ^ (-1/2) * W

            Args:
                W (Tensor): 迭代更新后的解混矩阵。

            Returns:
                Tensor: 对称去相关/正交化后的矩阵。
            """
            # 计算 W * W^T 的特征分解
            s, u = torch.linalg.eigh(W @ W.T)  # s 升序排列，u 是列特征向量
            # 对特征值裁剪防止数值不稳定
            s = torch.clamp(s, min=torch.finfo(W.dtype).tiny)
            # 计算 (W * W.T)^(-1/2) = U * s^(-1/2) * U^T
            scale = (1.0 / torch.sqrt(s)).unsqueeze(0)  # 转换为行向量广播
            u_scale = u * scale  # 特征向量按列缩放
            return u_scale @ u.T @ W

        t0 = time.time()
        W = _sym_decorrelation(w_init).to(self.device)
        n_components, n_samples = X_w.shape

        self.epochs = 0
        for ii in range(self.max_iter):
            # 前向传播和非线性变换
            gwtx, g_wtx = self.g_func(W @ X_w)

            # 更新规则
            term1 = (gwtx @ X_w.T) / n_samples  # (n_components, n_components)
            term2 = g_wtx * W
            W1 = _sym_decorrelation(term1 - term2)

            # 收敛性检查
            # 计算两个矩阵 W1 和 W 对应行向量的点积绝对值与 1 的最大绝对误差
            # 用于衡量它们的行向量在某种性质上的接近程度
            # 逐行点积， dot[i] = Σj (W1[i, j] * W[i, j])
            dot = torch.einsum("ij,ij->i", W1, W)
            lim = torch.max(torch.abs(torch.abs(dot) - 1)).item()

            W = W1
            self.epochs += 1
            if self.w_hook_func is not None:
                self.w_hook_func(W.detach().cpu())
            if S is not None and (ii % 20 == 0 or lim < 1e-6) and self.is_log:
                re_S = self._get_recover_S(W, n_components, X_w, X_mean, K).cpu()
                avg_similarity, _ = cosine_similarity(re_S, S)
                # avg_similarity, _ = cosine_similarity_allow_repeat(re_S, S)
                t1 = time.time()
                logger.info(
                    f"FastICA (par):[{ii:>3}], lim: {lim:.6f}, cos_similarity: {avg_similarity:.4f}, time: {t1-t0:.4f}s"
                )
            if math.isinf(lim) or math.isnan(lim):  # 求解失败
                logger.warning(f"FastICA(par):[{ii:>3}], lim: {lim}")
                return None
            elif lim < 1e-6:  # 收敛
                if ii == 0:
                    logger.warning(
                        f"FastICA (par): 收敛性计算异常，可能源信号相关性过大"
                    )
                    return None
                break
        else:
            logger.warning(f"FastICA not converging, lim={lim:.10f}")

        return W

    def _ica_def(self, X_w, w_init, K, n_components, X_mean, S=None):
        """【不常用，部分功能不完善】
        缩减型 FastICA (Deflationary FastICA) 算法。
        该算法逐个估计独立成分。对于每个成分，它优化一个投影向量 w，以最大化 w^T X_w 的非高斯性。
        在找到一个成分后，将其从混合信号中移除（通过投影），然后在剩余信号上估计下一个成分。
        实际上，这里通过 Gram-Schmidt 正交化来确保新找到的 w 与之前找到的 w 正交。

        Args:
            X_w (Tensor): 白化后的混合信号矩阵，形状为 (n_components, n_samples)。
            w_init (Tensor): 初始的解混矩阵 W ，形状为 (n_components, n_components)。
            K (Tensor): 白化矩阵，形状为 (n_components, n_features)。
            n_components (int): 独立成分个数。
            X_mean (Tensor): 原始混合信号的均值 (1, n_features) 。
            S (Tensor, optional): 真实的源信号，用于在迭代过程中监控还原效果。

        Returns:
            W (Tensor): 估计得到的解混矩阵，作用于白化数据 X_w 。形状为 (n_components, n_components)。
        """

        def _gs_decorrelation_torch(
            w: torch.Tensor, W: torch.Tensor, j: int
        ) -> torch.Tensor:
            """
            对向量 w 关于矩阵 W 的前 j 行进行 Gram-Schmidt 正交化。

            Args:
                w (Tensor): 需要被正交化的向量。(n_features,)
                W (Tensor): 定义了子空间的矩阵，w 将相对于此矩阵的前 j 行进行正交化。(n_components, n_features)
                    假设 W 的行是正交的（或者至少 W[:j] 的行是相互正交的）。
                j (int): 指定 W 矩阵中用于正交化的行数（从第0行开始，到第 j-1 行）。

            Returns:
                torch.Tensor: 正交化后的向量 w。
            """
            if j > 0:
                # W_sub 是 W 矩阵的前 j 行，代表已经找到的独立成分
                # W_sub 的形状是 (j, n_features)
                W_sub = W[:j, :]
                # 计算 w 在 W_sub 的行向量上的投影： P = w @ W_sub.T @ W_sub
                # 1. w @ W_sub.T 计算 w 与 W_sub 中每个行向量的内积，结果形状为 (j,)
                #    这代表 w 在每个 W_sub 的行向量上的投影系数
                # 2. (w @ W_sub.T) @ W_sub 将这些投影系数与对应的 W_sub 行向量相乘再相加
                #    结果形状为 (n_features,)，是 w 在 W_sub 张成的子空间上的总投影
                projection = w @ W_sub.T @ W_sub
                w -= projection
            return w

        t0 = time.time()
        n_components, n_samples = X_w.shape
        # 初始化解混合矩阵 W，用于存储找到的独立成分的投影向量
        # W 的每一行将是一个独立成分的投影向量 w
        W = torch.zeros(
            (n_components, n_components), dtype=X_w.dtype, device=X_w.device
        )

        # j 是当前正在提取的第 j 个独立成分的索引
        for j in range(n_components):
            # 从 w_init 中获取第 j 个成分的初始权重向量 w
            w = w_init[j, :].clone()

            # 对 w 进行归一化
            w /= torch.sqrt(torch.sum(w**2))

            # 单个成分的迭代提取过程
            for i in range(self.max_iter):
                # 1. 计算 w^T X
                # w 的形状是 (n_components)，X1 的形状是 (n_components, n_samples)
                # w_T_X 的形状将是 (n_samples)
                w_T_X = w @ X_w

                # 2. 应用非线性函数 g 及其导数 g'
                # gwtx 对应 g(w^T X)，g_wtx 对应 g'(w^T X)
                # 两者形状都应为 (n_samples,)
                gwtx, g_wtx = self.g_func(w_T_X)

                # 3. 更新 w： w_new = E[X * g(w^T X)] - E[g'(w^T X)] * w
                term1 = torch.mean(X_w * gwtx, dim=1)
                # term2 = torch.mean(g_wtx) * w
                term2 = g_wtx * w
                # 更新 w 的方向
                w_new = term1 - term2

                # 4. 对 w_new 进行正交化处理 (Gram-Schmidt decorrelation)
                w_new = _gs_decorrelation_torch(w_new, W, j)

                # 5. 对 w_new 进行归一化
                w_new /= torch.sqrt(torch.sum(w_new**2))
                # 更新 w
                w = w_new

                # 6. 检查收敛性
                # 计算 w_new 和旧的 w 之间的点积的绝对值
                # 如果它们方向相同且模为1，则点积为1。如果方向相反，点积为-1。
                # abs_dot_product 越接近1，说明 w_new 与 w 的方向越一致（或相反）。
                # 我们关心的是方向是否稳定，所以取绝对值。
                # lim 表示 | |w_new^T w_old| - 1 |，如果接近0，说明收敛。
                # (w_new * w).sum() 是点积
                lim = torch.abs(torch.abs(torch.sum(w_new * w)) - 1.0)

                if S is not None and i % 20 == 0:
                    re_S = self._get_recover_S(W, n_components, X_w, X_mean, K).cpu()
                    avg_similarity, _ = cosine_similarity(re_S, S)
                    logger.info(
                        f"{j}, {i} 还原相似度：{avg_similarity:.5f}, lim: {lim:.10f}"
                    )
                    t1 = time.time()
                    logger.info(
                        f"FastICA(def):[{j},{i:>3}], lim: {lim:.6f}, cos_similarity: {avg_similarity:.4f}, time: {t1-t0:.4f}s"
                    )
                if math.isinf(lim) or math.isnan(lim):  # 求解失败
                    logger.warning(f"FastICA(par):[{i:>3}], lim: {lim}")
                    return None
                elif lim < 1e-5:  # 收敛
                    break
            # 将收敛后的权重向量 w 存储到 W 矩阵的第 j 行
            W[j, :] = w
        # 返回估计得到的解混合矩阵 W
        return W

    def ica(
        self,
        X: Tensor,  # 混合信号矩阵
        n_components: int,  # 源信号个数
        S=None,  # 源信号（测试用）
        w_init=None,  # 初始解混矩阵
    ):
        """
        执行FastICA算法来分离独立成分。

        Args:
            X (Tensor): 混合信号矩阵，形状为 (n_features, n_samples)，即 (混合信号数量, 单个信号长度)
            n_components (int): 要估计的独立源信号的数量。这个值不能超过 n_features 和 n_samples。
            S (Tensor, optional): 真实的源信号矩阵，形状为 (n_components, n_samples)，即 (源信号数量, 单个信号长度)
                                  在当前场景中，我们已知它是【非负的】。
                                  用于在 FastICA 迭代过程中进行性能评估。
                                  默认为 None，表示不进行此评估。

        Returns:
            components (Tensor): 估计得到的还原信号，形状为 (n_components, n_samples)，即 (源信号数量, 单个信号长度)
        """
        if S is not None and S.device != self.device:
            S = S.to(self.device)
        X_centered, K, X_w, X_mean, n_components = self._preprocessing(X, n_components)
        # 解混矩阵，待优化。
        W = self._get_w_init(w_init, n_components)
        # 求解解混矩阵
        W = self.algorithm_func(X_w, W, K, X_centered, X_mean, S)
        if W is None:
            self.W = None
            return None
        # 获取还原信号
        re_S = self._get_recover_S(W, n_components, X_w, X_mean, K).cpu()
        self.W = W
        return re_S


"""
@article{hyvarinen2000independent,
  title={Independent component analysis: algorithms and applications},
  author={Hyv{\"a}rinen, Aapo and Oja, Erkki},
  journal={Neural networks},
  volume={13},
  number={4-5},
  pages={411--430},
  year={2000},
  publisher={Elsevier}
}
"""
