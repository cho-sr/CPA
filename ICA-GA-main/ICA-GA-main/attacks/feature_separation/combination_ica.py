# --------------------------------------------------
# 混合 Fast-ICA + GD-ICA
# --------------------------------------------------

from torch import Tensor

from .base_ica import BaseICA
from .fast_ica import FastICA
from .gradient_descent_ica import GradientDescentICA


class CombinationICA(BaseICA):
    def __init__(
        self,
        fast_ica_kwargs={},
        gd_ica_kwargs={},
    ):
        super().__init__()
        self.fast_ica = FastICA(**fast_ica_kwargs)
        self.gd_ica = GradientDescentICA(**gd_ica_kwargs)

    def ica(
        self,
        X: Tensor,  # 混合信号矩阵
        n_components: int,  # 源信号个数
        S=None,  # 源信号（测试用）
        w_init=None,  # 初始解混矩阵
    ):
        self.fast_ica.ica(X, n_components, S, w_init)
        W = self.fast_ica.W
        re_S = self.gd_ica.ica(X, n_components, S, W)
        self.W = self.gd_ica.W
        self.epochs = self.fast_ica.epochs + self.gd_ica.epochs
        return re_S
