import torch
from sklearn.decomposition import FastICA

from .base_ica import BaseICA


class SklearnFastICA(BaseICA):

    def ica(
        self,
        X: torch.Tensor,  # 混合信号矩阵
        S_size: int,  # 源信号个数
        S,
    ):
        super().__init__()
        X_array = X.detach().numpy().T
        ica = FastICA(
            n_components=S_size,
            random_state=0,
            whiten="arbitrary-variance",
            fun="cube",
        )
        S_numpy = ica.fit_transform(X_array)
        recover_S = torch.from_numpy(S_numpy).float().T
        return recover_S.abs()
