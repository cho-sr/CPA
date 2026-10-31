# --------------------------------------------------
# FGLA - FSG (Feature Separation from Gradients)
# https://github.com/pigeon-dove/FGLA/tree/master/gen_attack/algorithm.py
# --------------------------------------------------


import torch

from .base_ica import BaseSeparator


def fgla_feature_separation(g_w, g_b, y, offset):
    bz = len(y)
    if offset:
        offset_w = (
            torch.stack([g for idx, g in enumerate(g_w) if idx not in y], dim=0).mean(
                dim=0
            )
            * (bz - 1)
            / bz
        )
        offset_b = (
            torch.stack([g for idx, g in enumerate(g_b) if idx not in y], dim=0).mean()
            * (bz - 1)
            / bz
        )
        conv_out = (g_w[y] - offset_w) / (g_b[y] - offset_b).unsqueeze(1)
    else:
        conv_out = g_w[y] / g_b[y]
    conv_out[torch.isnan(conv_out)] = 0.0
    conv_out[torch.isinf(conv_out)] = 0.0
    return conv_out


class FSG(BaseSeparator):
    def __init__(self, offset: bool = True):
        super().__init__()
        self.offset = offset

    def run(
        self,
        grad_w: torch.Tensor,
        grad_b: torch.Tensor,
        y: torch.Tensor,
        real_feature: torch.Tensor,
    ) -> torch.Tensor:
        return fgla_feature_separation(grad_w, grad_b, y, self.offset)
