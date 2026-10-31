# --------------------------------------------------
# 用于训练生成器的损失项
# --------------------------------------------------

from .base_loss import BaseLoss, LossInputDict
from .simple_losses import MseLoss, FeatLoss, TvLoss, SsimLoss
from .lpips_loss import LpipsLoss

loss_dict = {
    "mse": MseLoss,
    "feat": FeatLoss,
    "tv": TvLoss,
    "ssim": SsimLoss,
    "lpips": LpipsLoss,
}
