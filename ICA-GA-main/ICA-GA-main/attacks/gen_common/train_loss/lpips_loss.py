import torch

from .base_loss import BaseLoss, LossInputDict
from utils.pytorch_lpips import Lpips


class LpipsLoss(BaseLoss):
    """感知相似度损失 (LPIPS Loss)
    Learned Perceptual Image Patch Similarity
    """

    def __init__(self, net: str = "vgg", device="cpu"):
        """
        初始化 LPIPS 损失函数。

        Args:
            net (str): 使用的网络结构。默认为 'vgg'。
                       可选值为 'alex' (AlexNet) 或 'squeeze' (SqueezeNet)。
        """
        super(LpipsLoss, self).__init__()
        self._lpips = Lpips(net, device)

    def __call__(self, loss_input_dict: LossInputDict) -> torch.Tensor:
        """
        调用损失函数
        """
        re_x = loss_input_dict["re_x"]
        real_x = loss_input_dict["real_x"]
        return self._lpips(re_x, real_x)
