import torch
import lpips


class Lpips:
    """感知相似度 LPIPS
    Learned Perceptual Image Patch Similarity
    """

    def __init__(self, net: str = "vgg", device="cpu"):
        """
        初始化 LPIPS

        Args:
            net (str): 使用的网络结构。默认为 'vgg'。
                       可选值为 'alex' (AlexNet) 或 'squeeze' (SqueezeNet)。
        """
        self._fn = lpips.LPIPS(net=net, lpips=True, pretrained=True)
        self._fn.eval()
        self.set_device(device)

    def set_device(self, device):
        self._fn = self._fn.to(device)
        self.device = device

    def __call__(self, re_x: torch.Tensor, real_x: torch.Tensor) -> torch.Tensor:
        """
        计算LPIPS

        Args:
            re_x (Tensor): 一组重建图像的张量。形状通常是 (n, 3, H, W) ，像素值范围 [0, 1]
            real_x (Tensor): 对应的原始图像张量。形状通常是 (n, 3, H, W)
        """
        # 像素值范围 [0, 1] 转为 [-1, 1]
        re_x_norm = re_x * 2.0 - 1.0
        real_x_norm = real_x * 2.0 - 1.0

        re_x_norm = re_x_norm.to(self.device)
        real_x_norm = real_x_norm.to(self.device)

        loss = self._fn(re_x_norm, real_x_norm)
        return loss.mean()
