# --------------------------------------------------
# GLA攻击基类
# --------------------------------------------------

import time
import torch
from abc import abstractmethod

from utils.evaluation import reorder_mse_psnr_ssim_lpips
from utils import ParameterListDict
from models.fl import BaseCnnModel


# 攻击基类
class BaseAttack:
    def __init__(
        self,
        device="cpu",
    ):
        self.other_infos = {}  # 额外描述信息
        self.device = device
        # 缓存最近一次的重建样本与真样本
        self.recover_x = None
        self.real_x = None

    @abstractmethod
    def run(
        self,
        fl_model: BaseCnnModel,
        batch_size: int,
        image_size: int,
        data: dict,
    ):
        """GLA攻击入口
        Args:
            batch_size (int): 批大小
            image_size (int): 样本图像尺寸
            fl_model (BaseCnnModel): FL全局模型
            data (Dict): 信息字典，键包括：
                - gradient:  聚合梯度
                - x:  真实样本
                - y:  真实标签
                - feature_vector:  真实特征向量
                - x_list:  真实样本列表
                - y_list:  真实标签列表
                - feature_vector_list:  真实特征向量列表

        Returns:
            recover_x (torch.Tensor): 重建样本
        """
        recover_x = None  # 返回重构的样本
        return recover_x

    def assess(
        self,
        fl_model: BaseCnnModel,
        batch_size: int,
        image_size: int,
        data: dict,
    ) -> dict:
        """调用攻击，并评估 GLA 重建效果。

        Args:
            batch_size (int): 批大小
            image_size (int): 样本图像尺寸
            fl_model (BaseCnnModel): FL全局模型
            data (Dict): 信息字典，键包括：
                - gradient:  聚合梯度
                - x:  真实样本
                - y:  真实标签
                - feature_vector:  真实特征向量
                - x_list:  真实样本列表
                - y_list:  真实标签列表
                - feature_vector_list:  真实特征向量列表

        Returns:
            dict: 信息字典，如：
                return {
                    "mse": mse,
                    "psnr": psnr,
                    "ssim": ssim,
                    "avg_cos": avg_cos,  # 重建样本的平均余弦相似度
                    "time": t,  # 重建耗时
                }

        执行后可用的属性：
            self.recover_x  # 重建样本
            self.real_x  # 真样本（排序后）
        """
        self.other_infos = {}
        t0 = time.time()
        real_x = data["x"]
        recover_x = self.run(fl_model, batch_size, image_size, data)
        t1 = time.time()
        recover_x = recover_x.cpu()
        real_x = real_x.cpu()

        # 兼容性检查：去除nan，限定范围0~1
        recover_x = torch.nan_to_num(recover_x, nan=0.0)
        recover_x = recover_x.clamp(0, 1)
        self.recover_x_original = recover_x

        # 重新排序、评估指标
        mse, psnr, ssim, lpips, avg_cos, recover_x, real_x = (
            reorder_mse_psnr_ssim_lpips(recover_x, real_x, allow_repeat=False)
        )
        self.recover_x = recover_x
        self.real_x = real_x
        info = {
            "attack": self.__class__.__name__,
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "avg_cos": avg_cos,
        }
        if self.other_infos:
            info.update(self.other_infos)
        info["time"] = t1 - t0
        return info
