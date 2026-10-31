# --------------------------------------------------
# 特征分离器基类
# --------------------------------------------------

import time
import torch
from abc import abstractmethod

from utils.evaluation import similarity_reorder


class BaseSeparator:
    @abstractmethod
    def run(
        self,
        grad_w: torch.Tensor,
        grad_b: torch.Tensor,
        y: torch.Tensor,
        real_feature: torch.Tensor,  # 用于评估
    ) -> torch.Tensor:
        # recover_feature = None
        # return recover_feature
        pass

    def assess(
        self,
        grad_w: torch.Tensor,  # 特征向量对应的权重梯度
        grad_b: torch.Tensor,  # 特征向量对应的偏置梯度
        y: torch.Tensor,  # 特征向量对应的标签
        real_feature: torch.Tensor,  # 特征向量对应的标签
        record_recover=False,  # 是否将重建向量保存到 self.recover_feature
    ):
        """调用特征分离并评估效果
        Args:
            grad_w (torch.Tensor): 特征向量对应的权重梯度
            grad_b (torch.Tensor): 特征向量对应的偏置梯度
            y (torch.Tensor): 特征向量对应的标签
            real_feature (torch.Tensor): 特征向量对应的标签
        """
        t0 = time.time()
        recover_feature = self.run(grad_w, grad_b, y, real_feature)
        t1 = time.time()
        if recover_feature is None:  # 求解失败
            self.recover_feature = None
            return {
                "feature separation": self.__class__.__name__,
                "avg_cos": "error",
                "time": t1 - t0,
            }
        recover_feature = recover_feature.cpu()
        real_feature = real_feature.cpu()

        # 重新排序、评估指标
        recover_feature, real_feature, avg_cos = similarity_reorder(
            recover_feature,
            real_feature,
        )
        if record_recover:
            self.recover_feature = recover_feature
        return {
            "feature separation": self.__class__.__name__,
            "avg_cos": avg_cos,
            "time": t1 - t0,
        }
