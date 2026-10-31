# --------------------------------------------------
# Fast Generation-Based Gradient Leakage Attacks: An Approach to Generate Training Data Directly From the Gradient
# https://ieeexplore.ieee.org/abstract/document/10505158
# https://github.com/pigeon-dove/FGLA
# --------------------------------------------------

import torch

from .base_attack import BaseAttack
from models.base_model import BaseModel
from models.generator import generator_models
from utils.evaluation import similarity_reorder, mse

from utils import ParameterListDict
from .feature_separation.fgla_fsg import fgla_feature_separation
from .gen_common.pretreatment import standardization


class FGLA(BaseAttack):
    def __init__(
        self,
        generator_model: str = "FglaGenerator50",  # 生成器类名
        model_save_dir: str = "./data/models",  # 解码器目录
        model_save_name: str = "",  # 解码器参数名称
        model_kwargs: dict = {},  # 模型参数
        offset=True,  # 抵消量
        standardization=False,  # 标准化，以适配我们的模型
        device="cpu",
    ):
        super().__init__(device)
        if not model_save_name:
            model_save_name = generator_model
        self.generator: BaseModel = generator_models[generator_model](
            save_dir=model_save_dir, save_name=model_save_name, **model_kwargs
        )
        self.generator.load_model()
        self.generator.eval()
        self.generator.set_requires_grad(False)
        self.offset = offset
        self.standardization = standardization
        # 重建特征向量
        self.recover_feature = None
        self.other_infos = {
            "generator_model": generator_model,
            "model_save_dir": model_save_dir,
            "model_save_name": model_save_name,
            "offset": offset,
            "standardization": standardization,
        }

    def run(
        self,
        fl_model,
        batch_size: int,
        image_size: int,
        data: dict,
    ):
        gradient = data["gradient"]
        real_y = data["y"]
        grad_fc_0_w = gradient["fully_connected.fc_0.weight"]
        grad_fc_0_b = gradient["fully_connected.fc_0.bias"]
        # 特征分离
        recover_feature = fgla_feature_separation(
            grad_fc_0_w, grad_fc_0_b, real_y, self.offset
        )
        self.recover_feature = recover_feature
        # 标准化
        if self.standardization:
            recover_feature = standardization(recover_feature)
        # 图像生成
        generator = self.generator.to(self.device)
        recover_feature = recover_feature.to(self.device)
        recover_x = generator(recover_feature)
        return recover_x

    def assess(
        self,
        fl_model,
        batch_size: int,
        image_size: int,
        data: dict,
    ):
        res = super().assess(fl_model, batch_size, image_size, data)
        # 特征向量重新排序
        real_feature = data["feature_vector"].cpu()
        self.recover_feature, _, avg_cos = similarity_reorder(
            self.recover_feature, real_feature
        )
        # 增加评估特征向量指标
        res.update(
            {
                "feature_cos_sim": avg_cos,
            }
        )
        return res
