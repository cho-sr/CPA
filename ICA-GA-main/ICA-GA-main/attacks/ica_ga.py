# ICA-driven Generative Attacks (ICA-GA)

import time
import copy
import torch

from .base_attack import BaseAttack
from .base_attack import BaseAttack
from models.base_model import BaseModel
from models.generator import generator_models
from utils.evaluation import similarity_reorder, mse
from utils import logger, ParameterListDict
from utils.evaluation import *
from .gen_common.pretreatment import standardization
from .feature_separation.combination_ica import CombinationICA


class ICAGA(BaseAttack):
    def __init__(
        self,
        generator_model: str = "FglaGenerator50",  # 生成器类名
        model_save_dir: str = "./data/models",  # 解码器目录
        model_save_name: str = "",  # 解码器参数名称
        model_kwargs: dict = {},  # 模型参数
        gd_ica_num_epochs: int = 1000,  # 梯度下降ICA迭代次数
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
        self.ica = CombinationICA(
            gd_ica_kwargs={"num_epochs": gd_ica_num_epochs, "device": device}
        )
        # 重建特征向量
        self.recover_feature = None
        self.other_infos = {
            "generator_model": generator_model,
            "model_save_dir": model_save_dir,
            "model_save_name": model_save_name,
            "gd_ica_num_epochs": gd_ica_num_epochs,
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
        real_feature = data["feature_vector"]
        S_size = data.get("input_x_num", real_y.shape[0])
        # 特征分离
        grad_fc_0_w = gradient["fully_connected.fc_0.weight"]
        recover_feature = self.ica.run(grad_fc_0_w, None, real_y, real_feature, S_size)
        self.recover_feature = recover_feature.cpu()
        # 标准化
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
            self.recover_feature, real_feature, allow_repeat=False
        )
        # 增加评估特征向量指标
        res["feature_cos_sim"] = avg_cos  # 总相似度
        _, _, avg_cos = similarity_reorder(
            self.recover_feature, real_feature, allow_repeat=True
        )
        res["feature_cos_sim (repeat)"] = avg_cos  # 允许重复匹配的相似度

        # FedAVG 中，评估每一批次的特征和样本
        if "feature_vector_list" in data:
            for i, tf in enumerate(data["feature_vector_list"]):
                tf = tf.cpu()
                s, _ = cosine_similarity_allow_repeat(self.recover_feature, tf)
                res[f"{i}_feature_cos_sim"] = s
        if "x_list" in data:
            for i, tx in enumerate(data["x_list"]):
                tx = tx.cpu()
                mse, psnr, ssim, lpips, _, _, _ = reorder_mse_psnr_ssim_lpips(
                    self.recover_x_original, tx
                )
                res[f"{i}_mse"] = mse
                res[f"{i}_psnr"] = psnr
                res[f"{i}_ssim"] = ssim
                res[f"{i}_lpips"] = lpips
        return res
