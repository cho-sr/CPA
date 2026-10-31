# --------------------------------------------------
# Cocktail Party Attack: Breaking Aggregation-Based Privacy in Federated Learning using Independent Component Analysis
# https://arxiv.org/abs/2209.05578
# https://github.com/facebookresearch/cocktail_party_attack
# --------------------------------------------------

import time
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from .base_attack import BaseAttack
from utils import logger
from .opt_common.grad_loss import cosine_distance
from .opt_common.regularization import total_variation
from .feature_separation.gradient_descent_ica import GradientDescentICA
from utils.evaluation import cosine_similarity, similarity_reorder
from utils import logger, LoopLog
from models.fl import BaseCnnModel


class CPA(BaseAttack):
    def __init__(
        self,
        num_epochs=24000,
        image_size=224,
        lr=0.1,
        fi=2,  # 特征反演权重
        tv=0.5,  # TV正则权重
        input_boxed: bool = True,  # True: 每次输入更新都限制到合理取值范围0~1
        # ICA
        use_true_feature=False,  # 跳过ICA，使用真实特征
        ica_num_epochs=4000,
        device="cpu",
    ):
        super().__init__(device)
        self.num_epochs = num_epochs
        self.image_size = image_size
        self.lr = lr
        self.fi = fi
        self.tv = tv
        self.use_true_feature = use_true_feature
        self.input_boxed = input_boxed
        self.ica = GradientDescentICA(num_epochs=ica_num_epochs, device=device)

    def run(
        self,
        fl_model: BaseCnnModel,
        batch_size: int,
        image_size: int,
        data: dict,
    ):
        self.other_infos = {}
        real_y = data["y"]
        gradient = data["gradient"].to(self.device)
        real_feature = data["feature_vector"].to(self.device)
        S_size = data.get("input_x_num", real_y.shape[0])
        dummy_y = real_y.to(self.device)
        fl_model = fl_model.to(self.device)

        # 特征分离
        if self.use_true_feature:
            recover_feature = real_feature
        else:
            grad_fc_0_w = gradient["fully_connected.fc_0.weight"]
            recover_feature = self.ica.run(
                grad_fc_0_w, None, real_y, real_feature, S_size
            )
            recover_feature = recover_feature.to(self.device)
        # FedAVG: 裁切特征向量组，保留批大小个特征
        if recover_feature.size(0) > batch_size:
            recover_feature = recover_feature[:batch_size]
        self.recover_feature = recover_feature.cpu()
        # 伪样本
        dummy_x = torch.rand(
            size=[batch_size, 3, image_size, image_size],
            requires_grad=True,
            device=self.device,
        )
        opt = Adam([dummy_x], lr=self.lr)
        sch = torch.optim.lr_scheduler.MultiStepLR(
            opt,
            milestones=[
                self.num_epochs // 2.667,
                self.num_epochs // 1.6,
                self.num_epochs // 1.142,
            ],
            gamma=0.1,
        )
        criterion = torch.nn.CrossEntropyLoss()
        cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-10)

        t0 = time.time()
        loop_log = LoopLog("CPA", self.num_epochs)
        try:
            for i in range(self.num_epochs):
                loss_cd = loss_fi = loss_tv = torch.tensor(0.0, device=self.device)

                opt.zero_grad()
                pred = fl_model(dummy_x)
                dummy_loss = criterion(pred, dummy_y)
                dummy_grad = torch.autograd.grad(
                    dummy_loss, fl_model.parameters(), create_graph=True
                )
                loss_cd = cosine_distance(dummy_grad, gradient)
                if self.fi > 0:
                    dummy_feature = fl_model.get_feature_vector()
                    loss_fi = (
                        1 - cosine_similarity(recover_feature, dummy_feature)
                    ).mean()
                if self.tv > 0:
                    loss_tv = total_variation(dummy_x)
                loss = loss_cd + (self.fi * loss_fi) + (self.tv * loss_tv)
                # 记录信息
                self.other_infos = {
                    "epochs": i + 1,
                    "loss": loss.item(),
                    "loss_cd": loss_cd.item(),
                    "loss_tv": loss_tv.item(),
                    "loss_fi": loss_fi.item(),
                    "time": time.time() - t0,
                }
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"CPA ep {i} : 损失计算异常！")
                    break
                loss.backward()
                opt.step()
                sch.step()
                # 输入更新限制到合理取值范围0~1
                if self.input_boxed:
                    dummy_x.data = dummy_x.data.detach().clamp(0, 1)

                loop_log.log(info_func=lambda: self.other_infos)
        except KeyboardInterrupt:
            logger.info("提前终止CPA迭代")
        self.other_infos.update(
            {
                "num_epochs": self.num_epochs,
                "lr": self.lr,
                "fi": self.fi,
                "tv": self.tv,
                "use_true_feature": self.use_true_feature,
                "input_boxed": self.input_boxed,
            }
        )
        with torch.no_grad():
            dummy_x = dummy_x.data.detach().clamp(0, 1)
            return dummy_x

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
        return res
