# --------------------------------------------------
# Inverting Gradients -- How easy is it to break privacy in federated learning?
# https://arxiv.org/abs/2003.14053
# https://github.com/JonasGeiping/invertinggradients
# --------------------------------------------------

import time
import torch
from torch.optim import Adam

from .base_attack import BaseAttack
from utils import logger, LoopLog
from .opt_common.grad_loss import cosine_distance
from .opt_common.regularization import total_variation
from utils import ParameterListDict
from models.fl import BaseCnnModel


class IG(BaseAttack):
    def __init__(
        self,
        num_epochs=24000,  # 搜索迭代次数
        lr=0.1,  # 优化学习率
        tv=0.1,  # TV正则化系数
        input_boxed: bool = True,  # True: 每次输入更新都限制到合理取值范围0~1
        device="cpu",
    ):
        super().__init__(device)
        self.num_epochs = num_epochs
        self.lr = lr
        self.tv = tv
        self.input_boxed = input_boxed

    def run(
        self,
        fl_model,
        batch_size: int,
        image_size: int,
        data: dict,
    ):
        fl_model = fl_model.to(self.device)
        self.other_infos = {}  # 额外描述信息
        real_y = data["y"]
        gradient = data["gradient"].to(self.device)
        logger.info(f"IG:  lr={self.lr}, tv={self.tv}")
        loop_log = LoopLog("IG", self.num_epochs)
        # 伪样本
        dummy_x = torch.rand(
            size=[batch_size, 3, image_size, image_size],
            requires_grad=True,
            device=self.device,
        )
        dummy_y = real_y.to(self.device)
        opt = Adam([dummy_x], lr=self.lr)
        # 3/8 5/8 7/8
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

        t0 = time.time()
        try:
            for i in range(self.num_epochs):
                loss_cd = loss_tv = torch.tensor(0.0, device=self.device)

                opt.zero_grad()
                pred = fl_model(dummy_x)
                dummy_loss = criterion(pred, dummy_y)
                dummy_grad = torch.autograd.grad(
                    dummy_loss, fl_model.parameters(), create_graph=True
                )
                loss_cd = cosine_distance(dummy_grad, gradient)
                if self.tv > 0:
                    loss_tv = total_variation(dummy_x)
                loss = loss_cd + (self.tv * loss_tv)
                # 记录信息
                self.other_infos = {
                    "epochs": i + 1,
                    "loss": loss.item(),
                    "loss_cd": loss_cd.item(),
                    "loss_tv": loss_tv.item(),
                    "time": time.time() - t0,
                }
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"IG ep {i} : 损失计算异常！")
                    break
                loss.backward()
                opt.step()
                sch.step()
                # 输入更新限制到合理取值范围0~1
                if self.input_boxed:
                    dummy_x.data = dummy_x.data.detach().clamp(0, 1)
                loop_log.log(info_func=lambda: self.other_infos)
        except KeyboardInterrupt:
            logger.info("提前终止IG迭代")
        self.other_infos.update(
            {
                "num_epochs": self.num_epochs,
                "lr": self.lr,
                "tv": self.tv,
                "input_boxed": self.input_boxed,
            }
        )
        with torch.no_grad():
            recover_x = dummy_x.data.detach().clamp(0, 1)
            return recover_x
