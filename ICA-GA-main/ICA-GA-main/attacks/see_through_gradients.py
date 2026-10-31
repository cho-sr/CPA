# --------------------------------------------------
# See through Gradients: Image Batch Recovery via GradInversion
# https://arxiv.org/abs/2104.07586
# --------------------------------------------------

import time
import math
import torch
from torch.optim import Adam

from .base_attack import BaseAttack
from utils import logger, LoopLog
from .opt_common.grad_loss import l2, cosine_distance
from .opt_common.regularization import total_variation
from utils import ParameterListDict
from models.fl import BaseCnnModel
from utils.bn_statistics_hook import BNStatisticsHook


def cosine_decay_lr(optimizer, epochs):

    def lr_lambda(epoch):
        warmup_iters = 50
        if epoch < warmup_iters:
            return 1.0
        else:
            decay_iters = epochs - warmup_iters
            progress = (epoch - warmup_iters) / decay_iters
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


class STG(BaseAttack):
    def __init__(
        self,
        num_epochs=24000,  # 搜索迭代次数
        lr=0.1,  # 优化学习率
        # 注意：l2和cd最好只有一个>0
        l2=0.001,  # l2损失系数
        cd=0.0,  # 梯度余弦相似度损失系数
        tv=0.1,  # TV正则化系数
        bn=0.001,  # BN正则化系数
        input_boxed: bool = True,  # True: 每次输入更新都限制到合理取值范围0~1
        device="cpu",
    ):
        super().__init__(device)
        self.num_epochs = num_epochs
        self.lr = lr
        self.l2 = l2
        self.cd = cd
        self.tv = tv
        self.bn = bn
        self.input_boxed = input_boxed

    def run(
        self,
        fl_model,
        batch_size: int,
        image_size: int,
        data: dict,
    ):
        self.other_infos = {}
        real_y = data["y"]
        gradient = data["gradient"].to(self.device)
        bn_statistics = data["bn_statistics"]
        fl_model = fl_model.to(self.device)
        bn_hook = BNStatisticsHook(model=fl_model)
        logger.info(
            f"STG:  lr={self.lr}, l2={self.l2}, cd={self.cd}, tv={self.tv}, bn={self.bn}"
        )
        loop_log = LoopLog("STG", self.num_epochs)
        # 伪样本
        dummy_x = torch.randn(
            size=[batch_size, 3, image_size, image_size],
            requires_grad=True,
            device=self.device,
        )
        dummy_y = real_y.to(self.device)
        opt = Adam([dummy_x], lr=self.lr)
        sch = cosine_decay_lr(opt, self.num_epochs)

        criterion = torch.nn.CrossEntropyLoss()

        t0 = time.time()
        try:
            for i in range(self.num_epochs):
                bn_hook.clear()
                bn_hook.register()
                loss_l2 = loss_cd = loss_tv = torch.tensor(0.0, device=self.device)
                loss_bn = torch.tensor(  # BN需要累加，单独开一个张量
                    0.0, device=self.device
                )
                opt.zero_grad()
                fl_model.zero_grad()
                pred = fl_model(dummy_x)
                dummy_loss = criterion(pred, dummy_y)
                dummy_grad = torch.autograd.grad(
                    dummy_loss, fl_model.parameters(), create_graph=True
                )
                dummy_bn_sts = bn_hook.mean_var_list
                bn_hook.unregister()
                # 求损失
                if self.l2 > 0:
                    loss_l2 = l2(dummy_grad, gradient)
                if self.cd > 0:
                    loss_cd = cosine_distance(dummy_grad, gradient)
                if self.tv > 0:
                    loss_tv = total_variation(dummy_x)
                if self.bn > 0:
                    bn_mean, bn_var = bn_statistics[0], bn_statistics[1]
                    dummy_bn_mean, dummy_bn_var = dummy_bn_sts[0], dummy_bn_sts[1]
                    loss_bn += l2(dummy_bn_mean, bn_mean)
                    loss_bn += l2(dummy_bn_var, bn_var)

                loss = (
                    self.l2 * loss_l2
                    + self.cd * loss_cd
                    + self.tv * loss_tv
                    + self.bn * loss_bn
                )
                # 记录信息
                self.other_infos = {
                    "epochs": i + 1,
                    "loss": loss.item(),
                    "loss_l2": loss_l2.item(),
                    "loss_cd": loss_cd.item(),
                    "loss_tv": loss_tv.item(),
                    "loss_bn": loss_bn.item(),
                    "time": time.time() - t0,
                }
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"STG ep {i} : 损失计算异常！")
                    break
                loss.backward()
                opt.step()
                sch.step()
                # 伪输入更新限制到合理取值范围0~1
                if self.input_boxed:
                    dummy_x.data = dummy_x.data.detach().clamp(0, 1)
                loop_log.log(info_func=lambda: self.other_infos)
        except KeyboardInterrupt:
            logger.info("提前终止STG迭代")
        self.other_infos.update(
            {
                "num_epochs": self.num_epochs,
                "lr": self.lr,
                "l2": self.l2,
                "cd": self.cd,
                "tv": self.tv,
                "bn": self.bn,
                "input_boxed": self.input_boxed,
            }
        )
        with torch.no_grad():
            recover_x = dummy_x.data.detach().clamp(0, 1)
            return recover_x
