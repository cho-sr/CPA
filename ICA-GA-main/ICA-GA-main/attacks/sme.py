# Surrogate Model Extension: A Fast and Accurate Weight Update Attack on Federated Learning
# https://arxiv.org/abs/2306.00127
# https://github.com/JunyiZhu-AI/surrogate_model_extension

import time
import copy
import torch

from .base_attack import BaseAttack
from utils import logger
from utils.utils import cosine_distance, total_variation


# 更新代理模型 wT
def _update_wT(
    wT,  # 代理模型
    w0,  # 初始模型
    params_update,  # 模型更新
    alpha,  # 插值系数
):
    with torch.no_grad():
        for pT, p0, pu in zip(
            wT.parameters(),
            w0.parameters(),
            params_update,
        ):
            pT.data = p0 + alpha * pu


def run_sme(
    w0,  # t0 时刻的原始模型
    params_update,  # t1 时刻的模型更新
    labels,  # 标签
    batch_size,
    num_epochs=5000,  # 优化次数
    image_size=224,
    lr=1,  # 伪样本的学习率
    alpha_lr=0.001,  # alpha的学习率
    alpha_0=0.5,  # 插值参数 alpha 初始值
    tv=0.005,  # TV正则化系数
    lr_decay=True,  # 是否启用 学习率动态调整
    is_log=True,  # 输出日志
    device="cpu",
):
    # 初始化 alpha
    alpha = torch.tensor(alpha_0, requires_grad=True, device=device)
    alpha.grad = torch.tensor(0.0).to(device)
    # 创建伪数据集，为正态分布的0~1的随机数。
    # 伪样本
    dummy_images = torch.normal(
        0,
        1,
        size=[batch_size, 3, image_size, image_size],
        requires_grad=True,
        device=device,
    )
    # 伪数据集的优化器与学习率调度器
    optimizer = torch.optim.Adam(params=[dummy_images], lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            num_epochs // 2.667,
            num_epochs // 1.6,
            num_epochs // 1.142,
        ],
        gamma=0.1,
    )
    # alpha的优化器与学习率调度器
    alpha_optimizer = torch.optim.Adam(params=[alpha], lr=alpha_lr)
    alpha_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        alpha_optimizer,
        milestones=[
            num_epochs // 2.667,
            num_epochs // 1.6,
            num_epochs // 1.142,
        ],
        gamma=0.1,
    )
    # 设置损失函数 交叉熵损失
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    # 用于反演攻击的代理模型 wT ，位于 t0~t1 时刻连线上、 alpha 处
    with torch.no_grad():
        wT = copy.deepcopy(w0).to(device)
        w0 = copy.deepcopy(w0).to(device)
    wT.eval()
    w0.eval()
    labels = labels.to(device)

    # =============== 开始攻击 ===============
    t0 = time.time()
    log_eps = num_epochs // 10
    for i in range(num_epochs):
        loss_cd = loss_tv = torch.tensor(0.0, device=device)
        optimizer.zero_grad()
        pred = wT(dummy_images)
        _update_wT(wT, w0, params_update, alpha)  # 更新代理模型
        wT.zero_grad()
        # 一轮推理，获取伪梯度
        pred = wT(dummy_images)
        dummy_loss = criterion(input=pred, target=labels)
        # 计算代理模型的 [伪梯度] 与 [模型更新] 的余弦相似度
        dummy_grad = torch.autograd.grad(dummy_loss, wT.parameters(), create_graph=True)
        loss_cd = cosine_distance(dummy_grad, params_update)
        # TV 正则
        if tv > 0:
            loss_tv = total_variation(dummy_images)
        # 总损失
        loss = loss_cd + (tv * loss_tv)
        loss.backward()  # 计算虚拟输入 self.x 的更新梯度
        optimizer.step()
        # 计算 alpha 梯度、更新 alpha
        with torch.no_grad():
            for pT, p0, pu in zip(
                wT.parameters(),
                w0.parameters(),
                params_update,
            ):
                zm = pT.grad.mul(pu).sum()
                alpha.grad += zm
        alpha_optimizer.step()
        with torch.no_grad():
            alpha.data = torch.clamp(alpha, 0, 1)
            dummy_images.data = torch.clamp(dummy_images, 0, 1)
        # 动态调整学习率
        if lr_decay:
            scheduler.step()
            alpha_scheduler.step()

        if is_log and (i == 0 or i % log_eps == log_eps - 1 or i == num_epochs - 1):
            t1 = time.time()
            logger.info(
                f"SME:{i+1:>5}, loss: {loss:.4f}, loss_cd: {loss_cd:.4f}, loss_tv: {loss_tv:.4f}, alpha: {alpha:.4f}, time: {t1-t0:.4f}s"
            )

    return dummy_images.data.detach()


class SME(BaseAttack):
    def __init__(
        self,
        num_epochs=5000,  # 优化次数
        image_size=224,
        lr=1,  # 伪样本的学习率
        alpha_lr=0.001,  # alpha的学习率
        alpha_0=0.5,  # 插值参数 alpha 初始值
        tv=0.005,  # TV正则化系数
        lr_decay=True,  # 是否启用 学习率动态调整
        is_log=True,  # 输出日志
        device="cpu",
    ):
        super().__init__(device)
        self.num_epochs = num_epochs
        self.image_size = image_size
        self.lr = lr
        self.alpha_lr = alpha_lr
        self.alpha_0 = alpha_0
        self.tv = tv
        self.lr_decay = lr_decay
        self.is_log = is_log

    def run(
        self,
        grads,  # 梯度，或参数更新
        x,  # 真样本，用于比较
        y,  # 真标签
        fl_model,  # FL全局模型
        fc_1_w_grad,  # 第1个FC层权重梯度
        fc_1_b_grad,  # 第1个FC层偏置梯度
        fc_1_input,  # 第1个FC层输入
        batch_size,  # 批大小
    ):
        dummy_x = run_sme(
            fl_model,  # w0
            grads,  # params_update
            y,
            batch_size,
            num_epochs=self.num_epochs,
            image_size=self.image_size,
            lr=self.lr,
            alpha_lr=self.alpha_lr,
            alpha_0=self.alpha_0,
            tv=self.tv,
            lr_decay=self.lr_decay,
            is_log=self.is_log,
            device=self.device,
        )
        return dummy_x
