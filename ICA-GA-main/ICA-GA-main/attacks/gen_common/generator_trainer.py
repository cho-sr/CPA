# --------------------------------------------------
# 训练生成器
# --------------------------------------------------

import os
import time
from typing import Union, Optional, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import Recorder, logger, dict_to_str
from utils.datasets import get_dataloader
from utils.evaluation import mse, psnr, ssim, lpips
from .pretreatment import pretreatments
from models.base_model import BaseModel
from models.fl import BaseCnnModel
from .train_loss import loss_dict, LossInputDict


class GeneratorTrainer:
    def __init__(
        self,
        encoder: BaseCnnModel,
        decoder: BaseModel,
        train_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        decoder_save_dir: str = "",
        decoder_save_name: str = "",
        train_dataset: str = "",
        test_dataset: str = "",
        image_size: int = 224,
        batch_size: int = 32,
        shuffle: bool = True,
        num_epochs: int = 10,
        max_batchs: int = float("inf"),
        lr: float = 0.0001,
        loss: dict = {
            "mse": 1.0,
            "feat": 0.0,
            "tv": 0.0,
            "ssim": 0.0,
            "lpips": 0.0,
        },
        is_save: bool = True,
        save_eval: str = "psnr",
        save_eval_up: bool = True,
        early_stop_patience: int = 10,
        stop_eval_value: float = -1,
        pretreatment: str = "standardization",
        device="cuda",
        use_multi_gpu=False,
        device_ids=[],
        recorder_dir: str = "",
        recorder: Optional[Recorder] = None,
    ):
        """生成器训练

        Args:
            encoder (BaseCnnModel): 预训练的编码器模型实例。
            decoder (BaseModel): 待训练的解码器模型实例。即生成器 Generator
            decoder_save_dir (str): 解码器模型保存目录。
            decoder_save_name (str): 解码器模型保存时的文件名标识。如果为空，则不保存。
            recorder_dir (str): 训练日志和结果的保存目录。
            train_dataloader (Optional[DataLoader]): 训练数据加载器。若为None，则根据 `train_dataset` 等参数创建。
            test_dataloader (Optional[DataLoader]): 测试数据加载器。若为None，则根据 `test_dataset` 等参数创建。
            train_dataset (str): 训练数据集的名称或路径，用于 `get_dataloader`。默认为空字符串。
            test_dataset (str): 测试数据集的名称或路径，用于 `get_dataloader`。默认为空字符串。
            image_size (int): 输入图像的尺寸 (假设为方形)。默认为 224。
            batch_size (int): 训练和测试时的批量大小。默认为 32。
            shuffle (bool): 是否在每个epoch开始时打乱训练数据。默认为 True。
            num_epochs (int): 训练的总轮数。
            max_batchs (int): 每轮训练中处理的最大批次数。默认为 `float("inf")`，即处理完整个数据集。
            lr (float): 解码器训练的学习率。默认为 0.0001。
            loss (dict): 要使用的损失项及权重。损失项key必须在 train_loss.loss_dict 中定义。<=0的权重将被忽略。
            is_save (bool): 是否保存文件
            save_eval (str): 用于选择最佳模型的评估指标名称 ("mse", "ssim", "psnr", "lpips")。若为空字符串，则每轮都保存模型。默认为 "psnr"。
            save_eval_up (bool): 指示 `save_eval` 指定的指标是否越大越好。True表示越大越好 (如psnr, ssim)，False表示越小越好 (如mse, lpips)。默认为 True。
            early_stop_patience (int): 早停轮数，必须设置save_eval才有效。当连续epochs轮指标没有提升时，提前结束训练。<=0则不进行早停
            stop_eval_value (float): 早停值。指标大于该值时，无视早停轮数，直接停止。>0时有效
            pretreatment (str): 一个可选的函数名，用于在特征向量输入解码器前对其进行预处理。默认为 `standardization`。
            device (str): 指定训练设备 ("cuda" 或 "cpu")。默认为 "cuda"。多GPU模式下设为主GPU
            use_multi_gpu (bool): 是否使用多个GPU进行训练 (如果可用)。默认为 False。
            device_ids (Optional[list]): 指定使用的GPU设备ID列表 (例如 `[0, 1]`)。如果 `use_multi_gpu` 为True且此参数为None，则使用所有可用的GPU。默认为 None。
            recorder (Optional[Recorder]): 可选，日志记录器实例。
        """

        # 处理模型
        encoder.train()  # 训练模式，开启BN
        encoder.set_requires_grad(False)  # 无需更新编码器参数
        decoder.set_save(decoder_save_dir, decoder_save_name)
        self.use_multi_gpu = use_multi_gpu
        if use_multi_gpu and torch.cuda.device_count() > 1:
            if device_ids:
                logger.info(f"Using GPUs {device_ids} for training.")
            else:
                logger.info(f"Using {torch.cuda.device_count()} GPUs for training.")
            self.encoder = nn.DataParallel(encoder, device_ids).to(device)
            self.decoder = nn.DataParallel(decoder, device_ids).to(device)
        else:
            logger.info(f"Using {device} for training.")
            self.encoder = encoder.to(device)
            self.decoder = decoder.to(device)
        # 处理数据集
        if train_dataloader is None:
            self.train_dataloader = get_dataloader(
                train_dataset, batch_size, shuffle, image_size, train=True
            )
        else:
            self.train_dataloader = train_dataloader
        if test_dataloader is None:
            if test_dataset:
                self.test_dataloader = get_dataloader(
                    test_dataset, batch_size, False, image_size, train=False
                )
            else:
                self.test_dataloader = None
        else:
            self.test_dataloader = test_dataloader
        # 加载损失项
        self.loss = {}
        for key, weight in loss.items():
            if key not in loss_dict:
                raise ValueError(
                    f"损失项 {key} 未被定义。允许的损失项： {loss_dict.keys()}"
                )
            if weight > 0:
                self.loss[key] = {
                    "weight": weight,
                    "func": loss_dict[key](device=device),
                }
        if not self.loss:
            raise ValueError(f"未定义任何损失项")
        # 处理训练参数
        if pretreatment and pretreatment in pretreatments:
            self.pretreatment = pretreatments[pretreatment]
        else:
            self.pretreatment = None
        self.is_save = is_save
        self.lr = lr
        self.optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.device = device
        self.max_batchs = max_batchs
        self.num_epochs = num_epochs
        self.save_eval_up = save_eval_up
        self.save_eval = save_eval
        # 最好的指标
        self.save_eval_best = float("-inf") if self.save_eval_up else float("inf")
        self.save_eval_epoch = 0  # 最好指标出现在第几轮
        self.early_stop_patience = early_stop_patience  # 早停耐心值
        self.stop_eval_value = stop_eval_value  # 早停具体值
        # 缓存最后一次测试的样本和重建样本，可用于可视化展示
        self.last_x = None
        self.last_recover_x = None
        # 日志记录器
        if recorder is None:
            self.recorder = Recorder(
                save_path=os.path.join(recorder_dir, decoder_save_name) + ".csv"
            )
            train_parameters = {
                "train_dataset": train_dataset,
                "test_dataset": test_dataset,
                "image_size": image_size,
                "batch_size": batch_size,
                "max_batchs": max_batchs if max_batchs < float("inf") else "inf",
                "num_epochs": num_epochs,
                "shuffle": shuffle,
                "save_eval": save_eval,
                "save_eval_up": save_eval_up,
                "pretreatment": (
                    self.pretreatment.__name__ if self.pretreatment is not None else ""
                ),
                "lr": lr,
                "encoder": type(encoder).__name__,
                "decoder": type(decoder).__name__,
            }
            for key, value in self.loss.items():
                train_parameters[f"loss_{key}_weight"] = value["weight"]
            self.recorder.add_extra(train_parameters)
        else:
            self.recorder = recorder

    def test(self) -> dict:
        """在测试数据集上评估当前解码器模型的性能。

        Returns:
            dict: 包含评估结果的字典，具体键值对如下：
                - "mse" (float): 所有测试样本的平均均方误差 (Mean Squared Error)。
                - "psnr" (float): 所有测试样本的平均峰值信噪比 (Peak Signal-to-Noise Ratio)。
                - "ssim" (float): 所有测试样本的平均结构相似性指数 (Structural Similarity Index)。
                - "lpips" (float): 所有测试样本的感知相似度指数 (Learned Perceptual Image Patch Similarity)。
                - "test_time" (float): 完成整个测试过程所需的时间（秒）。
        """
        assert (
            self.test_dataloader is not None
        ), "训练器未加载 test_dataloader ，无法执行测试"
        try:
            self.decoder.eval()
            total_mse = 0.0
            total_ssim = 0.0
            total_psnr = 0.0
            total_lpips = 0.0
            t0 = time.time()
            with torch.no_grad():
                for x, _ in self.test_dataloader:
                    x = x.to(self.device)  # 样本
                    feat = self.encoder(x)  # 获取特征向量
                    if self.pretreatment is not None:
                        feat = self.pretreatment(feat)
                    re_x = self.decoder(feat)  # 重建的样本
                    total_mse += mse(re_x, x)  # 获取一批样本的平均MSE
                    total_ssim += ssim(re_x, x)
                    total_psnr += psnr(re_x, x)
                    total_lpips += lpips(re_x, x)
            batchs = len(self.test_dataloader)
            mse_value = total_mse / batchs
            ssim_value = total_ssim / batchs
            psnr_value = total_psnr / batchs
            lpips_value = total_lpips / batchs
            t1 = time.time()
            info = {
                "mse": mse_value,
                "psnr": psnr_value,
                "ssim": ssim_value,
                "lpips": lpips_value,
                "test_time": t1 - t0,
            }
            self.last_x = x.cpu()
            self.last_recover_x = re_x.cpu()
            return info
        except KeyboardInterrupt:
            logger.info("提前终止测试")
            return {}

    def train_epoch(self) -> dict:
        """执行一个训练轮次

        Returns:
            dict: 包含该轮次训练信息的字典，具体键值对如下：
                - "train_time" (float): 完成该轮训练所需的时间（秒）。
                - "batchs" (int): 该轮实际处理的批次数。
                - "loss" (float): 该轮训练的平均损失。
                - "forced_stop" (bool, 可选): 如果训练被 `KeyboardInterrupt` 中断，则此键存在且值为True。
        """
        if isinstance(self.decoder, nn.DataParallel):
            self.decoder.module.set_requires_grad(True)
        else:
            self.decoder.set_requires_grad(True)
        self.decoder.train()
        t0 = time.time()
        running_loss = {
            "loss": 0,
        }
        for key in self.loss.keys():
            running_loss[f"loss_{key}"] = 0.0
            running_loss[f"loss_{key}_w"] = 0.0
        batchs = 0
        forced_stop = False
        memory_allocated = 0.0  # 显存占用
        try:
            for i, (x, _) in enumerate(self.train_dataloader):
                x = x.to(self.device)  # 样本
                with torch.no_grad():  # 获取特征向量
                    raw_feat = self.encoder(x)  # 原始特征
                    if self.pretreatment is None:  # 无后处理
                        feat = raw_feat
                    else:  # 有后处理
                        feat = self.pretreatment(raw_feat)
                re_x = self.decoder(feat)  # 重建的样本
                # 计算损失
                loss_input_dict: LossInputDict = {
                    "re_x": re_x,
                    "real_x": x,
                    "raw_feat": raw_feat,
                    "generator_trainer": self,
                }
                total_loss: torch.Tensor = None
                for key, value in self.loss.items():
                    _loss = value["func"](loss_input_dict)
                    _loss_w = _loss * value["weight"]
                    if total_loss is None:
                        total_loss = _loss_w
                    else:
                        total_loss += _loss_w
                    running_loss[f"loss_{key}"] += _loss.item()
                    running_loss[f"loss_{key}_w"] += _loss_w.item()
                if total_loss is None:
                    raise ValueError("total_loss 未被计算")
                running_loss["loss"] += total_loss.item()

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                # 显存占用，单位为GB
                if i == 0 and not self.use_multi_gpu and self.device != "cpu":
                    memory_allocated = (
                        torch.cuda.memory_allocated(device=self.device) / 1024**3
                    )
                batchs = i + 1
                if batchs >= self.max_batchs:
                    break
        except KeyboardInterrupt:
            forced_stop = True
            logger.info("提前终止训练")
        t1 = time.time()
        for k, v in running_loss.items():
            running_loss[k] = v / batchs
        info = {
            "train_time": t1 - t0,
            "batchs": batchs,
            "memory_allocated": memory_allocated,
        }
        info.update(running_loss)
        if forced_stop:
            info["forced_stop"] = True
        return info

    # 训练解码器
    def train(self) -> list:
        """执行完整的解码器训练流程

        Returns:
            list[dict]: 一个列表，其中每个元素是一个字典，包含了对应轮次的详细训练和测试信息。
                        字典的键包括 "epoch", "time" (该轮总时间), "is_save" (该轮模型是否被保存),
                        以及从 `train_epoch` 和 `test` 返回的所有键值对。
        """
        logger.info("开始训练。")
        info_list = []
        for epoch in range(self.num_epochs):
            # 训练、测试
            t0 = time.time()
            train_info = self.train_epoch()
            test_info = self.test()
            t = time.time() - t0
            is_save = False  # 是否进行保存
            if self.is_save:
                if not self.save_eval:
                    is_save = True
                elif self.save_eval and test_info:
                    # 有指标，且有测试结果，则检查
                    eval = test_info[self.save_eval]
                    if (self.save_eval_up and eval > self.save_eval_best) or (
                        not self.save_eval_up and eval < self.save_eval_best
                    ):  # 遇到了更好指标
                        self.save_eval_best = eval
                        self.save_eval_epoch = epoch
                        is_save = True
                if is_save:
                    if isinstance(self.decoder, nn.DataParallel):
                        self.decoder.module.save_model()
                    else:
                        self.decoder.save_model()
            # 保存、打印信息
            info = {
                "epoch": epoch,
                "time": t,
                "is_save": is_save,
            }
            info.update(test_info)
            info.update(train_info)
            info_list.append(info)
            self.recorder.add(info, is_log=False)
            logger.info(
                "训练轮："
                + dict_to_str(
                    {
                        "epoch": epoch,
                        "mse": test_info["mse"],
                        "psnr": test_info["psnr"],
                        "ssim": test_info["ssim"],
                        "lpips": test_info["lpips"],
                        "time": t,
                    }
                )
                if test_info
                else f"epoch: {epoch:>4}"
            )
            # 手动停止
            if "forced_stop" in train_info or not test_info:
                break
            # 早停
            if (
                self.early_stop_patience > 0
                and self.save_eval
                and epoch >= self.save_eval_epoch + self.early_stop_patience
            ):
                break
            if self.stop_eval_value > 0 and (
                (self.save_eval_up and eval >= self.stop_eval_value)
                or (not self.save_eval_up and eval <= self.stop_eval_value)
            ):
                break
        return info_list
