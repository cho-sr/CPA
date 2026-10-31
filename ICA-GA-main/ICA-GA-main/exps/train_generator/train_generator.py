# --------------------------------------------------
# 生成器训练实验
# --------------------------------------------------

"""
python3 exps/train_generator/train_generator.py \
    -result_dir=results\
    -fl_model_load.load_dir=./data/fl_models\
    -fl_model_load.load_name=ResNet50_no-avgpool_out-100_cifar100_fine-tuned\
    -generator_model_kwargs.save_dir=./data/gen_models\
    -generator_save_name=gen_1x-CT-RCSE_r50_1\
    -trainer_kwargs.train_dataset=imagenet\
    -trainer_kwargs.test_dataset=cifar100\
    -trainer_kwargs.batch_size=32\
    -trainer_kwargs.num_epochs=100\
    -trainer_kwargs.max_batchs=1000\
    -trainer_kwargs.lr=0.0001\
    -trainer_kwargs.loss.mse=1.0\
    -trainer_kwargs.loss.feat=0.01\
    -trainer_kwargs.loss.tv=0.01\
    -trainer_kwargs.loss.ssim=0.01\
    -trainer_kwargs.save_eval=psnr\
    -trainer_kwargs.save_eval_up=true\
    -trainer_kwargs.stop_eval_value=23\
    -trainer_kwargs.early_stop_patience=10
"""

import os
import sys
import torch.nn as nn

if "utils" not in sys.modules:
    # 重置工作目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.dirname(os.path.dirname(current_dir))
    os.chdir(work_dir)
    sys.path.insert(0, work_dir)

    from utils import (
        set_seed,
        load_params,
        get_current_date_str,
        Recorder,
        flatten_dict,
    )
    from utils.evaluation import init_lpips
    from models.base_model import BaseModel
    from models.fl import fl_models, BaseCnnModel
    from models.generator import generator_models
    from attacks.gen_common.generator_trainer import GeneratorTrainer

# =============== 实验参数 ===============
exp_params = {
    "result_dir": "results",  # 结果保存目录，相对于本文件目录
    "seed": 1,  # 随机种子
    "device": "cuda:0",  # 主设备。字典中所有为空的 "device" 字段会被设为主设备
    # ========== FL模型 ==========
    "fl_model": "ResNet50",  # 支持: ResNet18, ResNet50
    "fl_model_kwargs": {
        "is_pretrain": False,  # 是否预训练
        "is_avgpool": False,  # 必须为False
        "is_bn": True,  # 是否保留BN层。必须为True
        "output_length": 100,
    },
    # 可选，加载预训练的FL模型
    "fl_model_load": {
        "load_dir": "./data/fl_models",  # 可选，加载预训练的FL模型的目录
        "load_name": "",  # 可选，加载预训练的FL模型的名称。无需.pth后缀
    },
    # ========== 生成器模型 ==========
    "generator_model": "ModGen",
    "generator_model_kwargs": {
        "save_dir": "",
        "num_res_blocks_per_stage": 1,
        "res_block_type": "ResConvSE",
        "upsample_type": "ConvTranspose",
        "channels_schedule": [2048, 1024, 512, 256, 128, 64],
    },
    "generator_load_name": "",  # 读取生成器模型文件，可加载已预训练的模型。不填则使用初始化模型
    "generator_save_name": "",  # 生成器模型文件保存名称，不填则为 generator_model 。无需.pth后缀
    "recorder_save_name": "",  # 训练过程日志文件保存名称，不填则为 generator_save_name 。无需.pth后缀
    # ========== 评估 ==========
    "evaluation": {
        "device": "",  # 留空则设为主设备
    },
    # ========== 训练器参数 ==========
    "trainer_kwargs": {
        "train_dataset": "imagenet",  # 训练集
        "test_dataset": "cifar100",  # 测试集
        "image_size": 224,  # 图像边长
        "batch_size": 48,  # 训练批大小
        "shuffle": True,  # 训练时是否乱序
        "num_epochs": 10,  # 总训练轮数
        "max_batchs": float("inf"),  # 单轮训练的最大批数量
        "lr": 0.0001,  # 学习率
        "loss": {  # 各个损失项及其权重
            "mse": 1.0,
            "feat": 0.01,
            "tv": 0.01,
            "ssim": 0.01,
            "lpips": 0.0,
        },
        "save_eval": "psnr",  # 最佳模型文件保存的评估指标
        "save_eval_up": True,  # 指标是否越大越好
        "early_stop_patience": 100,  # 早停耐心值，连续该轮次评估无提升时终止训练。
        "stop_eval_value": -1.0,  # 早停具体值，某一轮达到该值时终止训练。>0
        "pretreatment": "standardization",  # 特征向量预处理策略
        "device": "",  # CPU、单GPU模式、多GPU模式的主GPU
        "use_multi_gpu": False,  # 多GPU模式
        "device_ids": [],  # 多GPU模式下使用的GPU编号，None为使用所有GPU
    },
}


def main(override_params: dict = {}):
    params = load_params(exp_params, __file__, override_params)
    set_seed(params["seed"])
    result_dir = os.path.join(current_dir, params["result_dir"])
    init_lpips(params["evaluation"]["device"])

    # 初始化编码器，解码器
    image_size = params["trainer_kwargs"]["image_size"]
    fl_model_kwargs = params["fl_model_kwargs"]
    fl_model_kwargs.update(
        {
            "input_channels": 3,
            "input_width": image_size,
            "input_height": image_size,
        }
    )
    encoder: BaseCnnModel = fl_models[params["fl_model"]](**fl_model_kwargs)
    # 加载FL模型（编码器）
    fl_load = params["fl_model_load"]
    if fl_load["load_name"]:
        encoder.set_save(fl_load["load_dir"], fl_load["load_name"])
        encoder.load_model()
    encoder.fully_connected = nn.Sequential()
    decoder: BaseModel = generator_models[params["generator_model"]](
        **params["generator_model_kwargs"]
    )
    if params["generator_load_name"]:
        load_path = (
            os.path.join(decoder.save_dir, params["generator_load_name"]) + ".pth"
        )
        decoder.load_model(load_path)
    # 初始化训练参数
    trainer_kwargs = params["trainer_kwargs"]
    decoder_save_name = params["generator_save_name"]
    if not decoder_save_name:
        decoder_save_name = f'{params["generator_model"]}_ds={trainer_kwargs["train_dataset"]}_pr={trainer_kwargs["pretreatment"][:5]}_{get_current_date_str()}'
    trainer_kwargs["decoder_save_name"] = decoder_save_name
    # 记录器
    recorder_save_name = params["recorder_save_name"]
    if not recorder_save_name:
        recorder_save_name = decoder_save_name
    recorder = Recorder(save_path=os.path.join(result_dir, recorder_save_name) + ".csv")
    recorder.add_extra(flatten_dict(params))

    # 启动训练
    trainer = GeneratorTrainer(
        encoder=encoder,
        decoder=decoder,
        recorder=recorder,
        decoder_save_dir=decoder.save_dir,
        **trainer_kwargs,
    )
    info_list = trainer.train()
    return info_list


if __name__ == "__main__":
    main()
