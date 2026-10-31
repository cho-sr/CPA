# --------------------------------------------------
# FedAVG 攻击实验
# --------------------------------------------------

"""
python3 exps/fedavg_attacks/fedavg_attacks.py \
    -result_dir=results\
    -result_name=result.csv\
    -fl_model_load.load_dir=./data/fl_models\
    -fl_model_load.load_name=ResNet50_no-avgpool_out-100_cifar100_fine-tuned\
    -fl_model_load.load_fc=False\
    -fl_model_kwargs.output_length=1000\
    -attacks.ICA-GA.model_save_dir=./data/gen_models\
    -attacks.ICA-GA.model_save_name=gen_1x-CT-RCSE_r50-fine-tuned\
    -data_iter_kwargs.dataset_name=cifar100\
    -data_iter_kwargs.batch_size=8\
    -data_iter_kwargs.local_steps=4\
    -data_iter_kwargs.local_epochs=1\
    -test_batchs=10\
    -device=cuda:0\
    -seed=42\
    -data_iter_kwargs.dataloader_seed=42\
    -test_batchs=10\
    -device=cuda:0
"""

import os
import sys
from typing import List, Dict, Type

# 重置工作目录
current_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.dirname(os.path.dirname(current_dir))
os.chdir(work_dir)
sys.path.insert(0, work_dir)

from utils.datasets import get_dataset_info
from utils.fl_data_iter import data_iter_fedavg
from utils.log import logger
from utils import Recorder, visualize, set_seed, dict_to_str, load_params
from models.fl import fl_models, BaseCnnModel
from attacks import gla_attacks, BaseAttack
from utils.evaluation import init_lpips, analyze_experiment_data

# =============== 实验参数 ===============
exp_params = {
    "result_dir": "results",  # 结果保存目录，相对于本文件目录
    "result_name": "result_atk_fedsgd.csv",  # 结果保存文件名称
    "seed": 1,  # 随机种子
    "device": "cuda:0",  # 主设备。字典中所有为空的 "device" 字段会被设为主设备
    # ========== FL模型 ==========
    "fl_model": "ResNet50",  # 支持: ResNet18, ResNet50
    "fl_model_kwargs": {
        "fc_layers": [],  # 除了最后的分类层外，每个FC层的输出宽度
        "is_pretrain": False,  # 是否预训练
        "is_avgpool": False,  # 是否保留特征提取模块最后的平均池化层
        "output_length": -1,  # 分类数，-1表示自动根据数据集设定
    },
    # 可选，加载预训练的FL模型
    "fl_model_load": {
        "load_dir": "./data/fl_models",  # 可选，加载预训练的FL模型的目录
        "load_name": "ResNet50_no-avgpool_out-100_cifar100_fine-tuned",  # 可选，加载预训练的FL模型的名称。无需.pth后缀
        "load_feat": True,  # 是否加载特征提取模块
        "load_fc": True,  # 是否加载FC层模块
        "load_output_length": 100,  # 加载FC层时，指定的输出宽度。-1为与 fl_model_kwargs.output_length 一致。
    },
    # ========== 客户端私有数据集 ==========
    "data_iter_kwargs": {
        # 支持: imagenet, caltech256, cifar10, cifar100, mnist
        "dataset_name": "cifar100",
        "batch_size": 8,  # 本地批大小
        "shuffle": True,  # 是否乱序
        "image_size": 224,  # 图像尺寸
        "train": True,  # 是否为训练集
        "skip_index": 0,  # 跳过前几个批次
        "dataloader_seed": 1,  # 数据加载器随机种子
        # 客户端设置
        "local_steps": 8,  # 批次数量
        "local_epochs": 1,  # 本地迭代轮次
        "local_lr": 0.001,
        "device": "",  # 训练设备，留空则设为主设备
    },
    "test_batchs": 100,  # 最多测试多少批数据
    # ========== 评估 ==========
    "evaluation": {
        "device": "",  # 留空则设为主设备
    },
    # ========== 攻击 ==========
    "attacks": {
        "ICA-GA": {
            "generator_model": "ModGen",
            "model_kwargs": {
                # 模型结构
                "num_res_blocks_per_stage": 1,  # 残差块数量
                "res_block_type": "ResConvSE",  # 残差块类型
                "upsample_type": "ConvTranspose",  # 上采样层类型
            },
            "model_save_dir": "./data/gen_models",
            "model_save_name": "gen_1x-CT-RCSE_r50-fine-tuned",
            "device": "",
        },
        "FGLA": {
            "generator_model": "ModGen",
            "model_kwargs": {
                # 模型结构
                "num_res_blocks_per_stage": 1,  # 残差块数量
                "res_block_type": "ResConvSE",  # 残差块类型
                "upsample_type": "ConvTranspose",  # 上采样层类型
            },
            "model_save_dir": "./data/gen_models",
            "model_save_name": "gen_1x-CT-RCSE_r50-fine-tuned",
            "offset": True,  # 抵消量
            "standardization": True,  # 启用标准化
            "device": "",
        },
        # "IG": {
        #     "num_epochs": 24000,  # 迭代次数
        #     "lr": 0.1,
        #     "tv": 0.5,
        #     "device": "",
        # },
        # "CPA": {
        #     "num_epochs": 24000,  # 迭代次数
        #     "lr": 0.1,
        #     "tv": 0.5,
        #     "use_true_feature": True,  # 跳过ICA，使用真实特征
        #     "device": "",
        # },
    },
}


def main(override_params: dict = {}):
    params = load_params(exp_params, __file__, override_params)
    set_seed(params["seed"])
    result_dir = os.path.join(current_dir, params["result_dir"])
    recorder = Recorder(save_path=os.path.join(result_dir, params["result_name"]))
    # fl model, dataset
    data_iter_kwargs = params["data_iter_kwargs"]
    image_size = data_iter_kwargs["image_size"]
    fl_model_cls: Type[BaseCnnModel] = fl_models[params["fl_model"]]
    fl_model_kwargs = params["fl_model_kwargs"]
    fl_model_kwargs["input_width"] = image_size
    fl_model_kwargs["input_height"] = image_size
    if fl_model_kwargs["output_length"] < 1:
        dataset_info = get_dataset_info(data_iter_kwargs["dataset_name"])
        fl_model_kwargs["output_length"] = dataset_info["num_classes"]
    fl_model = fl_model_cls(**fl_model_kwargs)
    output_length = fl_model_kwargs["output_length"]
    # 加载FL模块
    fl_load = params["fl_model_load"]
    if fl_load["load_name"] and (fl_load["load_feat"] or fl_load["load_fc"]):
        load_model_kwargs = {**fl_model_kwargs}
        load_output_length = fl_load["load_output_length"]
        can_load_fc = True
        if load_output_length > 1 and load_output_length != output_length:
            load_model_kwargs["output_length"] = load_output_length
            can_load_fc = False
        _m_loaded = fl_model_cls(**load_model_kwargs)
        _m_loaded.set_save(fl_load["load_dir"], fl_load["load_name"])
        _m_loaded.load_model()
        if fl_load["load_feat"]:
            fl_model.feature_extration = _m_loaded.feature_extration
            logger.debug("FL模型加载特征提取模块参数")
        if fl_load["load_fc"]:
            if can_load_fc:
                logger.debug("FL模型加载FC模块参数")
                fl_model.fully_connected = _m_loaded.fully_connected
            else:
                logger.debug(
                    f"加载分类器输出宽度为{load_output_length}，不适配当前宽度{output_length}，未加载FC模块参数。"
                )
    data_iter = data_iter_fedavg(
        model=fl_model,
        need_dY=True,
        **data_iter_kwargs,
    )
    # 等价批大小
    batch_size = data_iter_kwargs["batch_size"]
    s_size = batch_size * data_iter_kwargs["local_steps"]
    if s_size > 128:
        init_lpips("cpu")
    else:
        init_lpips(params["evaluation"]["device"])
    # attacks
    attacks: Dict[str, BaseAttack] = {}
    for key, kwargs in params["attacks"].items():
        if s_size > 64 and "generator_model" not in kwargs.keys():
            print(f"等价批大小为{s_size}，跳过 {key}")
            continue
        atk_cls = gla_attacks[key]
        atk = atk_cls(**kwargs)
        attacks[key] = atk
    recorder.add_extra(
        {
            "exp": "fedsgd_attacks",
            "image_size": image_size,
            "batch_size": batch_size,
        }
    )
    test_batchs = params["test_batchs"]
    atk_info_dict = {n: [] for n in attacks.keys()}  # 存储每种攻击的信息列表
    images_dir = os.path.join(result_dir, "images")
    test_num = 0
    for data in data_iter:
        test_num += 1
        logger.info("FedAVG 训练结束。")

        visualize(
            batch_tensor=data["x"],
            label="real",
            save_path=os.path.join(images_dir, f"{test_num}_real.png"),
            max_num=64,
        )
        logger.info("预处理完成，开始模拟攻击。")
        S_size = data["y"].shape[0]
        for atk_name, atk in attacks.items():
            logger.info(f"攻击 {atk_name} 开始")
            try:
                assess_info = atk.assess(fl_model, S_size, image_size, data)
            except Exception:
                logger.warning(f"攻击 {atk_name} 异常")
                continue
            recover_x = atk.recover_x
            recorder.add({"test_num": test_num, **assess_info})
            atk_info_dict[atk_name].append(assess_info)
            visualize(
                batch_tensor=recover_x,
                label=atk_name,
                save_path=os.path.join(images_dir, f"{test_num}_{atk_name}.png"),
                max_num=64,
            )
        if test_num >= test_batchs:
            break
    # 计算均值
    for atk_name, infos in atk_info_dict.items():
        info = analyze_experiment_data(infos)
        recorder.add({"test_num": "all", **info})
    logger.info(f"共进行了{test_num}次实验。")


if __name__ == "__main__":
    main()
