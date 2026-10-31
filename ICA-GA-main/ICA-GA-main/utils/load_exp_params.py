import sys
import os
import copy
from typing import Any
from utils import logger

try:
    import json5 as json
except ImportError:
    logger.warning(
        f'警告：未加载json5库。如需使用含有注释的json配置文件，请安装： "pip install json5"'
    )
    import json


def _convert_value(default_value: Any, new_value_str: str):
    """
    将字符串新值转换为与默认值相同的类型。

    Args:
        default_value (Any): 默认值，用于确定目标类型。
        new_value_str (str): 字符串形式的新值。

    Returns:
        转换后的新值。

    Raises:
        ValueError: 如果无法将 new_value_str 转换为 default_value 的类型。
    """

    # 如果默认值是None，则接受默认的字符串
    if default_value is None:
        return new_value_str

    target_type = type(default_value)
    if target_type == bool:
        if new_value_str.lower() == "true":
            return True
        elif new_value_str.lower() == "false":
            return False
        else:
            raise ValueError(
                f"无法将 '{new_value_str}' 转换为布尔类型。请使用 'true' 或 'false'。"
            )
    try:
        return target_type(new_value_str)
    except ValueError:
        raise ValueError(
            f"无法将 '{new_value_str}' 转换为类型 {target_type.__name__}。"
        )


def _update_params_from_cli_recursively(
    params_dict: dict, key_parts: list, value_str: str, default_template: dict
):
    """
    递归地更新参数字典 (用于命令行参数，需要字符串转换)。

    Args:
        params_dict: 当前要更新的参数字典部分。
        key_parts: 由 '.' 分割的键的列表。
        value_str: 字符串形式的新值。
        default_template: 对应于 params_dict 结构的默认参数模板。

    Raises:
        KeyError: 如果传入了未在默认参数中设计的键。
        TypeError: 如果 params_dict 或 default_template 中的某个层级不是字典，但 key_parts 仍有剩余。
        ValueError: 如果值类型转换失败。
    """
    key = key_parts[0]

    if key not in default_template:
        raise KeyError(
            f"参数 '{key}' 未在默认参数的相应层级 '{'.'.join(default_template.keys())}' 中定义。"
        )

    if len(key_parts) == 1:  # 到达最内层键
        default_val = default_template[key]
        if default_val is None:
            logger.warning(
                f'模板字典 {key} 的默认值为None。建议使用带有具体类型的空默认值，如 0.0 、 "" 等。'
            )
        params_dict[key] = _convert_value(default_val, value_str)
    else:  # 需要进一步递归
        if not isinstance(default_template[key], dict):
            raise TypeError(f"试图访问嵌套参数 '{key}'，但它在默认参数中不是一个字典。")

        if key not in params_dict:
            params_dict[key] = {}

        _update_params_from_cli_recursively(
            params_dict[key], key_parts[1:], value_str, default_template[key]
        )


def _merge_and_validate_params_from_dict(
    target_dict: dict,
    source_dict: dict,
    default_template_dict: dict,
    source_name: str,
    current_path: str = "",
):
    """
    递归地将 source_dict 中的参数合并到 target_dict，并根据 default_template_dict 进行验证和类型转换。

    Args:
        target_dict (Dict): 要更新的目标参数字典。
        source_dict (Dict): 包含新参数的源字典。
        default_template_dict (Dict): 默认参数模板，用于验证键和类型。
        source_name (str): 源参数的名称 (例如 "JSON文件", "传入字典")，用于错误消息。
        current_path (str, optional): 当前递归路径，用于构建完整的参数键名。

    Raises:
        KeyError: source_dict 中的键未在 default_template_dict 中定义。
        ValueError: 值的类型与默认模板不匹配且无法转换。
        TypeError: 尝试在非字典类型的参数上设置嵌套值。
    """
    for key, value in source_dict.items():
        param_path = f"{current_path}.{key}" if current_path else key

        if key not in default_template_dict:
            if "JSON" in source_name:
                target_dict[key] = value
                logger.debug(f"json文件增加参数 {key}: {value}")
                continue
            raise KeyError(
                f"来自 {source_name} 的参数 '{param_path}' 未在默认参数中定义。"
            )

        default_val_for_conversion = default_template_dict[key]

        if isinstance(value, dict) and isinstance(default_val_for_conversion, dict):
            # 确保 target_dict 中的相应路径也是一个字典
            if key not in target_dict or not isinstance(target_dict[key], dict):
                target_dict[key] = {}
            _merge_and_validate_params_from_dict(
                target_dict[key],
                value,
                default_template_dict[key],
                source_name,
                param_path,
            )
        else:
            # 类型验证和转换
            try:
                if default_val_for_conversion is not None and value is not None:
                    if type(value) is not type(default_val_for_conversion):
                        # 以json解析字符串
                        if isinstance(value, str) and (
                            isinstance(default_val_for_conversion, list)
                            or isinstance(default_val_for_conversion, dict)
                        ):
                            if (value.startswith("[") and value.endswith("]")) or (
                                value.startswith("{") and value.endswith("}")
                            ):
                                try:
                                    converted_value = json.loads(value)
                                    target_dict[key] = converted_value
                                except json.JSONDecodeError:
                                    raise ValueError(
                                        f"{source_name} 参数 '{param_path}' 是列表或字典，"
                                        f"但输入值无法json转换: {value}"
                                    )
                        # 尝试强制转换
                        try:
                            converted_value = type(default_val_for_conversion)(value)
                            target_dict[key] = converted_value
                        except (ValueError, TypeError) as e:
                            raise ValueError(
                                f"{source_name} 参数 '{param_path}' 的类型 ({type(value).__name__}) "
                                f"与默认类型 ({type(default_val_for_conversion).__name__}) 不匹配且无法转换: {e}"
                            )
                    else:
                        target_dict[key] = value  # 类型匹配
                elif default_val_for_conversion is None:
                    target_dict[key] = value  # 默认是None，接受源字典的值
                    logger.warning(
                        f'模板字典 {param_path} 的默认值为None。建议使用带有具体类型的空默认值，如 0.0 、 "" 等。'
                    )
                elif value is None and default_val_for_conversion is not None:
                    # 源字典中是None，但默认值不是None，表示想设为None
                    target_dict[key] = None
                else:  # 值不是None，但default_val_for_conversion是None。或者都是None
                    target_dict[key] = value
            except ValueError as e:
                raise ValueError(
                    f"处理来自 {source_name} 的参数 '{param_path}' 时出错: {e}"
                )


def _set_main_device(target_dict: Any, main_device: str):
    """将 target_dict 中所有为空的 'device' 设为 main_device"""
    if isinstance(target_dict, dict):
        for key in list(target_dict.keys()):
            value = target_dict[key]
            if key == "device":
                if value == "":
                    target_dict[key] = main_device
            else:
                _set_main_device(value, main_device)
    elif isinstance(target_dict, list):
        for item in target_dict:
            _set_main_device(item, main_device)


def load_params(
    default_params: dict, script_file_path: str = "", override_params: dict = None
):
    """
    加载实验参数，按照以下优先级顺序（从低到高）：
    1. 默认参数字典 (default_params)
    2. JSON文件 (与 script_file_path同路径，通过命令行 -json=xxx.json 指定)
    3. 传入的覆盖参数字典 (override_params)
    4. 命令行参数

    Args:
        default_params (dict): 包含默认参数的字典。
        script_file_path (str, optional): 调用此函数的脚本文件的路径 (例如 __file__)。
                                          如果提供，则会尝试加载命令行中 -json=xxx.json 指定的文件。
                                          不填则忽略json文件加载。
        override_params (dict, optional): 一个字典，用于在JSON文件之后、命令行参数之前覆盖参数。

    Returns:
        dict: 更新后的参数字典。

    Raises:
        FileNotFoundError: 如果脚本文件路径无效 (虽然这里主要用于推断JSON路径)。
        KeyError: 如果传入了未在默认参数中定义的参数。
        ValueError: 如果参数类型无法转换或不匹配。
        TypeError: 如果在尝试设置嵌套参数时，目标不是字典，或默认参数结构不一致。
    """

    # 0. 深拷贝默认参数，作为基础
    loaded_params = copy.deepcopy(default_params)

    # 1. 尝试从JSON文件加载参数
    json_filepath = ""
    arg_json = ""
    if script_file_path:
        # 从命令行中寻找json路径或文件名
        cli_args = sys.argv[1:]  # 跳过脚本名称
        for arg in cli_args:
            if not arg.startswith("-") or "=" not in arg:
                raise ValueError(
                    f"命令行参数格式不合法： '{arg}'。期望格式 -key=value 或 -key.subkey=value。"
                )
            key_value_part = arg[1:]  # 去掉开头的 '-'
            try:
                full_key, value_str = key_value_part.split("=", 1)
            except ValueError:
                raise ValueError(f"命令行参数格式不合法： '{arg}'。'=' 缺失。")
            key_parts = full_key.split(".")
            if full_key == "json":
                arg_json = value_str
                break
    if script_file_path and arg_json:
        if os.path.isabs(arg_json):  # 绝对路径
            json_filepath = arg_json
        else:  # 相对路径
            try:
                script_dir = os.path.dirname(os.path.abspath(script_file_path))
                json_filepath = os.path.join(script_dir, arg_json)
            except Exception as e:
                logger.warning(
                    f"警告：无法从 '{script_file_path}' 构建JSON文件路径: {e}"
                )
                json_filepath = ""

    if json_filepath and os.path.exists(json_filepath):
        try:
            with open(json_filepath, "r", encoding="utf-8") as f:
                json_params = json.load(f)

            _merge_and_validate_params_from_dict(
                loaded_params, json_params, default_params, "JSON文件"
            )
            logger.debug(f"从文件加载参数： {json_filepath}")
        except Exception as e:
            raise type(e)(f"从JSON文件 '{json_filepath}' 加载参数时出错: {e}")

    # 2. 尝试从传入的 override_params 字典加载参数
    if override_params:
        if not isinstance(override_params, dict):
            raise TypeError("传入的 'override_params' 必须是一个字典。")
        try:
            _merge_and_validate_params_from_dict(
                loaded_params, override_params, default_params, "传入字典"
            )
        except (KeyError, ValueError, TypeError) as e:
            raise type(e)(f"从 'override_params' 加载参数时出错: {e}")

    # 3. 尝试从命令行参数加载
    cli_args = sys.argv[1:]  # 跳过脚本名称
    for arg in cli_args:

        key_value_part = arg[1:]  # 去掉开头的 '-'
        full_key, value_str = key_value_part.split("=", 1)
        if full_key == "json":
            continue
        key_parts = full_key.split(".")

        try:
            _update_params_from_cli_recursively(
                loaded_params, key_parts, value_str, default_params
            )
        except (KeyError, TypeError, ValueError) as e:
            raise type(e)(f"处理命令行参数 '-{full_key}={value_str}' 时出错: {e}")

    # 4. 设置主设备
    device = loaded_params.get("device", None)
    if isinstance(device, str) and device.strip() != "":
        _set_main_device(loaded_params, device)
    return loaded_params
