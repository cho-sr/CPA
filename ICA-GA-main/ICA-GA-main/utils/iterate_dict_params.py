# --------------------------------------------------
# 迭代字典配置参数
# --------------------------------------------------

import copy
import itertools


def iterate_dict_params(input_dict):
    """
    根据包含列表的字典构造迭代器，生成所有可能的配置组合。

    Args:
        input_dict (dict): 输入的字典，可能包含列表作为需要迭代的参数。

    Yields:
        dict: 一个新的字典，代表一种参数组合。
    """
    paths_and_lists = []

    def find_list_paths(current_item, current_path):
        if isinstance(current_item, dict):
            for key, value in current_item.items():
                new_path = current_path + (key,)
                if isinstance(value, list):
                    # 只迭代非空列表
                    if value:
                        paths_and_lists.append((new_path, value))
                elif isinstance(value, dict):
                    find_list_paths(value, new_path)

    find_list_paths(input_dict, ())

    if not paths_and_lists:
        yield copy.deepcopy(input_dict)
        return

    # 反转列表，使得原始字典中“后面”或“更深”的列表参数作为迭代的外层循环
    paths_and_lists.reverse()

    iter_paths = [item[0] for item in paths_and_lists]
    iter_lists = [item[1] for item in paths_and_lists]

    for value_combination in itertools.product(*iter_lists):
        new_config = copy.deepcopy(input_dict)

        for i, path_keys in enumerate(iter_paths):
            current_level_dict = new_config
            for k_idx, key_part in enumerate(path_keys):
                if k_idx == len(path_keys) - 1:  # 到达路径的最后一个键
                    current_level_dict[key_part] = value_combination[i]
                else:
                    current_level_dict = current_level_dict[key_part]
        yield new_config
