# --------------------------------------------------
# csv 记录器
# --------------------------------------------------

import os
import csv
from typing import Optional, Dict, Any, List, Iterator

from .log import logger
from .utils import dict_to_str


class Recorder:
    def __init__(self, save_path: Optional[str] = None):
        self.data: List[Dict[str, Any]] = []  # 存储所有记录
        self.save_path = save_path
        self.csv_initialized = False
        self.headers = []  # 按键首次出现的顺序存储所有唯一的键
        self._header_set: set[str] = set()  # 用于快速查找键是否存在

        if self.save_path:
            directory = os.path.dirname(self.save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
        self.extra_info = None

    def add(self, record: dict, is_log=True):
        """
        添加一条记录。
        如果提供了 save_path，则记录将被写入（或追加到）CSV文件。
        如果记录包含新的键，CSV文件将被重写以包含新的列。
        """
        if self.extra_info is not None:
            record.update(self.extra_info)
            self.extra_info = None
        # 浮点数转为字符串
        for key, val in record.items():
            if isinstance(val, float):
                record[key] = f"{val:.10f}".rstrip("0").rstrip(".")
        self.data.append(record)

        if is_log:
            logger.info(dict_to_str(record))

        if not self.save_path:
            return

        current_record_keys: List[str] = list(record.keys())

        # 检查当前记录是否引入了新的表头
        # is_initial_header_setup 为 True 表示 self.headers 列表在本次调用 add 之前为空
        is_initial_header_setup: bool = not self.headers

        newly_added_keys_this_call: List[str] = []
        has_new_keys: bool = False

        if is_initial_header_setup:
            # 这是定义表头的第一个记录（或者之前的记录都是空的）
            # 所有来自此记录的键都将构成初始表头
            for key in current_record_keys:
                if key not in self._header_set:  # 对于初始设置，_header_set 也应为空
                    newly_added_keys_this_call.append(key)
                    self._header_set.add(key)
            if newly_added_keys_this_call:  # 仅当记录非空时
                self.headers.extend(newly_added_keys_this_call)
                has_new_keys = True  # 标记表头已更改/初始化
        else:
            # 表头已存在，检查此记录是否包含新键
            for key in current_record_keys:
                if key not in self._header_set:
                    newly_added_keys_this_call.append(key)
                    self._header_set.add(key)  # 立即添加到集合中
                    has_new_keys = True
            if has_new_keys:
                # 将新键按顺序追加到表头列表
                self.headers.extend(newly_added_keys_this_call)

        # 需要重新写入表头
        if is_initial_header_setup or has_new_keys:
            try:
                with open(
                    self.save_path, mode="w", newline="", encoding="utf-8"
                ) as file:
                    writer = csv.DictWriter(file, fieldnames=self.headers)
                    writer.writeheader()
                    writer.writerows(self.data)  # 写入内存中的所有数据
            except IOError as e:
                logger.error(f"写入CSV文件失败 (模式 'w'): {self.save_path}, {e}")
            except ValueError as e:
                # 可能由 DictWriter 引发 (例如，不正确的 fieldnames)
                logger.error(
                    f"CSV数据/表头错误，写入期间 (模式 'w'): {self.save_path}, {e}"
                )
        # 表头已建立且当前记录未引入新键，以追加模式 ("a") 打开文件并仅写入当前记录
        else:
            try:
                with open(
                    self.save_path, mode="a", newline="", encoding="utf-8"
                ) as file:
                    writer = csv.DictWriter(file, fieldnames=self.headers)
                    writer.writerow(record)
            except IOError as e:
                logger.error(f"追加到CSV文件失败 (模式 'a'): {self.save_path}, {e}")
            except ValueError as e:
                logger.error(f"CSV数据错误，追加期间 (模式 'a'): {self.save_path}, {e}")

    def add_extra(self, record: dict):
        """增加一个额外信息，将会添加到下一次add的结尾。"""
        if not self.extra_info:
            self.extra_info = record
        else:
            self.extra_info.update(record)

    def __getitem__(self, index):
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)
