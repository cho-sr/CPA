# --------------------------------------------------
# 用于循环的日志打印器
# --------------------------------------------------

from typing import Optional, Callable

from .log import logger
from .utils import dict_to_str


class LoopLog:
    def __init__(
        self,
        name: str,
        len: int,
        total: int = 10,
    ):
        """
        Args:
            name (str): 名称
            len (int): 循环长度
            total (int): 记录次数
        """
        self.name = name
        if total <= 0:
            total = 1
        self.len = len
        self.flag_ep = max(round(len / total), 1)
        self.ep = 0

    def log(
        self,
        info_dict: Optional[dict] = None,
        info_func: Optional[Callable] = None,
        ep: int = -1,
    ):
        """
        Args:
            ep (int): 当前循环次数
            info_dict (dict): 可选，信息字典
            info_func (Callable): 可选，返回信息字典的函数
        """
        if ep < 0:
            ep = self.ep
            self.ep += 1
        else:
            self.ep = ep
        if ep == 0 or ep % self.flag_ep == self.flag_ep - 1 or ep == self.len - 1:
            if not info_dict:
                if info_func:
                    info_dict: dict = info_func()
                else:
                    raise ValueError("info_dict and info_func are both None")

            log_str = f"{self.name}: {ep:>4}, " + dict_to_str(info_dict)
            logger.info(log_str)
