# --------------------------------------------------
# 数据类型： 参数 / 梯度列表字典
# --------------------------------------------------

import torch
import torch.nn as nn
from typing import Optional, Any, List, Dict, Union
from collections import UserList

from utils import slice_to_str


class ParameterListDict(UserList):

    def __init__(
        self,
        parameters: List[torch.Tensor],
        names: List[str],
        model_name: str,
        device: Optional[torch.device] = None,
    ):
        """一个torch模型的参数列表。用法兼容 list 类型，并支持额外属性和方法。
        创建后，不允许调整结构，只能调整内容。

        Args:
            parameters (List[torch.Tensor]): 任意模型 nn.Module.named_parameters() 的value列表或value.grad列表
            names (List[str]): 任意模型 nn.Module.named_parameters() 的key列表
            model_name (str): 模型名称或模型.__class__.__name__
            device (torch.device): 当前 ParameterListDict 的设备。默认自动推断
        """
        # 合法性检验
        len_params = len(parameters)
        len_names = len(names)
        assert len_params > 0, "Parameter list is empty"
        assert len_names > 0, "Parameter names are empty"
        assert (
            len_params == len_names
        ), f"Number of parameters ({len_params}) does not match number of names ({len_names})"
        assert len(names) == len(set(names)), "Duplicate parameter names found"

        # 禁用梯度
        for para in parameters:
            para.requires_grad = False

        super().__init__(parameters)  # self.data
        self.names = list(names)  # 参数名称列表，创建副本防止外部修改
        # 映射 参数名称:下标
        self.names_index = {name: index for index, name in enumerate(names)}
        self.model_name = model_name  # 模型名称

        # 处理设备参数
        if device is not None:
            if type(device) is str:
                device = torch.device(device)
            # 当传入device为 "cuda" 且未指定索引时，自动设置为当前CUDA设备索引
            # 以避免 "cuda" != "cuda:0" 的问题
            if device.type == "cuda" and device.index is None:
                current_cuda_index = torch.cuda.current_device()
                device = torch.device("cuda", index=current_cuda_index)
            self._device = device
        else:
            self._device = parameters[0].device

        # 确保所有层的设备一致性
        # Tips: 如果模型子类中不同层位于不同设备，那么应该重写模型的 get_parameter() 方法来规范设备
        for i, p in enumerate(parameters):
            assert (
                self._device == p.device
            ), f'Parameter {names[i]} is on device "{p.device}", which does not match ParameterListDict {model_name}\'s device "{device}"'

    # ---------------------------------------------
    # 设备操作接口
    # ---------------------------------------------

    @property
    def device(self) -> torch.device:
        """返回当前 ParameterListDict 的设备

        Returns:
            torch.device: 当前设备
        """
        return self._device

    def to(self, device: Union[str, torch.device]) -> "ParameterListDict":
        """转移所有参数到指定设备

        Args:
            device (torch.device): 希望转移到的设备

        Returns:
            ParameterListDict: 已转移设备的参数列表
        """
        if type(device) is str:
            device = torch.device(device)
        if self._device == device:
            return self
        new_data = [p.to(device) for p in self]
        return ParameterListDict(new_data, self.names, self.model_name, device)

    def clone(self) -> "ParameterListDict":
        """返回深拷贝所有参数

        Returns:
            ParameterListDict: 当前参数的副本
        """
        return ParameterListDict(
            [p.clone() for p in self], self.names.copy(), self.model_name, self.device
        )

    # ---------------------------------------------
    # 数值计算接口
    # Args:
    #     other: 要计算的值。允许的类型:
    #     - ParameterListDict : 要求参数结构与本实例一致
    #     - int,float : 标量数值
    #     - torch.Tensor : 必须是零维的标量，如 Tensor(2.0)
    # ---------------------------------------------

    def normalize_other(
        self,
        other: Union["ParameterListDict", int, float, torch.Tensor],
    ) -> Union["ParameterListDict", int, float, torch.Tensor]:
        """规范化传入值，确保 other 的结构和设备可以与当前实例进行计算

        Args:
            other: 要检查的参数。允许的类型:
            - ParameterListDict : 要求参数结构与本实例一致
            - int,float : 标量数值
            - torch.Tensor : 必须是零维的标量，如 Tensor(1.0)

        Returns:
            [ParameterListDict,int,float,torch.Tensor]: 规范化后的 other

        Raises:
            ValueError,TypeError: other 不合法时抛出
        """
        # ParameterListDict
        if isinstance(other, ParameterListDict):
            if self.names != other.names:
                raise ValueError(
                    f"Parameter names mismatch: {self.names} | {other.names}"
                )
            return other.to(self.device)

        # Tensor 标量
        elif isinstance(other, torch.Tensor):
            if other.dim() != 0:
                raise ValueError(
                    "Tensor must be a scalar (0-dimensional) to perform operation."
                )
            return other.to(self.device)
        # 数字类型
        elif isinstance(other, Union[int, float]):
            return other

        # 处理非法类型
        raise TypeError(
            f"Unsupported type {type(other)} for operation. Expected: ParameterListDict, int, float, or scalar tensor."
        )

    # self + other
    def __add__(
        self, other: Union["ParameterListDict", int, float, torch.Tensor]
    ) -> "ParameterListDict":
        """加法：计算当前 ParameterListDict 与 other 的和，返回新实例

        Example:
            ```
            new_params = params_1 + params_2
            new_params = params_1 + 1.0
            new_params = params_1 + Tensor(2.0)
            ```
        """
        other = self.normalize_other(other)
        if isinstance(other, ParameterListDict):
            new_data = [p1 + p2 for p1, p2 in zip(self, other)]
        else:
            new_data = [p + other for p in self]
        return ParameterListDict(new_data, self.names, self.model_name, self.device)

    # other + self
    def __radd__(self, other: Union[int, float, torch.Tensor]) -> "ParameterListDict":
        """右侧加法：计算 other 标量与当前 ParameterListDict 的和，返回新实例

        Example:
            ```
            new_params = 1.0 + params_1
            new_params = Tensor(2.0) + params_1
            ```
        """
        return self.__add__(other)

    # self += other
    def __iadd__(
        self, other: Union["ParameterListDict", int, float, torch.Tensor]
    ) -> "ParameterListDict":
        """原地加法：当前 ParameterListDict 实例的值原地加上 other

        Example:
            ```
            params_1 += params_2
            params_1 += 1.0
            params_1 += Tensor(2.0)
            ```
        """
        other = self.normalize_other(other)
        if isinstance(other, ParameterListDict):
            for i in range(len(self)):
                self.data[i] += other.data[i]
        else:
            for i in range(len(self)):
                self.data[i] += other
        return self

    # self - other
    def __sub__(self, other: "ParameterListDict") -> "ParameterListDict":
        """减法：计算当前 ParameterListDict 与 other 的差，返回新实例

        Example:
            ```
            new_params = params_1 - params_2
            new_params = params_1 - 1.0
            new_params = params_1 - Tensor(2.0)
            ```
        """
        other = self.normalize_other(other)
        if isinstance(other, ParameterListDict):
            new_data = [p1 - p2 for p1, p2 in zip(self, other)]
        else:
            new_data = [p - other for p in self]
        return ParameterListDict(new_data, self.names, self.model_name, self.device)

    # other - self
    def __rsub__(self, other: Union[int, float, torch.Tensor]) -> "ParameterListDict":
        """右侧减法：计算 other 标量与当前 ParameterListDict 的差，返回新实例

        Example:
            ```
            new_params = 1.0 - params_1
            new_params = Tensor(2.0) - params_1
            ```
        """
        other = self.normalize_other(other)
        new_data = [other - p for p in self]
        return ParameterListDict(new_data, self.names, self.model_name, self.device)

    # self -= other
    def __isub__(self, other: "ParameterListDict"):
        """原地减法：当前 ParameterListDict 实例的值原地减去 other

        Example:
            ```
            params_1 -= params_2
            params_1 -= 1.0
            params_1 -= Tensor(2.0)
            ```
        """
        other = self.normalize_other(other)
        if isinstance(other, ParameterListDict):
            for i in range(len(self)):
                self.data[i] -= other.data[i]
        else:
            for i in range(len(self)):
                self.data[i] -= other
        return self

    # - self
    def __neg__(self) -> "ParameterListDict":
        """取负：返回所有参数取负后的新实例

        Example:
            ```
            new_params = - params_1
            ```
        """
        new_data = [-p for p in self]
        return ParameterListDict(new_data, self.names, self.model_name, self.device)

    # self * other
    def __mul__(
        self, other: Union["ParameterListDict", int, float, torch.Tensor]
    ) -> "ParameterListDict":
        """乘法：计算当前 ParameterListDict 与 other 的乘积，返回新实例

        Example:
            ```
            new_params = params_1 * params_2
            new_params = params_1 * 1.0
            new_params = params_1 * Tensor(2.0)
            ```
        """
        other = self.normalize_other(other)
        if isinstance(other, ParameterListDict):
            new_data = [p1 * p2 for p1, p2 in zip(self, other)]
        else:
            new_data = [p * other for p in self]
        return ParameterListDict(new_data, self.names, self.model_name, self.device)

    # other * self
    def __rmul__(self, other: Union[int, float, torch.Tensor]) -> "ParameterListDict":
        """右侧乘法：计算 other 标量与当前 ParameterListDict 的乘积，返回新实例

        Example:
            ```
            new_params = 1.0 * params_1
            new_params = Tensor(2.0) * params_1
            ```
        """
        return self.__mul__(other)

    # self *= other
    def __imul__(
        self, other: Union["ParameterListDict", int, float, torch.Tensor]
    ) -> "ParameterListDict":
        """原地乘法：当前 ParameterListDict 实例的值原地乘上 other

        Example:
            ```
            params_1 *= params_2
            params_1 *= 1.0
            params_1 *= Tensor(2.0)
            ```
        """
        other = self.normalize_other(other)
        if isinstance(other, ParameterListDict):
            for i in range(len(self)):
                self.data[i] *= other.data[i]
        else:
            for i in range(len(self)):
                self.data[i] *= other
        return self

    # self / other
    def __truediv__(
        self, other: Union["ParameterListDict", int, float, torch.Tensor]
    ) -> "ParameterListDict":
        """除法：计算当前 ParameterListDict 与 other 的商，返回新实例

        Example:
            ```
            new_params = params_1 / params_2
            new_params = params_1 / 1.0
            new_params = params_1 / Tensor(2.0)
            ```
        """
        other = self.normalize_other(other)
        if isinstance(other, ParameterListDict):
            new_data = [p1 / p2 for p1, p2 in zip(self, other)]
        else:
            new_data = [p / other for p in self]
        return ParameterListDict(new_data, self.names, self.model_name, self.device)

    # other / self
    def __rtruediv__(
        self, other: Union[int, float, torch.Tensor]
    ) -> "ParameterListDict":
        """右侧除法：计算 other 标量与当前 ParameterListDict 的商，返回新实例

        Example:
            ```
            new_params = 1.0 / params_1
            new_params = Tensor(2.0) / params_1
            ```
        """
        other = self.normalize_other(other)
        new_data = [other / p for p in self]
        return ParameterListDict(new_data, self.names, self.model_name, self.device)

    # self /= other
    def __itruediv__(
        self, other: Union["ParameterListDict", int, float, torch.Tensor]
    ) -> "ParameterListDict":
        """原地除法：当前 ParameterListDict 实例的值原地除去 other

        Example:
            ```
            params_1 /= params_2
            params_1 /= 1.0
            params_1 /= Tensor(2.0)
            ```
        """
        other = self.normalize_other(other)
        if isinstance(other, ParameterListDict):
            for i in range(len(self)):
                self.data[i] /= other.data[i]
        else:
            for i in range(len(self)):
                self.data[i] /= other
        return self

    def norm(self) -> torch.Tensor:
        """计算整组参数的模长（欧几里得范数）。

        sqrt(Σ_i ||param_i||^2)

        Example:
            ```
            model_params = ParameterListDict(...)
            total_norm = model_params.norm()
            print(f"模型总参数 模长: {total_norm.item():.2f}")
            ```

        Returns:
            torch.Tensor: 标量张量（0维），表示整组参数的模长
        """
        # 初始化平方和（确保与参数相同设备和类型）
        total_sq = torch.tensor(0.0, device=self.device, dtype=self[0].dtype)
        # 逐参数累加平方和（避免一次性展平全部参数）
        for param in self.data:
            total_sq = total_sq + torch.sum(torch.square(param))
        return torch.sqrt(total_sq)

    # ---------------------------------------------
    # 序列化接口
    # ---------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """序列化接口（兼容 torch 的保存机制）

        Returns:
            Dict:
            ```
            {
                "data": List[移动到CPU的参数],
                "names": List[参数名称],
                "model_name": "模型名称",
                "device": torch.device(原本所在的设备),
            }
            ```
        """
        return {
            "data": [p.cpu() for p in self],
            "names": self.names,
            "model_name": self.model_name,
            "device": self.device,
        }

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any]):
        """反序列化接口

        Args:
            state_dict (Dict):
            ```
            {
                "data": List[移动到CPU的参数],
                "names": List[参数名称],
                "model_name": "模型名称",
                "device": torch.device(原本所在的设备),
            }
            ```
        """
        return cls(
            [p.to(state_dict["device"]) for p in state_dict["data"]],
            state_dict["names"],
            state_dict["model_name"],
            state_dict["device"],
        )

    # ---------------------------------------------
    # 访问接口，支持列表式/字典式风格，用下标/name来访问
    # ---------------------------------------------

    def __getitem__(
        self, key: Union[int, str, slice]
    ) -> Union[torch.Tensor, "ParameterListDict"]:
        """获取参数值。支持多种访问方式：

        Example:
            ```
            # 整数下标访问，返回单个 Tensor
            param_1 = param_list[1]

            # 切片访问，返回新的 ParameterListDict 实例
            param_1_3 = param_list[1:3]
            print(param_1 is param_1_3[0])  # 比较地址用is，得到True
            print(param_1_3.model_name)  # 如 "原模型名[1:3]"

            # 名称访问，返回单个 Tensor
            conv_weight = param_list["conv1.weight"]
            ```
        """
        if isinstance(key, str):
            # 名称访问，返回单个 Tensor
            return self.data[self.names_index[key]]
        elif isinstance(key, slice):
            # 切片访问，返回新的 ParameterListDict 实例
            sliced_parameters = self.data[key]
            sliced_names = self.names[key]
            new_model_name = f"{self.model_name}{slice_to_str(key)}"
            return ParameterListDict(
                parameters=sliced_parameters,
                names=sliced_names,
                model_name=new_model_name,
                device=self.device,
            )
        else:
            # 整数下标访问，按父类设计，返回单个 Tensor
            return super().__getitem__(key)

    def __contains__(self, key: Union[str, torch.Tensor]) -> bool:
        """检查是否包含指定参数（支持名称或张量地址）

        Example:
            ```
            # 名称检查
            has_conv = "conv1.bias" in param_list

            # 张量检查
            param_list = ParameterListDict(parameters=model.parameters(), ...)
            tensor = model.parameters()[1]
            exists = tensor in param_list  # 按对象地址判断
            ```
        """
        # 字符串：比较名称
        if isinstance(key, str):
            return key in self.names_index
        # 张量：比较对象地址（is 运算符）
        else:
            return any(p is key for p in self.data)

    def get(self, name: str, default: Optional[Any] = None) -> torch.Tensor:
        """安全获取参数（类似字典的get方法）

        Example:
            ```
            # 获取存在的参数
            embedding = param_list.get("embedding.weight")

            # 获取不存在的参数返回默认值
            missing_param = param_list.get("nonexist.layer", torch.zeros(10))
            ```
        """
        try:
            return self[name]
        except KeyError:
            return default

    def keys(self):
        """返回所有参数名称的迭代器。（日常使用 names 即可）

        Example:
            ```
            # 获取所有参数名称列表
            all_names = list(param_list.keys())

            # 等价于：
            all_names = param_list.names
            ```
        """
        return iter(self.names)

    def values(self):
        """返回所有参数的迭代器

        Example:
            ```
            # 遍历所有参数张量
            for param in param_list.values():
                print(param.shape)
            ```
        """
        return iter(self.data)

    def items(self):
        """返回 (名称, 参数) 对的迭代器

        Example:
            ```
            # 同时遍历名称和参数
            for name, param in param_list.items():
                print(f"{name}: {param.dtype}")
            ```
        """
        return zip(self.names, self.data)

    def index(self, key: Union[str, torch.Tensor]) -> int:
        """覆盖原index方法，支持按名称或张量地址查找下标

        Example:
            ```
            # 通过名称获取下标
            idx = param_list.index("fc.weight")

            # 通过张量对象获取下标
            param_list = ParameterListDict(parameters=model.parameters(), ...)
            tensor = model.parameters()[1]
            tensor_idx = param_list.index(tensor)
            ```
        """
        # 字符串：名称查找
        if isinstance(key, str):
            return self.names_index[key]
        # 张量：对象地址查找
        else:
            for i, para in enumerate(self.data):
                if para is key:
                    return i
            raise ValueError(f"{key} not in {self.model_name}")

    def get_zeros_like(self) -> "ParameterListDict":
        """创建与当前实例结构相同的零值参数列表

        Returns:
            ParameterListDict: 新实例，所有参数值为零，
            保留原参数的形状、设备、名称和模型名称

        Example:
            ```
                plist = ParameterListDict(...)
                zero_plist = plist.get_zeros_like()
                print(zero_plist[0])  # 形状与原参数一致的全零张量
            ```
        """
        # 生成零值参数列表
        zero_parameters = [torch.zeros_like(p) for p in self]

        # 构建新实例
        return ParameterListDict(
            parameters=zero_parameters,
            names=self.names,
            model_name=self.model_name,
            device=self.device,
        )

    # ---------------------------------------------
    # 覆盖暂时未使用的方法或所有可能修改列表的方法，使其不可变
    # TODO: 未来如果需要调整 ParameterListDict 的形状，
    # 那么可以重写以下方法，返回新的 ParameterListDict 副本。
    # ---------------------------------------------

    def __delitem__(self, index):
        raise AttributeError("ParameterListDict is immutable; cannot delete items")

    def append(self, item):
        raise AttributeError("ParameterListDict is immutable; cannot append items")

    def insert(self, index, item):
        raise AttributeError("ParameterListDict is immutable; cannot insert items")

    def extend(self, items):
        raise AttributeError("ParameterListDict is immutable; cannot extend list")

    def pop(self, index=-1):
        raise AttributeError("ParameterListDict is immutable; cannot pop items")

    def remove(self, value):
        raise AttributeError("ParameterListDict is immutable; cannot remove items")

    def clear(self):
        raise AttributeError("ParameterListDict is immutable; cannot clear list")
