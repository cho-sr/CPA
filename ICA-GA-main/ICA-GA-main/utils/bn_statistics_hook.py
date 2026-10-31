# --------------------------------------------------
# BN 层钩子，STG依赖
# --------------------------------------------------

import torch.nn as nn


class BNStatisticsHook:
    def __init__(self, model):
        self.mean_var_list = [[], []]
        self.hook_list = []
        self.model = model

    def hook_fn(self, _, input_data):
        mean = input_data[0].mean(dim=[0, 2, 3]).detach()
        var = input_data[0].var(dim=[0, 2, 3]).detach()
        self.mean_var_list[0].append(mean)
        self.mean_var_list[1].append(var)

    def clear(self):
        self.mean_var_list = [[], []]

    def register(self):
        self.unregister()
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                hook = module.register_forward_pre_hook(self.hook_fn)
                self.hook_list.append(hook)

    def unregister(self):
        for hook in self.hook_list:
            hook.remove()
        self.hook_list = []
