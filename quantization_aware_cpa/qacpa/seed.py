from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def torch_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    try:
        generator = torch.Generator(device=str(device))
    except TypeError:
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator
