# --------------------------------------------------
# 图像生成器模型
# --------------------------------------------------

from .mod_gen.model import ModGen
from .fgla.model import FglaGenerator, FglaGenerator18, FglaGenerator50

generator_models = {
    "FglaGenerator18": FglaGenerator18,
    "FglaGenerator50": FglaGenerator50,
    "ModGen": ModGen,
}
