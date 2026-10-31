# --------------------------------------------------
# 特征分离相关模块
# --------------------------------------------------

from .base_separator import BaseSeparator
from .fgla_fsg import FSG
from .sklearn_fast_ica import SklearnFastICA
from .fast_ica import FastICA
from .gradient_descent_ica import GradientDescentICA
from .combination_ica import CombinationICA

feature_separators = {
    "FSG": FSG,
    "FastICA": FastICA,
    "GD-ICA": GradientDescentICA,
    "CombinationICA": CombinationICA,
}
