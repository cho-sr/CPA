# --------------------------------------------------
# 联邦学习模型
# --------------------------------------------------

from .base_cnn_model import BaseCnnModel
from .resnet import ResNet18, ResNet50
from .vgg import Vgg16, Vgg19

fl_models = {
    "ResNet18": ResNet18,
    "ResNet50": ResNet50,
    "Vgg16": Vgg16,
    "Vgg19": Vgg19,
}
