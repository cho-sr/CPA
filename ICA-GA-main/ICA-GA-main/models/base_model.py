import os
import torch
import torch.nn as nn
from torchvision import models

from utils import logger


class BaseModel(nn.Module):

    def __init__(self, save_dir="./data/models", save_name=""):
        super(BaseModel, self).__init__()
        self.set_save(save_dir, save_name)

    def set_save(self, save_dir="./data/models", save_name=""):
        self.save_dir = save_dir
        self.save_name = save_name
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if self.save_name:
            self.model_path = os.path.join(self.save_dir, self.save_name) + ".pth"
        else:
            self.model_path = None

    def save_model(self, model_path=""):
        if not model_path:
            model_path = self.model_path
        if model_path:
            try:
                torch.save(self.state_dict(), model_path)
                logger.info(f"Model saved to {model_path}")
            except Exception:
                logger.error("Failed to save model.", exc_info=True, stack_info=True)
        else:
            logger.info("Model name is empty. Cannot save model.")

    def load_model(self, model_path=""):
        if not model_path:
            model_path = self.model_path
        if model_path and os.path.exists(model_path):
            try:
                self.load_state_dict(
                    torch.load(
                        model_path, weights_only=True, map_location=torch.device("cpu")
                    )
                )
                logger.info(f"Model loaded from {model_path}")
            except Exception:
                logger.error(
                    f"Failed to load model {self.__class__.__name__} {model_path}.",
                    exc_info=True,
                    stack_info=True,
                )
        else:
            logger.warning(
                f'{self.__class__.__name__} Model path "{model_path}" does not exist or model name is empty. Cannot load model.'
            )

    def set_requires_grad(self, requires_grad=True):
        for param in self.parameters():
            param.requires_grad = requires_grad
