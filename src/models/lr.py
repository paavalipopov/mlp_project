# pylint: disable=invalid-name, missing-function-docstring, unused-argument, too-many-arguments, too-few-public-methods
""" Logistic Regression model module """

import torch
from torch import nn

from omegaconf import OmegaConf, DictConfig


def get_model(cfg: DictConfig, model_cfg: DictConfig):
    return LogisticRegression()


def default_HPs(cfg: DictConfig):
    model_cfg = {
        "input_size": cfg.dataset.data_info.main.data_shape[2],
        "output_size": cfg.dataset.data_info.main.n_classes,
    }
    return OmegaConf.create(model_cfg)


class LogisticRegression(nn.Module):

    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        self.linear = nn.Linear(model_cfg.input_size, model_cfg.output_size)

    def forward(self, x):
        logits = self.linear(x)
        return logits
