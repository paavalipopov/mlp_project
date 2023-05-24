# pylint: disable=invalid-name, missing-function-docstring, missing-class-docstring, unused-argument, too-few-public-methods, no-member, too-many-arguments, line-too-long, too-many-instance-attributes
""" BrainNetCNN model module from https://github.com/Wayfear/BrainNetworkTransformer"""

import bisect
import math

from torch.nn import functional as F
from torch import nn, optim
import torch
from omegaconf import OmegaConf, DictConfig


def get_model(cfg: DictConfig, model_cfg: DictConfig):
    return BrainNetCNN(model_cfg)


def default_HPs(cfg: DictConfig):
    model_cfg = {
        "node_sz": cfg.dataset.data_info.main.data_shape[1],
        "output_size": cfg.dataset.data_info.main.n_classes,
        "optimizer": {"lr": 1e-4, "weight_decay": 1e-4},
        "scheduler": {
            "mode": "cos",  # ['step', 'poly', 'cos']
            "base_lr": 1e-4,
            "target_lr": 1e-5,
            "decay_factor": 0.1,  # for step mode
            "milestones": [0.3, 0.6, 0.9],
            "poly_power": 2.0,  # for poly mode
            "lr_decay": 0.98,
            "warm_up_from": 0.0,
            "warm_up_steps": 0,
        },
    }
    return OmegaConf.create(model_cfg)


def get_optimizer(cfg: DictConfig, model_cfg: DictConfig, model):
    optimizer = optim.Adam(
        model.parameters(),
        lr=model_cfg.optimizer.lr,
        weight_decay=model_cfg.optimizer.weight_decay,
    )

    return optimizer


def get_scheduler(cfg: DictConfig, model_cfg: DictConfig, optimizer):
    return LRScheduler(cfg, model_cfg, optimizer)


class LRScheduler:
    def __init__(self, cfg: DictConfig, model_cfg: DictConfig, optimizer):
        self.optimizer = optimizer

        self.current_step = 0

        self.scheduler_cfg = model_cfg.scheduler

        self.lr_mode = model_cfg.scheduler.mode
        self.base_lr = model_cfg.scheduler.base_lr
        self.target_lr = model_cfg.scheduler.target_lr

        self.warm_up_from = model_cfg.scheduler.warm_up_from
        self.warm_up_steps = model_cfg.scheduler.warm_up_steps
        self.total_steps = cfg.mode.max_epochs

        self.lr = None

        assert self.lr_mode in ["step", "poly", "cos"]

    def step(self, metric):
        assert 0 <= self.current_step <= self.total_steps
        if self.current_step < self.warm_up_steps:
            current_ratio = self.current_step / self.warm_up_steps
            self.lr = (
                self.warm_up_from + (self.base_lr - self.warm_up_from) * current_ratio
            )
        else:
            current_ratio = (self.current_step - self.warm_up_steps) / (
                self.total_steps - self.warm_up_steps
            )
            if self.lr_mode == "step":
                count = bisect.bisect_left(self.scheduler_cfg.milestones, current_ratio)
                self.lr = self.base_lr * pow(self.scheduler_cfg.decay_factor, count)
            elif self.lr_mode == "poly":
                poly = pow(1 - current_ratio, self.scheduler_cfg.poly_power)
                self.lr = self.target_lr + (self.base_lr - self.target_lr) * poly
            elif self.lr_mode == "cos":
                cosine = math.cos(math.pi * current_ratio)
                self.lr = (
                    self.target_lr + (self.base_lr - self.target_lr) * (1 + cosine) / 2
                )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr
        self.current_step += 1


class E2EBlock(torch.nn.Module):
    """E2Eblock."""

    def __init__(self, in_planes, planes, roi_num, bias=True):
        super().__init__()
        self.d = roi_num
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)


class BrainNetCNN(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        self.in_planes = 1
        self.d = model_cfg.node_sz

        self.e2econv1 = E2EBlock(1, 32, self.d, bias=True)
        self.e2econv2 = E2EBlock(32, 64, self.d, bias=True)
        self.E2N = torch.nn.Conv2d(64, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 256, (self.d, 1))
        self.dense1 = torch.nn.Linear(256, 128)
        self.dense2 = torch.nn.Linear(128, 30)
        self.dense3 = torch.nn.Linear(30, model_cfg.output_size)

    def forward(self, node_feature: torch.tensor):
        node_feature = node_feature.unsqueeze(dim=1)
        out = F.leaky_relu(self.e2econv1(node_feature), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(self.dense1(out), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(self.dense2(out), negative_slope=0.33), p=0.5)
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)

        return out
