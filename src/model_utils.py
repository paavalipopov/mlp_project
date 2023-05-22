# pylint: disable=invalid-name
"""Models for experiments and functions for setting them up"""

from importlib import import_module
from torch import nn, optim


def criterion_factory(cfg, model_cfg):
    """Criterion factory"""
    if "custom_criterion" not in cfg.model or not cfg.model.custom_criterion:
        criterion = CEloss(model_cfg)
    else:
        try:
            model_module = import_module(f"src.models.{cfg.model.name}")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"No module named '{cfg.model.name}' \
                                    found in 'src.models'. Check if model name \
                                    in config file and its module name are the same"
            ) from e

        try:
            get_criterion = model_module.get_criterion
        except AttributeError as e:
            raise AttributeError(
                f"'src.models.{cfg.model.name}' has no function\
                                'get_criterion'. Is the function misnamed/not defined?"
            ) from e

        try:
            criterion = get_criterion(model_cfg)
        except TypeError:
            criterion = get_criterion()

    return criterion


class CEloss:
    """Basic Cross-entropy loss"""

    def __init__(self, model_config):
        self.ce_loss = nn.CrossEntropyLoss()

    def __call__(self, logits, target, model, device):
        ce_loss = self.ce_loss(logits, target)

        return ce_loss


# # TODO: move it to DICE model CRITERION implementation
# class DICEregCEloss:
#     """Cross-entropy loss with model regularization"""

#     def __init__(self, model_cfg):
#         self.ce_loss = nn.CrossEntropyLoss()

#         self.reg_param = model_cfg["reg_param"]

#     def __call__(self, logits, target, model, device):
#         ce_loss = self.ce_loss(logits, target)

#         reg_loss = torch.zeros(1).to(device)

#         for name, param in model.gta_embed.named_parameters():
#             if "bias" not in name:
#                 reg_loss += self.reg_param * torch.norm(param, p=1)

#         for name, param in model.gta_attend.named_parameters():
#             if "bias" not in name:
#                 reg_loss += self.reg_param * torch.norm(param, p=1)

#         loss = ce_loss + reg_loss
#         return loss
# # TODO: move it to DICE model SCHEDULER implementation
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer,
#             patience=model_config["scheduler"]["patience"],
#             factor=model_config["scheduler"]["factor"],
#             cooldown=0,
#         )


def optimizer_factory(cfg, model, model_cfg):
    """Optimizer factory"""
    if "custom_optimizer" not in cfg.model or not cfg.model.custom_optimizer:
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(model_cfg["lr"]),
        )
    else:
        try:
            model_module = import_module(f"src.models.{cfg.model.name}")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"No module named '{cfg.model.name}' \
                                    found in 'src.models'. Check if model name \
                                    in config file and its module name are the same"
            ) from e

        try:
            get_optimizer = model_module.get_optimizer
        except AttributeError as e:
            raise AttributeError(
                f"'src.models.{cfg.model.name}' has no function\
                                'get_optimizer'. Is the function misnamed/not defined?"
            ) from e

        try:
            optimizer = get_optimizer(model_cfg)
        except TypeError:
            optimizer = get_optimizer()

    return optimizer


def scheduler_factory(cfg, optimizer, model_cfg):
    """Scheduler factory"""
    if "custom_scheduler" not in cfg.model or not cfg.model.custom_scheduler:
        scheduler = DummyScheduler()
    else:
        try:
            model_module = import_module(f"src.models.{cfg.model.name}")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"No module named '{cfg.model.name}' \
                                    found in 'src.models'. Check if model name \
                                    in config file and its module name are the same"
            ) from e

        try:
            get_scheduler = model_module.get_scheduler
        except AttributeError as e:
            raise AttributeError(
                f"'src.models.{cfg.model.name}' has no function\
                                'get_scheduler'. Is the function misnamed/not defined?"
            ) from e

        try:
            scheduler = get_scheduler(optimizer, model_cfg)
        except TypeError:
            scheduler = get_scheduler(optimizer)

    return scheduler


class DummyScheduler:
    """Dummy scheduler that does nothing"""

    def __init__(self):
        pass

    def step(self, metric):
        pass
