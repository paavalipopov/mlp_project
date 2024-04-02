# pylint: disable=invalid-name, too-many-branches, line-too-long
"""Models for experiments and functions for setting them up"""

from importlib import import_module
import os

from omegaconf import OmegaConf, DictConfig, open_dict

from src.settings import LOGS_ROOT


def model_config_factory(cfg: DictConfig, k=None):
    """Model config factory"""
    if cfg.mode.name == "tune":
        model_config = get_tune_config(cfg)
    elif cfg.mode.name == "exp":
        model_config = get_best_config(cfg, k)
    else:
        raise NotImplementedError

    return model_config


def get_tune_config(cfg: DictConfig):
    """Returns random HPs defined by the models random_HPs() function"""
    if "tunable" in cfg.model:
        assert cfg.model.tunable, "Model is specified as not tunable, aborting"

    try:
        model_module = import_module(f"src.models.{cfg.model.name}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"No module named '{cfg.model.name}' \
                                  found in 'src.models'. Check if model name \
                                  in config file and its module name are the same"
        ) from e

    try:
        random_HPs = model_module.random_HPs
    except AttributeError as e:
        raise AttributeError(
            f"'src.models.{cfg.model.name}' has no function\
                             'random_HPs'. Is the model not supposed to be\
                             tuned, or the function misnamed/not defined?"
        ) from e

    model_cfg = random_HPs(cfg)

    print("Tuning model config:")
    print(f"{OmegaConf.to_yaml(model_cfg)}")

    return model_cfg


def get_best_config(cfg: DictConfig, k=None):
    """
    1. If cfg.single_HP is True, return the HPs stored in cfg.model_cfg_path. You should avoid using it.
    2. If cfg.model.default_HP is True, return the HPs defined by 'default_HPs(cfg)' function in the model's .py module
    3. If the above is false, returns the optimal HPs of the given model for the given k,
        or, if cfg.dataset.tuning_holdout is True, a single optimal set of HPs
    """

    try:
        model_module = import_module(f"src.models.{cfg.model.name}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"No module named '{cfg.model.name}' \
                                found in 'src.models'. Check if model name \
                                in config file and its module name are the same"
        ) from e

    try:
        default_HPs = model_module.default_HPs
    except AttributeError as e:
        raise AttributeError(
            f"'src.models.{cfg.model.name}' has no function\
                            'default_HPs'. Is the function misnamed/not defined?"
        ) from e

    model_cfg = default_HPs(cfg)

    print("Loaded model config:")
    print(f"{OmegaConf.to_yaml(model_cfg)}")
    return model_cfg


def model_factory(cfg: DictConfig, model_cfg: DictConfig):
    """Models factory"""
    try:
        model_module = import_module(f"src.models.{cfg.model.name}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"No module named '{cfg.model.name}' \
                                  found in 'src.models'. Check if model name \
                                  in config file and its module name are the same"
        ) from e

    try:
        get_model = model_module.get_model
    except AttributeError as e:
        raise AttributeError(
            f"'src.models.{cfg.model.name}' has no function\
                             'get_model'. Is the function misnamed/not defined?"
        ) from e

    model = get_model(cfg, model_cfg)

    return model
