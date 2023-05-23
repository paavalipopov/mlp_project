# pylint: disable=invalid-name
"""Models for experiments and functions for setting them up"""

from importlib import import_module
import os

from omegaconf import OmegaConf, open_dict

from src.settings import LOGS_ROOT


def model_config_factory(cfg, k=None):
    """Model config factory"""
    if cfg.mode.name == "tune":
        model_config = get_tune_config(cfg)
    elif cfg.mode.name == "exp":
        model_config = get_best_config(cfg, k)
    else:
        raise NotImplementedError

    return model_config


def get_tune_config(cfg):
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


def get_best_config(cfg, k=None):
    """
    1. If cfg.single_HP is True, return the HPs stored in cfg.model_cfg_path. You should avoid using it.
    2. If cfg.model.default_HP is True, return the HPs defined by 'default_HPs(cfg)' function in the model's .py module
    3. If the above is false, returns the optimal HPs of the given model for the given k,
        or, if cfg.dataset.tuning_holdout is True, a single optimal set of HPs
    """
    if "single_HPs" in cfg and cfg.single_HPs:
        # 1. try to load config from cfg.model_cfg_path.
        # Note: dataset shape information is loaded as 'data_info' key,
        # just in case
        assert cfg.model_cfg_path.endswith(
            ".yaml", ".json"
        ), f"'{cfg.model_cfg_path}' \
            is not json or yaml file, aborting"
        try:
            with open(cfg.model_cfg_path, "r", encoding="utf8") as f:
                model_cfg = OmegaConf.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"'{cfg.model_cfg_path}' config is not found"
            ) from e

        with open_dict(model_cfg):
            model_cfg.data_info = cfg.dataset.data_info

    elif cfg.model.default_HP:
        # 2. try to get the HPs defined by 'default_HPs(cfg)' function in the model's .py module
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

    else:
        # 3. load the optimal HPs from logs
        if "tuning_holdout" in cfg.dataset and cfg.dataset.tuning_holdout:
            assert k is not None

        searched_dir = cfg.project_name.split("-")
        if cfg.used_default_prefix:
            searched_dir = "tune-" + "-".join(searched_dir[2:])
        else:
            searched_dir = f"{cfg.prefix}-tune-" + "-".join(searched_dir[2:])

        print(f"Searching trained model in '{LOGS_ROOT}/*{searched_dir}'")
        dirs = []
        for logdir in os.listdir(LOGS_ROOT):
            if logdir.endswith(searched_dir):
                dirs.append(os.path.join(LOGS_ROOT, logdir))

        assert (
            len(dirs) != 0
        ), "No matching directory found. \
            Did you set wrong project prefix, or did not run HP tuning? \
                If you intended to load default HPs, you should \
                      set 'default_HP' to True in cfg.model"
        # if multiple run files found, choose the latest
        found_dir = sorted(dirs)[-1]
        print(f"Using best model from {found_dir}/k_{k:02d}/")

        # get model config
        if "tuning_holdout" in cfg.dataset and cfg.dataset.tuning_holdout:
            best_config_path = f"{found_dir}/best_config.yaml"
        else:
            best_config_path = f"{found_dir}/k_{k:02d}/best_config.yaml"
        with open(best_config_path, "r", encoding="utf8") as f:
            model_cfg = OmegaConf.load(f)

    print("Loaded model config:")
    print(model_cfg)
    return model_cfg


def model_factory(cfg, model_cfg):
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

    model = get_model(model_cfg)

    return model
