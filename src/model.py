# pylint: disable=invalid-name
"""Models for experiments and functions for setting them up"""

from importlib import import_module
from omegaconf import OmegaConf


def model_config_factory(cfg, k=None):
    """Model config factory"""
    if cfg.exp.mode == "tune":
        model_config = get_tune_config(cfg)
    elif cfg.exp.mode == "exp":
        model_config = get_best_config(cfg, k)
    else:
        raise NotImplementedError

    return model_config


def get_tune_config(cfg):
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

    try:
        model_cfg = random_HPs(cfg)
    except TypeError:
        model_cfg = random_HPs()

    print("Tuning model config:")
    print(f"{OmegaConf.to_yaml(model_cfg)}")

    return model_cfg


def get_best_config(cfg, k):
    pass


# TODO: implement
# if conf.glob:
#     pass
#     # with open(f"{HYPERPARAMS_ROOT}/{conf.model}.json", "r", encoding="utf8") as fp:
#     #     model_config = json.load(fp)
#     #     model_config["input_size"] = conf.data_info["data_shape"]["main"][2]
#     #     model_config["output_size"] = conf.data_info["n_classes"]

# else:
#     assert k is not None

#     model_config = {}

#     # find and load the best tuned model
#     exp_dirs = []

#     searched_dir = conf.project_name.split("-")
#     searched_dir = "-".join(searched_dir[2:4])
#     searched_dir = f"tune-{searched_dir}"
#     if conf.prefix != conf.default_prefix:
#         searched_dir = f"{conf.prefix}-{searched_dir}"
#     print(f"Searching trained model in {LOGS_ROOT}/*{searched_dir}")
#     for logdir in os.listdir(LOGS_ROOT):
#         if logdir.endswith(searched_dir):
#             exp_dirs.append(os.path.join(LOGS_ROOT, logdir))

#     # if multiple run files found, choose the latest
#     exp_dir = sorted(exp_dirs)[-1]
#     print(f"Using best model from {exp_dir}")

#     # get model config
#     df = pd.read_csv(
#         f"{exp_dir}/k_{k:02d}/runs.csv", delimiter=",", index_col=False
#     )
#     # pick hyperparams of a model with the highest test_score
#     best_config_path = df.loc[df["score"].idxmax()].to_dict()
#     best_config_path = best_config_path["path_to_config"]
#     with open(best_config_path, "r", encoding="utf8") as fp:
#         model_config = json.load(fp)

# print("Loaded model config:")
# print(model_config)

# return model_config


def model_factory(cfg, model_cfg=None):
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

    try:
        model = get_model(model_cfg)
    except TypeError:
        model = get_model()

    return model
