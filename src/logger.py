"""Logger factory"""
from omegaconf import open_dict
import wandb


def logger_factory(cfg, model_cfg):
    """Basic logger factory"""
    logger = wandb.init(
        project=cfg.project_name,
        name=cfg.wandb_trial_name,
        save_code=True,
        dir="/tmp/ppopov1/wandb"
    )

    # save tuning process wandb link
    if cfg.mode.name == "tune":
        link = logger.get_url()
        with open_dict(model_cfg):
            model_cfg.link = link

    return logger
