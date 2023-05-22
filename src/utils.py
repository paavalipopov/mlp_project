# pylint: disable=invalid-name
"""Auxilary functions"""

from omegaconf import open_dict

from src.settings import UTCNOW, LOGS_ROOT


def set_project_name(cfg):
    """set project name and project dir based on config"""

    default_prefix = f"{UTCNOW}"

    if cfg.prefix is None:
        prefix = default_prefix
        used_default_prefix = True
    else:
        if len(cfg.prefix) == 0:
            prefix = default_prefix
            used_default_prefix = True
        else:
            # '-'s are reserved for project name parsing
            prefix = cfg.prefix.replace("-", "_")
            used_default_prefix = False

    project_name = f"{prefix}-{cfg.exp.mode}-{cfg.model.name}-{cfg.dataset.name}"

    if "multiclass" in cfg.dataset and cfg.dataset.multiclass:
        project_name += "-multiclass"
    if "zscore" in cfg.dataset and cfg.dataset.zscore:
        project_name += "-zscore"
    if "single_HP" in cfg.exp and cfg.exp.single_HP:
        project_name += "-single_HP"

    project_dir = str(LOGS_ROOT.joinpath(project_name))

    with open_dict(cfg):
        cfg.project_name = project_name
        cfg.project_dir = project_dir
        cfg.used_default_prefix = used_default_prefix


def set_run_name(cfg, outer_k=None, trial=None, inner_k=None):
    """set wandb run name and run directories"""
    if cfg.exp.mode == "tune":
        if cfg.exp.single_HP:
            wandb_trial_name = f"trial_{trial:04d}-k_{inner_k:02d}"
            k_dir = f"{cfg.project_dir}"
        else:
            wandb_trial_name = f"k_{outer_k:02d}-trial_{trial:04d}-k_{inner_k:02d}"
            k_dir = f"{cfg.project_dir}/k_{outer_k:02d}"

        trial_dir = f"{k_dir}/trial_{trial:04d}"
        run_dir = f"{trial_dir}/k_{inner_k:02d}"
        with open_dict(cfg):
            cfg.wandb_trial_name = wandb_trial_name
            cfg.k_dir = k_dir
            cfg.trial_dir = trial_dir
            cfg.run_dir = run_dir

    elif cfg.exp.mode == "exp":
        wandb_trial_name = f"k_{outer_k:02d}-trial_{trial:04d}"
        k_dir = f"{cfg.project_dir}/k_{outer_k:02d}"
        run_dir = f"{k_dir}/trial_{trial:04d}"
        with open_dict(cfg):
            cfg.wandb_trial_name = wandb_trial_name
            cfg.k_dir = k_dir
            cfg.run_dir = run_dir
