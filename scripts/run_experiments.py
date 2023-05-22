# pylint: disable=too-many-statements, too-many-locals, invalid-name, unbalanced-tuple-unpacking
"""Script for running experiments: tuning and testing hypertuned models"""
import sys
import os
import json

from omegaconf import DictConfig, OmegaConf
import hydra

import pandas as pd
import numpy as np

from src.utils import set_project_name, set_run_name
from src.data import data_factory, data_postfactory
from src.dataloader import dataloader_factory, cross_validation_split
from src.model import model_config_factory, model_factory
from src.model_utils import criterion_factory, optimizer_factory, scheduler_factory
from src.logger import logger_factory
from src.trainer import trainer_factory


@hydra.main(version_base=None, config_path="../src/conf", config_name="exp_config")
def start(cfg):
    """Main script for starting experiments"""

    # set wandb environment
    os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
    os.environ["WANDB_MODE"] = "offline" if cfg.wandb_offline else "online"

    # set project name and directory
    set_project_name(cfg)
    os.makedirs(cfg.project_dir, exist_ok=True)
    with open(f"{cfg.project_dir}/general_config.yaml", "w", encoding="utf8") as f:
        OmegaConf.save(cfg, f)

    # load dataset, compute FNCs if model requires them.
    # if exp.single_HP and exp.tuning_holdout set to True,
    # returns tuning/experimental data depending on exp.mode
    original_data = data_factory(cfg)

    print(OmegaConf.to_yaml(cfg))

    if cfg.exp.mode == "tune":
        if cfg.exp.single_HP:
            # use the whole original_data for tuning.
            # the obtained HPs are better to be used on other datasets
            # (unless exp.tuning_holdout is True),
            # otherwise there is a testing-on-train-data danger
            tune(cfg=cfg, original_data=original_data)

        else:
            # tune each original_data CV fold independently
            for k in range(0, cfg.exp.n_splits):
                tune_fold_data = original_data
                tune_fold_data["main"], _ = cross_validation_split(
                    tune_fold_data["main"], cfg.exp.n_splits, k
                )
                tune(cfg=cfg, original_data=tune_fold_data, outer_k=k)

    # if cfg.exp.mode == "tune":
    #     # outer CV: for each test set, we are looking for a unique set of optimal hyperparams
    #     for outer_k in range(0, cfg.exp.n_splits):
    #         # num trials: number of hyperparams sets to test
    #         for trial in range(0, cfg.exp.n_trials):
    #             model_config = model_config_factory(cfg)
    #             # some models require data postprocessing (based on their config)
    #             data, conf.data_info = data_postfactory(
    #                 cfg,
    #                 model_config,
    #                 original_data,
    #             )
    #             model_config["data_info"] = conf.data_info

    #             set_run_name(cfg, outer_k, trial, inner_k)
    #             os.makedirs(conf.run_dir, exist_ok=True)

    #             ###########

    #             # inner CV: CV of the chosen hyperparams
    #             for inner_k in range(conf.n_splits):
    #                 print(
    #                     f"Running tune: k: {outer_k:02d}, Trial: {trial:03d}, \
    #                         Inner k: {inner_k:02d},"
    #                 )

    #                 (
    #                     conf.wandb_trial_name,
    #                     conf.outer_k_dir,
    #                     conf.trial_dir,
    #                     conf.run_dir,
    #                 ) = run_name(conf, outer_k, trial, inner_k)
    #                 os.makedirs(conf.run_dir, exist_ok=True)

    #                 dataloaders = dataloader_factory(
    #                     conf, data, outer_k, trial, inner_k
    #                 )
    #                 model = model_factory(conf, model_config)
    #                 criterion = criterion_factory(conf, model_config)
    #                 optimizer = optimizer_factory(conf, model, model_config)
    #                 scheduler = scheduler_factory(conf, optimizer, model_config)

    #                 logger, model_config["link"] = logger_factory(conf, model_config)

    #                 trainer = trainer_factory(
    #                     conf,
    #                     model_config,
    #                     dataloaders,
    #                     model,
    #                     criterion,
    #                     optimizer,
    #                     scheduler,
    #                     logger,
    #                 )
    #                 results = trainer.run()
    #                 # save results of nested CV
    #                 df = pd.DataFrame(results, index=[0])
    #                 with open(conf.trial_dir + "runs.csv", "a", encoding="utf8") as f:
    #                     df.to_csv(f, header=f.tell() == 0, index=False)

    #                 logger.finish()

    #             # save model config
    #             with open(
    #                 conf.trial_dir + "model_config.json", "w", encoding="utf8"
    #             ) as fp:
    #                 json.dump(model_config, fp, indent=2, cls=NpEncoder)
    #             # read inner CV results, save the average in the fold dir
    #             df = pd.read_csv(conf.trial_dir + "runs.csv")
    #             score = np.mean(df["test_score"].to_numpy())
    #             loss = np.mean(df["test_average_loss"].to_numpy())
    #             df = pd.DataFrame(
    #                 {
    #                     "trial": trial,
    #                     "score": score,
    #                     "loss": loss,
    #                     "path_to_config": conf.trial_dir + "model_config.json",
    #                 },
    #                 index=[0],
    #             )
    #             with open(conf.outer_k_dir + "runs.csv", "a", encoding="utf8") as f:
    #                 df.to_csv(f, header=f.tell() == 0, index=False)

    # elif conf.mode == "exp":
    #     # outer CV: for each test set, we are loading a unique set of optimal hyperparams
    #     for outer_k in range(start_k, conf.n_splits):
    #         # loading best config requires project_name
    #         model_config = model_config_factory(conf, k=outer_k)

    #         # some models require data postprocessing (based on their config)
    #         data, conf.data_info = data_postfactory(
    #             conf,
    #             model_config,
    #             original_data,
    #         )

    #         # num trials: for each test set, we are splitting training set into train/val randomly
    #         if outer_k != start_k:
    #             start_trial = 0
    #         for trial in range(start_trial, conf.n_trials):
    #             print(f"Running exp: k: {outer_k:02d}, Trial: {trial:03d}")

    #             (conf.wandb_trial_name, conf.outer_k_dir, conf.run_dir) = run_name(
    #                 conf, outer_k, trial
    #             )
    #             os.makedirs(conf.run_dir, exist_ok=True)
    #             with open(
    #                 conf.outer_k_dir + "model_config.json", "w", encoding="utf8"
    #             ) as fp:
    #                 json.dump(model_config, fp, indent=2, cls=NpEncoder)

    #             dataloaders = dataloader_factory(conf, data, outer_k, trial)
    #             model = model_factory(conf, model_config)
    #             criterion = criterion_factory(conf, model_config)
    #             optimizer = optimizer_factory(conf, model, model_config)
    #

    #             logger, model_config["link"] = logger_factory(conf, model_config)

    #             trainer = trainer_factory(
    #                 conf,
    #                 model_config,
    #                 dataloaders,
    #                 model,
    #                 criterion,
    #                 optimizer,
    #                 scheduler,
    #                 logger,
    #             )
    #             results = trainer.run()
    #             # save results of the trial
    #             df = pd.DataFrame(results, index=[0])
    #             with open(conf.project_dir + "runs.csv", "a", encoding="utf8") as f:
    #                 df.to_csv(f, header=f.tell() == 0, index=False)

    #             logger.finish()

    # else:
    #     raise ValueError(f"{conf.model} is not recognized")


def tune(cfg, original_data, outer_k=None):
    # for each trial get new set of HPs, test them using CV
    for trial in range(0, cfg.exp.n_trials):
        # get random model config
        model_cfg = model_config_factory(cfg)
        # reshape data according to model config (if needed)
        data = data_postfactory(
            cfg,
            model_cfg,
            original_data,
        )
        # run nested CV
        for inner_k in range(0, cfg.exp.n_splits):
            set_run_name(cfg, outer_k=outer_k, trial=trial, inner_k=inner_k)
            os.makedirs(cfg.run_dir, exist_ok=True)
            dataloaders = dataloader_factory(cfg, data, k=inner_k)
            results = run_trial(cfg, model_cfg, dataloaders)

            # save results of nested CV in the trial directory
            df = pd.DataFrame(results, index=[0])
            with open(f"{cfg.trial_dir}/CV_runs.csv", "a", encoding="utf8") as f:
                df.to_csv(f, header=f.tell() == 0, index=False)

        # save model config
        with open(f"{cfg.trial_dir}/model_config.yaml", "w", encoding="utf8") as f:
            OmegaConf.save(model_cfg, f)

        # summarize the trial's CV results and save them
        df = pd.read_csv(f"{cfg.trial_dir}/CV_runs.csv")
        score = np.mean(df["test_score"].to_numpy())
        loss = np.mean(df["test_average_loss"].to_numpy())
        df = pd.DataFrame(
            {
                "trial": trial,
                "score": score,
                "loss": loss,
                "path_to_config": f"{cfg.trial_dir}/model_config.yaml",
            },
            index=[0],
        )
        with open(f"{cfg.k_dir}/trial_runs.csv", "a", encoding="utf8") as f:
            df.to_csv(f, header=f.tell() == 0, index=False)

    # get optimal config and save it
    df = pd.read_csv(f"{cfg.k_dir}/trial_runs.csv")
    best_idx = df["score"].idxmax()
    best_config_path = df.loc[best_idx]["path_to_config"]
    with open(best_config_path, "r", encoding="utf8") as f:
        best_config = OmegaConf.load(f)
    with open(f"{cfg.k_dir}/best_config.yaml", "w", encoding="utf8") as f:
        OmegaConf.save(best_config, f)


def run_trial(cfg, model_cfg, dataloaders):
    model = model_factory(cfg, model_cfg)
    criterion = criterion_factory(cfg, model_cfg)
    optimizer = optimizer_factory(cfg, model, model_cfg)
    scheduler = scheduler_factory(cfg, optimizer, model_cfg)
    logger = logger_factory(cfg, model_cfg)

    trainer = trainer_factory(
        cfg,
        model_cfg,
        dataloaders,
        model,
        criterion,
        optimizer,
        scheduler,
        logger,
    )

    results = trainer.run()
    logger.finish()

    return results


if __name__ == "__main__":
    start()
