# pylint: disable=W0201,W0223,C0103,C0115,C0116,R0902,E1101,R0914
"""Experiment on OASIS data with LSTM model"""
import argparse
import json

from apto.utils.misc import boolean_flag
from apto.utils.report import get_classification_report
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from animus import EarlyStoppingCallback, IExperiment
from animus.torch.callbacks import TorchCheckpointerCallback
import wandb

from src.settings import LOGS_ROOT, UTCNOW
from src.ts_data import (
    load_ABIDE1,
    load_COBRE,
    load_FBIRN,
    load_OASIS,
    TSQuantileTransformer,
)
from src.ts_model import LSTM, MLP, Transformer


class Experiment(IExperiment):
    def __init__(self, max_epochs: int, logdir: str, project_name: str) -> None:
        super().__init__()
        self._trial: optuna.Trial = None
        self.max_epochs = max_epochs
        self.logdir = logdir
        self.project_name = project_name

    def initialize_data(self) -> None:
        features, labels = load_OASIS()
        self.data_shape = features.shape

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        skf.get_n_splits(features, labels)

        train_index, test_index = list(skf.split(features, labels))[self.k]

        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=165,
            random_state=42 + self._trial.number,
            stratify=y_train,
        )

        X_train = np.swapaxes(X_train, 1, 2)  # [n_samples; seq_len; n_features]
        X_val = np.swapaxes(X_val, 1, 2)  # [n_samples; seq_len; n_features]
        X_test = np.swapaxes(X_test, 1, 2)

        self._train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.int64),
        )
        self._valid_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.int64),
        )
        self._test_ds = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.int64),
        )

    def on_experiment_start(self, exp: "IExperiment"):
        # init wandb logger
        self.wandb_logger: wandb.run = wandb.init(
            project=self.project_name,
            name=f"{UTCNOW}-k_{self.k}-trial_{self._trial.number}",
        )

        super().on_experiment_start(exp)

        self.initialize_data()

        # setup experiment
        self.num_epochs = 64
        # setup data
        self.batch_size = 16
        self.datasets = {
            "train": DataLoader(
                self._train_ds, batch_size=self.batch_size, num_workers=0, shuffle=True
            ),
            "valid": DataLoader(
                self._valid_ds, batch_size=self.batch_size, num_workers=0, shuffle=False
            ),
        }
        # setup model
        hidden_size = 52
        num_layers = 3
        bidirectional = False
        fc_dropout = 0.2626756675371412
        lr = 0.000403084751422323

        self.model = LSTM(
            input_size=self.data_shape[1],  # 53
            input_len=self.data_shape[2],  # 156
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            fc_dropout=fc_dropout,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
        )
        # setup callbacks
        self.callbacks = {
            "early-stop": EarlyStoppingCallback(
                minimize=True,
                patience=16,
                dataset_key="valid",
                metric_key="loss",
                min_delta=0.001,
            ),
            "checkpointer": TorchCheckpointerCallback(
                exp_attr="model",
                logdir=f"{self.logdir}/k_{self.k}/{self._trial.number:04d}",
                dataset_key="valid",
                metric_key="loss",
                minimize=True,
            ),
        }

        self.wandb_logger.config.update(
            {
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "bidirectional": bidirectional,
                "fc_dropout": fc_dropout,
                "lr": lr,
            }
        )

    def run_dataset(self) -> None:
        all_scores, all_targets = [], []
        total_loss = 0.0
        self.model.train(self.is_train_dataset)

        with torch.set_grad_enabled(self.is_train_dataset):
            for self.dataset_batch_step, (data, target) in enumerate(tqdm(self.dataset)):
                self.optimizer.zero_grad()
                logits = self.model(data)
                loss = self.criterion(logits, target)
                score = torch.softmax(logits, dim=-1)

                all_scores.append(score.cpu().detach().numpy())
                all_targets.append(target.cpu().detach().numpy())
                total_loss += loss.sum().item()
                if self.is_train_dataset:
                    loss.backward()
                    self.optimizer.step()

        total_loss /= self.dataset_batch_step

        y_test = np.hstack(all_targets)
        y_score = np.vstack(all_scores)
        y_pred = np.argmax(y_score, axis=-1).astype(np.int32)
        report = get_classification_report(
            y_true=y_test, y_pred=y_pred, y_score=y_score, beta=0.5
        )
        for stats_type in [0, 1, "macro", "weighted"]:
            stats = report.loc[stats_type]
            for key, value in stats.items():
                if "support" not in key:
                    self._trial.set_user_attr(f"{key}_{stats_type}", float(value))

        self.dataset_metrics = {
            "score": report["auc"].loc["weighted"],
            "loss": total_loss,
        }

    def on_epoch_end(self, exp: "IExperiment") -> None:
        super().on_epoch_end(self)
        self.wandb_logger.log(
            {
                "train_score": self.epoch_metrics["train"]["score"],
                "train_loss": self.epoch_metrics["train"]["loss"],
                "valid_score": self.epoch_metrics["valid"]["score"],
                "valid_loss": self.epoch_metrics["valid"]["loss"],
            },
        )

    def run_test_dataset(self) -> None:

        test_ds = DataLoader(
            self._test_ds, batch_size=self.batch_size, num_workers=0, shuffle=False
        )

        f = open(f"{self.logdir}/k_{self.k}/{self._trial.number:04d}/model.storage.json")
        logpath = json.load(f)["storage"][0]["logpath"]
        checkpoint = torch.load(logpath, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint)
        self.model.train(False)
        self.model.zero_grad()

        all_scores, all_targets = [], []
        total_loss = 0.0

        with torch.set_grad_enabled(False):
            for self.dataset_batch_step, (data, target) in enumerate(tqdm(test_ds)):
                self.optimizer.zero_grad()
                logits = self.model(data)
                loss = self.criterion(logits, target)
                score = torch.softmax(logits, dim=-1)

                all_scores.append(score.cpu().detach().numpy())
                all_targets.append(target.cpu().detach().numpy())
                total_loss += loss.sum().item()

        total_loss /= self.dataset_batch_step

        y_test = np.hstack(all_targets)
        y_score = np.vstack(all_scores)
        y_pred = np.argmax(y_score, axis=-1).astype(np.int32)
        report = get_classification_report(
            y_true=y_test, y_pred=y_pred, y_score=y_score, beta=0.5
        )
        for stats_type in [0, 1, "macro", "weighted"]:
            stats = report.loc[stats_type]
            for key, value in stats.items():
                if "support" not in key:
                    self._trial.set_user_attr(f"{key}_{stats_type}", float(value))

        print("AUC ", report["auc"].loc["weighted"])
        print("Loss ", total_loss)
        self.wandb_logger.log(
            {
                "test_score": report["auc"].loc["weighted"],
                "test_loss": total_loss,
            },
        )

    def on_experiment_end(self, exp: "IExperiment") -> None:
        super().on_experiment_end(exp)
        self._score = self.callbacks["early-stop"].best_score

        print("Run test dataset")
        self.run_test_dataset()

        self.wandb_logger.finish()

    def _objective(self, trial) -> float:
        self._trial = trial
        self.run()

        return self._score

    def tune(self, n_trials: int):
        for k in range(5):
            self.k = k
            self.study = optuna.create_study(direction="maximize")
            self.study.optimize(self._objective, n_trials=n_trials, n_jobs=1)
            logfile = f"{self.logdir}/optuna.csv"
            df = self.study.trials_dataframe()
            df.to_csv(logfile, index=False)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--num-trials", type=int, default=1)
    args = parser.parse_args()

    project_name = "lstm-oasis"
    Experiment(
        max_epochs=args.max_epochs,
        logdir=f"{LOGS_ROOT}/{UTCNOW}-{project_name}/",
        project_name=project_name,
    ).tune(n_trials=args.num_trials)
