# pylint: disable=W0201,W0223,C0103,C0115,C0116,R0902,E1101,R0914
"""Script for tuning different models on different datasets"""
import os
import csv
import argparse
import json

import pandas as pd
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

from src.settings import LOGS_ROOT, ASSETS_ROOT, UTCNOW
from src.ts_data import (
    load_ABIDE1,
    load_COBRE,
    load_FBIRN,
    load_OASIS,
    load_ABIDE1_869,
    load_UKB,
    TSQuantileTransformer,
)
from src.ts_model import (
    LSTM,
    AnotherLSTM,
    MLP,
    Transformer,
    AttentionMLP,
    NewAttentionMLP,
    AnotherAttentionMLP,
    EnsembleLogisticRegression,
    AnotherEnsembleLogisticRegression,
    MySVM,
    EnsembleSVM,
    No_Res_MLP,
    No_Ens_MLP,
    Transposed_MLP,
)


class Experiment(IExperiment):
    def __init__(
        self,
        mode: str,
        model: str,
        dataset: str,
        quantile: bool,
        n_splits: int,
        max_epochs: int,
        logdir: str,
    ) -> None:
        super().__init__()

        self._mode = mode
        assert not quantile, "Not implemented yet"
        self._model = model
        self._dataset = dataset
        self._quantile: bool = quantile
        self.n_splits = n_splits
        self._trial: optuna.Trial = None
        self.max_epochs = max_epochs
        self.logdir = logdir

    def initialize_dataset(self) -> None:
        if self._dataset == "oasis":
            features, labels = load_OASIS()
        elif self._dataset == "abide":
            features, labels = load_ABIDE1()
        elif self._dataset == "fbirn":
            features, labels = load_FBIRN()
        elif self._dataset == "cobre":
            features, labels = load_COBRE()
        elif self._dataset == "abide_869":
            features, labels = load_ABIDE1_869()
        elif self._dataset == "ukb":
            features, labels = load_UKB()

        self.data_shape = features.shape

        print("data shape: ", self.data_shape)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        skf.get_n_splits(features, labels)

        train_index, test_index = list(skf.split(features, labels))[self.k]

        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=self.data_shape[0] // self.n_splits,
            random_state=42 + self._trial.number,
            stratify=y_train,
        )

        X_train = np.swapaxes(X_train, 1, 2)  # [n_samples; seq_len; n_features]
        X_val = np.swapaxes(X_val, 1, 2)  # [n_samples; seq_len; n_features]
        X_test = np.swapaxes(X_test, 1, 2)  # [n_samples; seq_len; n_features]

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

    def get_model(self):
        config = {}

        if self._mode == "experiment":
            config_file = ""
            config_files = []
            for logdir in os.listdir(LOGS_ROOT):
                if logdir.endswith(f"tune-{args.model}-{args.ds}"):
                    config_files.append(os.path.join(LOGS_ROOT, logdir, "runs.csv"))

            config_file = sorted(config_files)[len(config_files) - 1]

            df = pd.read_csv(config_file, delimiter=",")
            config = df.loc[df["test_score"].idxmax()].to_dict()
            print(config)

            config.pop("test_score")
            config.pop("test_accuracy")
            config.pop("test_loss")

        if self._mode == "tune":
            config["num_epochs"] = self._trial.suggest_int(
                "exp.num_epochs", 30, self.max_epochs
            )
            config["batch_size"] = self._trial.suggest_int(
                "data.batch_size", 4, 32, log=True
            )
        else:
            config["num_epochs"] = self.max_epochs
            config["batch_size"] = int(config["batch_size"])

        if self._model in [
            "mlp",
            "attention_mlp",
            "another_attention_mlp",
            "new_attention_mlp",
            "nores_mlp",
            "noens_mlp",
            "trans_mlp",
        ]:
            if self._mode == "tune":
                config["hidden_size"] = self._trial.suggest_int(
                    "mlp.hidden_size", 32, 256, log=True
                )
                config["num_layers"] = self._trial.suggest_int("mlp.num_layers", 0, 4)
                config["dropout"] = self._trial.suggest_uniform("mlp.dropout", 0.1, 0.9)
                if self._model == "new_attention_mlp":
                    config["attention_size"] = self._trial.suggest_int(
                        "mlp.attention_size", 32, 256, log=True
                    )

            if self._model == "mlp":
                model = MLP(
                    input_size=self.data_shape[1],  # PRIOR
                    output_size=2,  # PRIOR
                    hidden_size=int(config["hidden_size"]),
                    num_layers=int(config["num_layers"]),
                    dropout=config["dropout"],
                )
            elif self._model == "attention_mlp":
                model = AttentionMLP(
                    input_size=self.data_shape[1],  # PRIOR
                    time_length=self.data_shape[2],
                    output_size=2,  # PRIOR
                    hidden_size=int(config["hidden_size"]),
                    num_layers=int(config["num_layers"]),
                    dropout=config["dropout"],
                )
            elif self._model == "another_attention_mlp":
                model = AnotherAttentionMLP(
                    input_size=self.data_shape[1],  # PRIOR
                    time_length=self.data_shape[2],
                    output_size=2,  # PRIOR
                    hidden_size=int(config["hidden_size"]),
                    num_layers=int(config["num_layers"]),
                    dropout=config["dropout"],
                )
            elif self._model == "new_attention_mlp":
                model = NewAttentionMLP(
                    input_size=self.data_shape[1],  # PRIOR
                    time_length=self.data_shape[2],
                    output_size=2,  # PRIOR
                    hidden_size=int(config["hidden_size"]),
                    attention_size=int(config["attention_size"]),
                    num_layers=int(config["num_layers"]),
                    dropout=config["dropout"],
                )
            elif self._model == "nores_mlp":
                model = No_Res_MLP(
                    input_size=self.data_shape[1],  # PRIOR
                    output_size=2,  # PRIOR
                    hidden_size=int(config["hidden_size"]),
                    num_layers=int(config["num_layers"]),
                    dropout=config["dropout"],
                )
            elif self._model == "noens_mlp":
                model = No_Ens_MLP(
                    input_size=self.data_shape[1] * self.data_shape[2],  # PRIOR
                    output_size=2,  # PRIOR
                    hidden_size=int(config["hidden_size"]),
                    num_layers=int(config["num_layers"]),
                    dropout=config["dropout"],
                )
            elif self._model == "trans_mlp":
                model = Transposed_MLP(
                    input_size=self.data_shape[2],  # PRIOR
                    output_size=2,  # PRIOR
                    hidden_size=int(config["hidden_size"]),
                    num_layers=int(config["num_layers"]),
                    dropout=config["dropout"],
                )

        elif self._model in ["lstm", "another_lstm"]:
            if self._mode == "tune":
                config["hidden_size"] = self._trial.suggest_int(
                    "lstm.hidden_size", 32, 256, log=True
                )
                config["num_layers"] = self._trial.suggest_int("lstm.num_layers", 1, 4)
                config["bidirectional"] = self._trial.suggest_categorical(
                    "lstm.bidirectional", [True, False]
                )
                config["fc_dropout"] = self._trial.suggest_uniform(
                    "lstm.fc_dropout", 0.1, 0.9
                )

            if self._model == "lstm":
                model = LSTM(
                    input_size=self.data_shape[1],  # PRIOR
                    output_size=2,  # PRIOR
                    hidden_size=int(config["hidden_size"]),
                    num_layers=int(config["num_layers"]),
                    batch_first=True,
                    bidirectional=config["bidirectional"],
                    fc_dropout=config["fc_dropout"],
                )
            elif self._model == "another_lstm":
                model = AnotherLSTM(
                    input_size=self.data_shape[1],  # PRIOR
                    output_size=2,  # PRIOR
                    hidden_size=int(config["hidden_size"]),
                    num_layers=int(config["num_layers"]),
                    batch_first=True,
                    bidirectional=False,
                    fc_dropout=config["fc_dropout"],
                )

        elif self._model == "transformer":
            if self._mode == "tune":
                config["hidden_size"] = self._trial.suggest_int(
                    "transformer.hidden_size", 4, 128, log=True
                )
                config["num_heads"] = self._trial.suggest_int(
                    "transformer.num_heads", 1, 4
                )
                config["num_layers"] = self._trial.suggest_int(
                    "transformer.num_layers", 1, 4
                )
                config["fc_dropout"] = self._trial.suggest_uniform(
                    "transformer.fc_dropout", 0.1, 0.9
                )

            model = Transformer(
                input_size=self.data_shape[1],  # PRIOR
                output_size=2,  # PRIOR
                hidden_size=int(config["hidden_size"]) * int(config["num_heads"]),
                num_layers=int(config["num_layers"]),
                num_heads=int(config["num_heads"]),
                fc_dropout=config["fc_dropout"],
            )

        else:
            raise NotImplementedError()

        if self._mode == "tune":
            config["lr"] = self._trial.suggest_float("adam.lr", 1e-5, 1e-3, log=True)

        return model, config

    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)

        # init data
        self.initialize_dataset()

        # init wandb logger
        self.wandb_logger: wandb.run = wandb.init(
            project=f"{UTCNOW}-{self._mode}-{self._model}-{self._dataset}",
            name=f"{UTCNOW}-k_{self.k}-trial_{self._trial.number}",
            save_code=True,
        )

        # init model
        self.model, self.config = self.get_model()
        if self._mode == "tune":
            self.config["link"] = self.wandb_logger.get_url()

        self.num_epochs = self.config["num_epochs"]

        self.datasets = {
            "train": DataLoader(
                self._train_ds,
                batch_size=self.config["batch_size"],
                num_workers=0,
                shuffle=True,
            ),
            "valid": DataLoader(
                self._valid_ds,
                batch_size=self.config["batch_size"],
                num_workers=0,
                shuffle=False,
            ),
        }

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["lr"],
        )

        # setup callbacks
        self.callbacks = {
            "early-stop": EarlyStoppingCallback(
                minimize=True,
                patience=30,
                dataset_key="valid",
                metric_key="loss",
                min_delta=0.001,
            ),
            "checkpointer": TorchCheckpointerCallback(
                exp_attr="model",
                logdir=f"{self.logdir}k_{self.k}/{self._trial.number:04d}",
                dataset_key="valid",
                metric_key="loss",
                minimize=True,
            ),
        }

        self.wandb_logger.config.update(self.config)

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
            "accuracy": report["precision"].loc["accuracy"],
            "score": report["auc"].loc["weighted"],
            "loss": total_loss,
        }

    def on_epoch_end(self, exp: "IExperiment") -> None:
        super().on_epoch_end(self)
        self.wandb_logger.log(
            {
                "train_accuracy": self.epoch_metrics["train"]["accuracy"],
                "train_score": self.epoch_metrics["train"]["score"],
                "train_loss": self.epoch_metrics["train"]["loss"],
                "valid_accuracy": self.epoch_metrics["valid"]["accuracy"],
                "valid_score": self.epoch_metrics["valid"]["score"],
                "valid_loss": self.epoch_metrics["valid"]["loss"],
            },
        )

    def run_test_dataset(self) -> None:

        test_ds = DataLoader(
            self._test_ds,
            batch_size=self.config["batch_size"],
            num_workers=0,
            shuffle=False,
        )

        f = open(f"{self.logdir}/k_{self.k}/{self._trial.number:04d}/model.storage.json")
        logpath = json.load(f)["storage"][0]["logpath"]
        checkpoint = torch.load(logpath, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint)
        self.model.train(False)
        self.model.zero_grad()

        all_scores, all_targets, all_logits = [], [], []
        total_loss = 0.0

        with torch.set_grad_enabled(False):
            for self.dataset_batch_step, (data, target) in enumerate(tqdm(test_ds)):
                self.optimizer.zero_grad()
                logits = self.model(data)
                loss = self.criterion(logits, target)
                score = torch.softmax(logits, dim=-1)

                all_scores.append(score.cpu().detach().numpy())
                all_targets.append(target.cpu().detach().numpy())
                all_logits.append(logits.cpu().detach().numpy())
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

        print("Accuracy ", report["precision"].loc["accuracy"])
        print("AUC ", report["auc"].loc["weighted"])
        print("Loss ", total_loss)
        self.wandb_logger.log(
            {
                "test_accuracy": report["precision"].loc["accuracy"],
                "test_score": report["auc"].loc["weighted"],
                "test_loss": total_loss,
            },
        )
        self.config["test_accuracy"] = report["precision"].loc["accuracy"]
        self.config["test_score"] = report["auc"].loc["weighted"]
        self.config["test_loss"] = total_loss

        # logits for logistic regression
        all_targets = np.concatenate(all_targets, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)

        np.savez(
            f"{self.logdir}/k_{self.k}/{self._trial.number:04d}/test_scores.npz",
            logits=all_logits,
            targets=all_targets,
        )

    def on_experiment_end(self, exp: "IExperiment") -> None:
        super().on_experiment_end(exp)
        self._score = self.callbacks["early-stop"].best_score

        print("Run test dataset")
        self.run_test_dataset()

        self.wandb_logger.finish()
        if os.path.exists(f"{self.logdir}/runs.csv"):
            with open(f"{self.logdir}/runs.csv", "a") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.config.keys())
                writer.writerow(self.config)
        else:
            with open(f"{self.logdir}/runs.csv", "a") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.config.keys())
                writer.writeheader()
                writer.writerow(self.config)

    def _objective(self, trial) -> float:
        self._trial = trial
        self.run()
        return self._score

    def tune(self, n_trials: int):
        for k in range(self.n_splits):
            self.k = k
            self.study = optuna.create_study(direction="maximize")
            self.study.optimize(self._objective, n_trials=n_trials, n_jobs=1)
            logfile = f"{self.logdir}/optuna_{k}.csv"
            df = self.study.trials_dataframe()
            df.to_csv(logfile, index=False)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "tune",
            "experiment",
        ],
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "mlp",
            "attention_mlp",
            "another_attention_mlp",
            "new_attention_mlp",
            "nores_mlp",
            "noens_mlp",
            "trans_mlp",
            "lstm",
            "another_lstm",
            "transformer",
            "my_lr",
            "ens_lr",
            "another_ens_lr",
            "my_svm",
            "ens_svm",
        ],
        required=True,
    )
    parser.add_argument(
        "--ds",
        type=str,
        choices=["oasis", "abide", "fbirn", "cobre", "abide_869", "ukb"],
        required=True,
    )
    boolean_flag(parser, "quantile", default=False)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--num-splits", type=int, default=5)
    args = parser.parse_args()

    Experiment(
        mode=args.mode,
        model=args.model,
        dataset=args.ds,
        quantile=args.quantile,
        n_splits=args.num_splits,
        max_epochs=args.max_epochs,
        logdir=f"{LOGS_ROOT}/{UTCNOW}-{args.mode}-{args.model}-{args.ds}/",
    ).tune(n_trials=args.num_trials)
