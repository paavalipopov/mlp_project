# pylint: disable=W0201,W0223,C0103,C0115,C0116,R0902,E1101,R0914
"""Script for tuning different models on different datasets"""
import os
import csv
import sys
import argparse
import json
import re
import shutil

import pandas as pd
from apto.utils.misc import boolean_flag
from apto.utils.report import get_classification_report
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
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
    load_ABIDE1_869,
    load_UKB,
    load_BSNIP,
    load_time_FBIRN,
    load_ROI_FBIRN,
    load_HCP,
    load_ROI_HCP,
    load_ROI_ABIDE,
)
from src.ts_model import (
    LSTM,
    NoahLSTM,
    MLP,
    Transformer,
    AttentionMLP,
    NewAttentionMLP,
)
from src.ts_model_tests import (
    NewestAttentionMLP,
    No_Res_MLP,
    No_Ens_MLP,
    Transposed_MLP,
    UltimateAttentionMLP,
    DeepNoahLSTM,
    MeanTransformer,
    LastNoahLSTM,
)


class Experiment(IExperiment):
    def __init__(
        self,
        mode: str,
        path: str,
        model: str,
        dataset: str,
        scaled: bool,
        test_datasets: list,
        prefix: str,
        quantile: bool,
        n_splits: int,
        n_trials: int,
        max_epochs: int,
    ) -> None:
        super().__init__()
        self.utcnow = UTCNOW
        # starting fold/trial; used in resumed experiments
        self.start_k = 0
        self.start_trial = 0

        if mode == "resume":
            (
                mode,
                model,
                dataset,
                scaled,
                test_datasets,
                prefix,
            ) = self.acquire_cont_params(path, n_trials)

        self._mode = mode  # tune or experiment mode
        assert not quantile, "Not implemented yet"
        self._model = model  # model name
        self._dataset = dataset  # main dataset name (used for training)
        self._scaled = scaled  # if dataset should be scaled by sklearn's StandardScaler
        self._quantile = quantile  # Not implemented, False

        if test_datasets is None:  # additional test datasets
            self._test_datasets = []
        else:
            if self._mode == "tune":
                print("'Tune' mode overrides additional test datasets")
                self._test_datasets = []
            else:
                self._test_datasets = test_datasets
        if self._dataset in self._test_datasets:
            # Fraction of the main dataset is always used as a test dataset;
            # no need for it in the list of test datasets
            print(
                f"Received main dataset {self._dataset} among test datasets {self._test_datasets}; removed"
            )
            self._test_datasets.remove(self._dataset)

        self.n_splits = n_splits  # num of splits for StratifiedKFold
        self.n_trials = n_trials  # num of trials for each fold
        self._scaler: StandardScaler = StandardScaler()  # scaler for datasets
        self._trial: optuna.Trial = None
        self.max_epochs = max_epochs

        # set project name prefix
        if len(prefix) == 0:
            self.project_prefix = f"{self.utcnow}"
        else:
            # '-'s are reserved for name parsing
            self.project_prefix = prefix.replace("-", "_")

        self.project_name = f"{self._mode}-{self._model}-{self._dataset}"
        if self._scaled:
            self.project_name = f"{self._mode}-{self._model}-scaled_{self._dataset}"
        if len(self._test_datasets) != 0:
            project_ending = "-tests-" + "_".join(self._test_datasets)
            self.project_name += project_ending

        self.logdir = f"{LOGS_ROOT}/{self.project_prefix}-{self.project_name}/"

    def acquire_cont_params(self, path, n_trials):
        """
        Used for extracting experiments set-up from the
        given path for continuing an interrupted experiment
        """
        project_params = path.split("/")
        project_params = project_params[-1].split("-")

        # get params
        if re.match("(^\d{6}.\d{6}$)", project_params[0]):
            self.utcnow = project_params[0]
            prefix = ""
        else:
            prefix = project_params[0]

        mode = project_params[1]
        model = project_params[2]

        if "scaled" in project_params[3]:
            scaled = True
            dataset = project_params[3].replace("scaled_", "")
        else:
            scaled = False
            dataset = project_params[3]

        try:
            test_datasets = project_params[4]
            test_datasets.split("_")
            test_datasets = test_datasets[1:]
        except IndexError:
            test_datasets = None

        # find when the experiment got interrupted
        with open(path + "/runs.csv", "r") as fp:
            lines = len(fp.readlines()) - 1
            self.start_k = lines // n_trials
            self.start_trial = lines - self.start_k * n_trials

        # delete failed run
        faildir = path + f"/k_{self.start_k}/{self.start_trial:04d}"
        print("Deleting interrupted run logs in " + faildir)
        try:
            shutil.rmtree(faildir)
        except FileNotFoundError:
            print("Could not delete interrupted run logs - FileNotFoundError")

        return mode, model, dataset, scaled, test_datasets, prefix

    def initialize_dataset(self, dataset, for_test=False):
        # load core dataset (or additional datasets if for_test==True)
        # your dataset should have shape [n_features; n_channels; time_len]
        if dataset in [
            "fbirn_cobre",
            "fbirn_bsnip",
            "bsnip_cobre",
            "bsnip_fbirn",
            "cobre_fbirn",
            "cobre_bsnip",
        ]:
            # cross datasets; somewhat cheaty, don't use
            if dataset == "fbirn_cobre":
                X_train, y_train = load_FBIRN()
                X_test, y_test = load_COBRE()
            if dataset == "fbirn_bsnip":
                X_train, y_train = load_FBIRN()
                X_test, y_test = load_BSNIP()
            if dataset == "bsnip_cobre":
                X_train, y_train = load_BSNIP()
                X_test, y_test = load_COBRE()
            if dataset == "bsnip_fbirn":
                X_train, y_train = load_BSNIP()
                X_test, y_test = load_FBIRN()
            if dataset == "cobre_fbirn":
                X_train, y_train = load_COBRE()
                X_test, y_test = load_FBIRN()
            if dataset == "cobre_bsnip":
                X_train, y_train = load_COBRE()
                X_test, y_test = load_BSNIP()

            self.data_shape = X_train.shape
            print(f"data shapes: {self.data_shape}; {X_test.shape}")

            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=self.data_shape[0] // self.n_splits,
                random_state=42 * self.k + self.trial,
                stratify=y_train,
            )

        else:
            if dataset == "oasis":
                features, labels = load_OASIS()
            elif dataset == "abide":
                features, labels = load_ABIDE1()
            elif dataset == "fbirn":
                features, labels = load_FBIRN()
            elif dataset == "cobre":
                features, labels = load_COBRE()
            elif dataset == "abide_869":
                features, labels = load_ABIDE1_869()
            elif dataset == "ukb":
                features, labels = load_UKB()
            elif dataset == "bsnip":
                features, labels = load_BSNIP()
            elif dataset == "time_fbirn":
                features, labels = load_time_FBIRN()
            elif dataset == "fbirn_100":
                features, labels = load_ROI_FBIRN(100)
            elif dataset == "fbirn_200":
                features, labels = load_ROI_FBIRN(200)
            elif dataset == "fbirn_400":
                features, labels = load_ROI_FBIRN(400)
            elif dataset == "fbirn_1000":
                features, labels = load_ROI_FBIRN(1000)
            elif dataset == "hcp":
                features, labels = load_HCP()
            elif dataset == "hcp_roi":
                features, labels = load_ROI_HCP()
            elif dataset == "abide_roi":
                features, labels = load_ROI_ABIDE()
            else:
                raise NotImplementedError()

            if self._scaled:
                # # Time scaling
                # features_shape = features.shape # [n_features; n_channels; time_len]
                # features = features.reshape(-1, features_shape[2])
                # features = features.swapaxes(0, 1)

                # features = self._scaler.fit_transform(features) # first dimension is scaled

                # features = features.swapaxes(0, 1)
                # features = features.reshape(features_shape)

                # # channel scaling
                # features = features.swapaxes(1, 2)
                # features_shape = features.shape  # [n_features; time_len; n_channels]
                # features = features.reshape(-1, features_shape[2])
                # features = features.swapaxes(0, 1)

                # features = self._scaler.fit_transform(
                #     features
                # )  # first dimension is scaled

                # features = features.swapaxes(0, 1)
                # features = features.reshape(features_shape)
                # features = features.swapaxes(1, 2)

                # time-channel scaling
                features_shape = features.shape  # [n_features; n_channels; time_len]
                features = features.reshape(features_shape[0], -1)
                features = features.swapaxes(0, 1)

                features = self._scaler.fit_transform(
                    features
                )  # first dimension is scaled

                features = features.swapaxes(0, 1)
                features = features.reshape(features_shape)

            if for_test:
                # if dataset is loaded for tests, it should not be
                # split into train/val/test

                features = np.swapaxes(
                    features, 1, 2
                )  # [n_features; time_len; n_channels]

                return TensorDataset(
                    torch.tensor(features, dtype=torch.float32),
                    torch.tensor(labels, dtype=torch.int64),
                )

            self.data_shape = features.shape  # [n_features; n_channels; time_len]

            print("data shape: ", self.data_shape)
            # train-val/test split
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            skf.get_n_splits(features, labels)

            train_index, test_index = list(skf.split(features, labels))[self.k]

            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # train/val split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=self.data_shape[0] // self.n_splits,
                random_state=42 + self.trial,
                stratify=y_train,
            )

        X_train = np.swapaxes(X_train, 1, 2)  # [n_features; time_len; n_channels;]
        X_val = np.swapaxes(X_val, 1, 2)  # [n_features; time_len; n_channels;]
        X_test = np.swapaxes(X_test, 1, 2)  # [n_features; time_len; n_channels;]

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
        # return model for experiments
        config = {}

        if self._mode == "experiment":
            # find and load the best tuned model
            config_files = []

            searched_dir = self.project_name.split("-")
            serached_dir = f"tune-{searched_dir[1]}-{searched_dir[2]}"
            if self.project_prefix != f"{self.utcnow}":
                serached_dir = f"{self.project_prefix}-{serached_dir}"
            print(f"Searching trained model in ./assets/logs/*{serached_dir}")
            for logdir in os.listdir(LOGS_ROOT):
                if logdir.endswith(serached_dir):
                    config_files.append(os.path.join(LOGS_ROOT, logdir, "runs.csv"))

            # if multiple configs found, choose the latest
            config_file = sorted(config_files)[-1]
            print(f"Using best model from {config_file}")

            df = pd.read_csv(config_file, delimiter=",")
            # pick hyperparams of a model with the highest test_score
            config = df.loc[df["test_score"].idxmax()].to_dict()
            print(config)

            config.pop("test_score")
            config.pop("test_accuracy")
            config.pop("test_loss")

        if self._mode == "tune":
            # pick hyperparams randomly from some range

            config["num_epochs"] = self._trial.suggest_int(
                "exp.num_epochs", 30, self.max_epochs
            )
            # pick the max batch_size based on the data shape (fix for /0 exception for some datasets)
            max_batch_size = min((32, int(self.data_shape[0] / self.n_splits) - 1))
            config["batch_size"] = self._trial.suggest_int(
                "data.batch_size", 4, max_batch_size, log=True
            )
        else:
            config["num_epochs"] = self.max_epochs
            config["batch_size"] = int(config["batch_size"])

        if self._model in [
            "mlp",
            "wide_mlp",
            "deep_mlp",
            "attention_mlp",
            "new_attention_mlp",
            "newest_attention_mlp",
            "nores_mlp",
            "noens_mlp",
            "trans_mlp",
            "ultimate_attention_mlp",
        ]:
            if self._mode == "tune":
                if self._model == "wide_mlp":
                    config["hidden_size"] = self._trial.suggest_int(
                        "mlp.hidden_size", 256, 1024, log=True
                    )
                    config["num_layers"] = self._trial.suggest_int(
                        "mlp.num_layers", 0, 4
                    )
                elif self._model == "deep_mlp":
                    config["hidden_size"] = self._trial.suggest_int(
                        "mlp.hidden_size", 32, 256, log=True
                    )
                    config["num_layers"] = self._trial.suggest_int(
                        "mlp.num_layers", 4, 20
                    )
                else:
                    config["hidden_size"] = self._trial.suggest_int(
                        "mlp.hidden_size", 32, 256, log=True
                    )
                    config["num_layers"] = self._trial.suggest_int(
                        "mlp.num_layers", 0, 4
                    )
                config["dropout"] = self._trial.suggest_uniform("mlp.dropout", 0.1, 0.9)
                if self._model in [
                    "new_attention_mlp",
                    "newest_attention_mlp",
                    "ultimate_attention_mlp",
                ]:
                    config["attention_size"] = self._trial.suggest_int(
                        "mlp.attention_size", 32, 256, log=True
                    )

            if self._model in ["mlp", "wide_mlp", "deep_mlp"]:
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
            elif self._model == "newest_attention_mlp":
                model = NewestAttentionMLP(
                    input_size=self.data_shape[1],  # PRIOR
                    time_length=self.data_shape[2],
                    output_size=2,  # PRIOR
                    hidden_size=int(config["hidden_size"]),
                    attention_size=int(config["attention_size"]),
                    num_layers=int(config["num_layers"]),
                    dropout=config["dropout"],
                )
            elif self._model == "ultimate_attention_mlp":
                model = UltimateAttentionMLP(
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

        elif self._model in ["lstm", "noah_lstm", "deep_noah_lstm", "last_noah_lstm"]:
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
            elif self._model == "noah_lstm":
                model = NoahLSTM(
                    input_size=self.data_shape[1],  # PRIOR
                    output_size=2,  # PRIOR
                    hidden_size=int(config["hidden_size"]),
                    num_layers=int(config["num_layers"]),
                    batch_first=True,
                    bidirectional=False,
                    fc_dropout=config["fc_dropout"],
                )
            elif self._model == "deep_noah_lstm":
                model = DeepNoahLSTM(
                    input_size=self.data_shape[1],  # PRIOR
                    output_size=2,  # PRIOR
                    hidden_size=int(config["hidden_size"]),
                    num_layers=int(config["num_layers"]),
                    batch_first=True,
                    bidirectional=False,
                    fc_dropout=config["fc_dropout"],
                )
            elif self._model == "last_noah_lstm":
                model = LastNoahLSTM(
                    input_size=self.data_shape[1],  # PRIOR
                    output_size=2,  # PRIOR
                    hidden_nodes=int(config["hidden_size"]),
                )

        elif self._model in ["transformer", "mean_transformer"]:
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
            if self._model == "transformer":
                model = Transformer(
                    input_size=self.data_shape[1],  # PRIOR
                    output_size=2,  # PRIOR
                    hidden_size=int(config["hidden_size"]) * int(config["num_heads"]),
                    num_layers=int(config["num_layers"]),
                    num_heads=int(config["num_heads"]),
                    fc_dropout=config["fc_dropout"],
                )
            elif self._model == "mean_transformer":
                model = MeanTransformer(
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
            config["link"] = self.wandb_logger.get_url()

        return model, config

    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)

        # init wandb logger
        self.wandb_logger: wandb.run = wandb.init(
            project=f"{self.project_prefix}-{self.project_name}",
            name=f"{self.utcnow}-k_{self.k}-trial_{self.trial}",
            save_code=True,
        )

        # init data
        self.initialize_dataset(self._dataset)

        # init model
        self.model, self.config = self.get_model()

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
                logdir=f"{self.logdir}k_{self.k}/{self.trial:04d}",
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

        total_loss /= self.dataset_batch_step + 1

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

    def run_test_dataset(self, dataset_name, test_dataset) -> None:
        # runs given dataset in a test mode
        test_ds = DataLoader(
            test_dataset,
            batch_size=self.config["batch_size"],
            num_workers=0,
            shuffle=False,
        )

        # load bst model weights
        f = open(f"{self.logdir}/k_{self.k}/{self.trial:04d}/model.storage.json")
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

        total_loss /= self.dataset_batch_step + 1

        y_test = np.hstack(all_targets)
        y_score = np.vstack(all_scores)
        y_pred = np.argmax(y_score, axis=-1).astype(np.int32)
        report = get_classification_report(
            y_true=y_test, y_pred=y_pred, y_score=y_score, beta=0.5
        )
        for stats_type in [0, 1, "macro", "weighted"]:
            stats = report.loc[stats_type]
            for key, value in stats.items():
                if "support" not in key and dataset_name == "self":
                    self._trial.set_user_attr(f"{key}_{stats_type}", float(value))

        print(f"On {dataset_name} dataset:")
        print("Accuracy ", report["precision"].loc["accuracy"])
        print("AUC ", report["auc"].loc["weighted"])
        print("Loss ", total_loss)

        if dataset_name == "self":
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

            # save logits for logistic regression
            all_targets = np.concatenate(all_targets, axis=0)
            all_logits = np.concatenate(all_logits, axis=0)

            np.savez(
                f"{self.logdir}/k_{self.k}/{self.trial:04d}/test_scores.npz",
                logits=all_logits,
                targets=all_targets,
            )
        else:
            self.wandb_logger.log(
                {
                    f"{dataset_name}_test_accuracy": report["precision"].loc["accuracy"],
                    f"{dataset_name}_test_score": report["auc"].loc["weighted"],
                    f"{dataset_name}_test_loss": total_loss,
                },
            )

    def on_experiment_end(self, exp: "IExperiment") -> None:
        super().on_experiment_end(exp)
        self._score = self.callbacks["early-stop"].best_score

        print("Run test dataset")
        self.run_test_dataset("self", self._test_ds)

        print("Run additional test datasets")
        for dataset_name in self._test_datasets:
            self.run_test_dataset(
                dataset_name, self.initialize_dataset(dataset_name, for_test=True)
            )

        self.wandb_logger.finish()
        # log hyperparams and metrics (stored in self.condig); crucial for tuning
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

    def tune(self):
        folds_of_interest = []

        if self.start_trial != self.n_trials:
            for trial in range(self.start_trial, self.n_trials):
                folds_of_interest += [(self.start_k, trial)]
            for k in range(self.start_k + 1, self.n_splits):
                for trial in range(self.n_trials):
                    folds_of_interest += [(k, trial)]
        else:
            raise IndexError

        for k, trial in folds_of_interest:
            self.k = k  # k'th test fold
            self.trial = trial  # trial'th trial on the k'th fold
            self.study = optuna.create_study(direction="maximize")
            self.study.optimize(self._objective, n_trials=1, n_jobs=1)
            # log optuna Study's metricts (not really used)
            logfile = f"{self.logdir}/optuna_{k}_{trial}.csv"
            df = self.study.trials_dataframe()
            df.to_csv(logfile, index=False)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    for_continue = "resume" in sys.argv

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "tune",
            "experiment",
            "resume",
        ],
        required=True,
        help="'tune' for model hyperparams tuning; 'experiment' for experiments with tuned model; 'resume' for resuming interrupted experiment",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=for_continue,
        help="path to the interrupted experiment (e.g., /Users/user/mlp_project/assets/logs/prefix-mode-model-ds)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "mlp",
            "wide_mlp",
            "deep_mlp",
            "attention_mlp",
            "new_attention_mlp",
            "newest_attention_mlp",
            "nores_mlp",
            "noens_mlp",
            "trans_mlp",
            "lstm",
            "noah_lstm",
            "deep_noah_lstm",
            "last_noah_lstm",
            "transformer",
            "mean_transformer",
            "my_lr",
            "ens_lr",
            "another_ens_lr",
            "my_svm",
            "ens_svm",
            "ultimate_attention_mlp",
        ],
        required=not for_continue,
        help="Name of the model to run",
    )
    parser.add_argument(
        "--ds",
        type=str,
        choices=[
            "oasis",
            "abide",
            "fbirn",
            "cobre",
            "abide_869",
            "ukb",
            "bsnip",
            "fbirn_cobre",
            "fbirn_bsnip",
            "bsnip_cobre",
            "bsnip_fbirn",
            "cobre_fbirn",
            "cobre_bsnip",
            "time_fbirn",
            "fbirn_100",
            "fbirn_200",
            "fbirn_400",
            "fbirn_1000",
            "hcp",
            "hcp_roi",
            "abide_roi",
        ],
        required=not for_continue,
        help="Name of the dataset to use for training",
    )

    parser.add_argument(
        "--test-ds",
        nargs="*",
        type=str,
        choices=[
            "oasis",
            "abide",
            "fbirn",
            "cobre",
            "abide_869",
            "ukb",
            "bsnip",
        ],
        help="Additional datasets for testing",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for the project name (body of the project name \
            is '$mode-$model-$dataset'): default: UTC time",
    )

    boolean_flag(parser, "quantile", default=False)  # not implemented
    # whehter dataset should be scaled by sklearn's StandardScaler
    boolean_flag(parser, "scaled", default=False)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=30,
        help="Max number of epochs (min 30)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=10,
        help="Number of trials to run on each test fold",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=5,
        help="Number of splits for StratifiedKFold (affects the number of test folds)",
    )
    args = parser.parse_args()

    Experiment(
        mode=args.mode,
        path=args.path,
        model=args.model,
        dataset=args.ds,
        scaled=args.scaled,
        test_datasets=args.test_ds,
        prefix=args.prefix,
        quantile=args.quantile,
        n_splits=args.num_splits,
        n_trials=args.num_trials,
        max_epochs=args.max_epochs,
    ).tune()
