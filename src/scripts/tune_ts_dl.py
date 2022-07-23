# pylint: disable=W0201,W0223,C0103,C0115,C0116,R0902,E1101,R0914
"""Script for tuning different models on different datasets"""
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
from src.ts_model import LSTM, MLP, Transformer, AttentionMLP, AnotherAttentionMLP
from src.ts_MILC import combinedModel, NatureOneCNN, subjLSTM


class Experiment(IExperiment):
    def __init__(
        self,
        model: str,
        dataset: str,
        quantile: bool,
        n_splits: int,
        max_epochs: int,
        logdir: str,
    ) -> None:
        super().__init__()
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

        if self._model == "milc":
            # MILC encoder is trained for this data shape (needs check or encoder retraining)
            if self._dataset == "oasis":
                features = features[:, :, :140]
                self.data_shape = features.shape

            sample_x = self.data_shape[1]
            sample_y = 20
            samples_per_subject = 13
            window_shift = 10

            # reshape initial data into [num_of_subjects, samples_per_subject, sample_x, sample_y]
            # there will be a window_shift overlap in 4th (sample_y) dimension
            finalData = np.zeros(
                (self.data_shape[0], samples_per_subject, sample_x, sample_y)
            )
            for i in range(self.data_shape[0]):
                for j in range(samples_per_subject):
                    # print(
                    #     f"finalData[{i}, {j}, :, :] = features [{i}, :, {(j * window_shift)} : {(j * window_shift) + sample_y}]"
                    # )
                    finalData[i, j, :, :] = features[
                        i, :, (j * window_shift) : (j * window_shift) + sample_y
                    ]
            features = torch.from_numpy(finalData).float()
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

        if self._model != "milc":
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

    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)

        # init data
        self.initialize_dataset()

        # init wandb logger
        self.wandb_logger: wandb.run = wandb.init(
            project=f"tune-{self._model}-{self._dataset}",
            name=f"{UTCNOW}-k_{self.k}-trial_{self._trial.number}",
        )
        # config dict for wandb
        wandb_config = {}

        self.num_epochs = self._trial.suggest_int("exp.num_epochs", 30, self.max_epochs)

        # setup data
        self.batch_size = self._trial.suggest_int("data.batch_size", 4, 32, log=True)
        self.datasets = {
            "train": DataLoader(
                self._train_ds, batch_size=self.batch_size, num_workers=0, shuffle=True
            ),
            "valid": DataLoader(
                self._valid_ds, batch_size=self.batch_size, num_workers=0, shuffle=False
            ),
        }

        # setup model
        if self._model in ["mlp", "attention_mlp", "another_attention_mlp"]:
            hidden_size = self._trial.suggest_int("mlp.hidden_size", 32, 256, log=True)
            num_layers = self._trial.suggest_int("mlp.num_layers", 0, 4)
            dropout = self._trial.suggest_uniform("mlp.dropout", 0.1, 0.9)

            if self._model == "mlp":
                self.model = MLP(
                    input_size=self.data_shape[1],  # PRIOR
                    output_size=2,  # PRIOR
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            elif self._model == "attention_mlp":
                self.model = AttentionMLP(
                    input_size=self.data_shape[1],  # PRIOR
                    time_length=self.data_shape[2],
                    output_size=2,  # PRIOR
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            if self._model == "another_attention_mlp":
                self.model = AnotherAttentionMLP(
                    input_size=self.data_shape[1],  # PRIOR
                    time_length=self.data_shape[2],
                    output_size=2,  # PRIOR
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            wandb_config = {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
            }

        elif self._model == "lstm":
            hidden_size = self._trial.suggest_int("lstm.hidden_size", 32, 256, log=True)
            num_layers = self._trial.suggest_int("lstm.num_layers", 1, 4)
            bidirectional = self._trial.suggest_categorical(
                "lstm.bidirectional", [True, False]
            )
            fc_dropout = self._trial.suggest_uniform("lstm.fc_dropout", 0.1, 0.9)

            self.model = LSTM(
                input_size=self.data_shape[1],  # PRIOR
                output_size=2,  # PRIOR
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                fc_dropout=fc_dropout,
            )
            wandb_config = {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "bidirectional": bidirectional,
                "fc_dropout": fc_dropout,
            }

        elif self._model == "transformer":
            hidden_size = self._trial.suggest_int(
                "transformer.hidden_size", 4, 128, log=True
            )
            num_heads = self._trial.suggest_int("transformer.num_heads", 1, 4)
            num_layers = self._trial.suggest_int("transformer.num_layers", 1, 4)
            fc_dropout = self._trial.suggest_uniform("transformer.fc_dropout", 0.1, 0.9)

            self.model = Transformer(
                input_size=self.data_shape[1],  # PRIOR
                output_size=2,  # PRIOR
                hidden_size=hidden_size * num_heads,
                num_layers=num_layers,
                num_heads=num_heads,
                fc_dropout=fc_dropout,
            )
            wandb_config = {
                "hidden_size": hidden_size,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "fc_dropout": fc_dropout,
            }

        elif self._model == "milc":
            feature_size = 256
            no_downsample = True
            fMRI_twoD = False
            end_with_relu = False
            method = "sub-lstm"

            encoder = NatureOneCNN(
                self.data_shape[2],
                feature_size,
                no_downsample,
                fMRI_twoD,
                end_with_relu,
                method,
            )

            if torch.cuda.is_available():
                cudaID = str(torch.cuda.current_device())
                device = torch.device("cuda:" + cudaID)
                # device = torch.device("cuda:" + str(args.cuda_id))
            else:
                device = torch.device("cpu")

            lstm_size = 200
            lstm_layers = 1
            ID = 4
            gain = [0.05, 0.05, 0.05, 0.05, 0.05]
            current_gain = gain[ID]

            lstm_model = subjLSTM(
                device,
                feature_size,
                lstm_size,
                num_layers=lstm_layers,
                freeze_embeddings=True,
                gain=current_gain,
            )

            # path = ASSETS_ROOT.joinpath("pretrained_encoder/encoder.pt")
            # model_dict = torch.load(path, map_location=device)

            pre_training = "milc"
            exp = "UFPT"
            oldpath = ASSETS_ROOT.joinpath("pretrained_encoder")
            self.model = combinedModel(
                encoder,
                lstm_model,
                gain=current_gain,
                PT=pre_training,
                exp=exp,
                device=device,
                oldpath=oldpath,
            )

        else:
            raise NotImplementedError()

        self.criterion = nn.CrossEntropyLoss()

        if self._model == "milc":
            lr = 0.0003

            if exp in ["UFPT", "NPT"]:
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=lr, eps=1e-5
                )
            else:
                if pre_training in ["milc", "two-loss-milc"]:
                    self.optimizer = torch.optim.Adam(
                        list(self.model.decoder.parameters()),
                        lr=lr,
                        eps=1e-5,
                    )
                else:
                    self.optimizer = torch.optim.Adam(
                        list(self.model.decoder.parameters())
                        + list(self.model.attn.parameters())
                        + list(self.model.lstm.parameters()),
                        lr=lr,
                        eps=1e-5,
                    )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min"
            )
        else:
            lr = self._trial.suggest_float("adam.lr", 1e-5, 1e-3, log=True)
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
            )
            wandb_config["lr"] = lr

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

        wandb_config["num_epochs"] = self.num_epochs
        wandb_config["batch_size"] = self.batch_size
        self.wandb_logger.config.update(wandb_config)

    def run_dataset(self) -> None:
        all_scores, all_targets = [], []
        total_loss = 0.0
        self.model.train(self.is_train_dataset)

        with torch.set_grad_enabled(self.is_train_dataset):
            for self.dataset_batch_step, (data, target) in enumerate(tqdm(self.dataset)):
                self.optimizer.zero_grad()
                logits, _ = self.model(data)
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
            self._test_ds, batch_size=self.batch_size, num_workers=0, shuffle=False
        )

        f = open(f"{self.logdir}/k_{self.k}/{self._trial.number:04d}/model.storage.json")
        logpath = json.load(f)["storage"][0]["logpath"]
        checkpoint = torch.load(logpath, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint)
        self.model.train(False)
        self.model.zero_grad()

        all_scores, all_targets, all_raw_scores = [], [], []
        total_loss = 0.0

        with torch.set_grad_enabled(False):
            for self.dataset_batch_step, (data, target) in enumerate(tqdm(test_ds)):
                self.optimizer.zero_grad()
                logits, raw_logits = self.model(data)
                loss = self.criterion(logits, target)
                score = torch.softmax(logits, dim=-1)

                all_scores.append(score.cpu().detach().numpy())
                all_targets.append(target.cpu().detach().numpy())
                total_loss += loss.sum().item()

                # prepare raw scores (score[number_of_time_slices] for each subject)
                raw_scores = torch.softmax(raw_logits, dim=-1)
                all_raw_scores.append(raw_scores.cpu().detach().numpy())

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

        # raw scores for logistic regression
        all_targets = np.concatenate(all_targets, axis=0)
        all_raw_scores = np.concatenate(all_raw_scores, axis=0)
        all_mean_scores = np.mean(all_raw_scores, axis=1)
        print("all_targets.shape: ", all_targets.shape)
        print("all_raw_scores.shape: ", all_raw_scores.shape)
        print("all_mean_scores.shape: ", all_mean_scores.shape)

        np.savez(
            f"{self.logdir}/k_{self.k}/{self._trial.number:04d}/test_scores.npz",
            raw_scores=all_raw_scores,
            mean_scores=all_mean_scores,
            targets=all_targets,
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
        for k in range(self.n_splits):
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
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "mlp",
            "attention_mlp",
            "another_attention_mlp",
            "lstm",
            "transformer",
            "milc",
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
        model=args.model,
        dataset=args.ds,
        quantile=args.quantile,
        n_splits=args.num_splits,
        max_epochs=args.max_epochs,
        logdir=f"{LOGS_ROOT}/{UTCNOW}-tune-{args.model}-{args.ds}/",
    ).tune(n_trials=args.num_trials)
