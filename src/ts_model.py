# pylint: disable=C0115,C0103,C0116,R1725,R0913
"""Models for experiments and functions for setting them up"""
import os
import json
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from scipy.stats import randint, uniform, loguniform

from src.scripts.ts_dl_experiments import Experiment
from src.settings import LOGS_ROOT


def get_config(exp: Experiment):
    config = {}
    model_config = {}

    if exp.mode == "experiment":
        # find and load the best tuned model
        runs_files = []

        searched_dir = exp.project_name.split("-")
        searched_dir = "-".join(searched_dir[1:3])
        serached_dir = f"tune-{searched_dir}"
        if exp.project_prefix != exp.utcnow:
            serached_dir = f"{exp.project_prefix}-{serached_dir}"
        print(f"Searching trained model in {LOGS_ROOT}/*{serached_dir}")
        for logdir in os.listdir(LOGS_ROOT):
            if logdir.endswith(serached_dir):
                runs_files.append(os.path.join(LOGS_ROOT, logdir))

        # if multiple runs files found, choose the latest
        runs_file = sorted(runs_files)[-1]
        print(f"Using best model from {runs_file}")

        # get general config
        with open(f"{runs_file}/config.json", "r") as fp:
            config = json.load(fp)
        # the only thing needed from the general config is batch size
        batch_size = config["batch_size"]

        # get model config
        df = pd.read_csv(f"{runs_file}/runs.csv", delimiter=",", index_col=False)
        # pick hyperparams of a model with the highest test_score
        best_config_path = df.loc[df["test_score"].idxmax()].to_dict()
        best_config_path = best_config_path["config_path"]
        with open(best_config_path, "r") as fp:
            model_config = json.load(fp)

        print("Loaded model coonfig:")
        print(model_config)

    elif exp.mode == "tune":
        # add the link to the wandb run
        model_config["link"] = exp.wandb_logger.get_url()
        # set batch size
        batch_size = randint.rvs(4, min(32, int(exp.data_shape[0] / exp.n_splits) - 1))

        # pick random hyperparameters
        if exp.model in [
            "mlp",
            "wide_mlp",
            "deep_mlp",
            "attention_mlp",
            "new_attention_mlp",
        ]:
            model_config["hidden_size"] = randint.rvs(32, 256)
            model_config["num_layers"] = randint.rvs(0, 4)
            model_config["dropout"] = uniform.rvs(0.1, 0.9)

            if exp.model == "wide_mlp":
                model_config["hidden_size"] = randint.rvs(256, 1024)
            elif exp.model == "deep_mlp":
                model_config["num_layers"] = randint.rvs(4, 20)

            if exp.model == "attention_mlp":
                model_config["time_length"] = exp.data_shape[1]
            elif exp.model == "new_attention_mlp":
                model_config["time_length"] = exp.data_shape[1]
                model_config["attention_size"] = randint.rvs(32, 256)

            model_config["input_size"] = exp.data_shape[2]
            model_config["output_size"] = exp.n_classes

        elif exp.model in ["lstm", "noah_lstm"]:
            model_config["hidden_size"] = randint.rvs(32, 256)
            model_config["num_layers"] = randint.rvs(1, 4)
            model_config["bidirectional"] = bool(randint.rvs(0, 1))
            model_config["fc_dropout"] = uniform.rvs(0.1, 0.9)

            model_config["input_size"] = exp.data_shape[2]
            model_config["output_size"] = exp.n_classes
        elif exp.model in ["transformer", "mean_transformer"]:
            model_config["head_hidden_size"] = randint.rvs(4, 128)
            model_config["num_heads"] = randint.rvs(1, 4)
            model_config["num_layers"] = randint.rvs(1, 4)
            model_config["fc_dropout"] = uniform.rvs(0.1, 0.9)

            model_config["input_size"] = exp.data_shape[2]
            model_config["output_size"] = exp.n_classes

        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    return int(batch_size), model_config


def get_model(model, model_config):
    if model in ["mlp", "wide_mlp", "deep_mlp"]:
        return MLP(model_config)
    elif model == "attention_mlp":
        return AttentionMLP(model_config)
    elif model == "new_attention_mlp":
        return NewAttentionMLP(model_config)

    elif model == "lstm":
        return LSTM(model_config)
    elif model in "noah_lstm":
        return NoahLSTM(model_config)

    elif model == "transformer":
        return Transformer(model_config)
    elif model == "mean_transformer":
        return MeanTransformer(model_config)

    raise NotImplementedError()


def get_criterion(model):
    return nn.CrossEntropyLoss()


def get_optimizer(exp: Experiment, model_config):
    if exp.mode == "tune":
        model_config["lr"] = loguniform.rvs(1e-5, 1e-3)

    optimizer = optim.Adam(
        exp._model.parameters(),
        lr=float(model_config["lr"]),
    )

    return model_config["lr"], optimizer


class ResidualBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor):
        return self.block(x) + x


class MLP(nn.Module):
    def __init__(self, model_config):
        super(MLP, self).__init__()

        input_size = int(model_config["input_size"])
        output_size = int(model_config["output_size"])
        dropout = float(model_config["dropout"])
        hidden_size = int(model_config["hidden_size"])
        num_layers = int(model_config["num_layers"])

        layers = [
            nn.LayerNorm(input_size),
            nn.Dropout(p=dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            layers.append(
                ResidualBlock(
                    nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                    )
                )
            )
        layers.append(
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, output_size),
            )
        )

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        bs, ln, fs = x.shape
        # bs:  batch size
        # ln:  length in time, 295
        # fs: number of channels, 53
        fc_output = self.fc(x.view(-1, fs))
        fc_output = fc_output.view(bs, ln, -1)
        logits = fc_output.mean(1)
        return logits


class AttentionMLP(nn.Module):
    def __init__(self, model_config):
        super(AttentionMLP, self).__init__()

        input_size = int(model_config["input_size"])
        time_length = int(model_config["time_length"])
        output_size = int(model_config["output_size"])
        dropout = float(model_config["dropout"])
        hidden_size = int(model_config["hidden_size"])
        num_layers = int(model_config["num_layers"])

        layers = [
            nn.LayerNorm(input_size),
            nn.Dropout(p=dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            layers.append(
                ResidualBlock(
                    nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                    )
                )
            )
        layers.append(
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, output_size),
            )
        )

        self.fc = nn.Sequential(*layers)

        self.attn = nn.Sequential(
            nn.Linear(time_length + 1, 3 * hidden_size),
            nn.ReLU(),
            nn.Linear(3 * hidden_size, time_length),
        )

    def get_attention(self, outputs):
        # calculate mean over time
        outputs_mean = outputs.mean(1).unsqueeze(1)

        # add output's mean to the end of each output time dimension
        # as a reference for attention layer
        # and pass it to attention layer
        weights_list = []
        for output, output_mean in zip(outputs, outputs_mean):
            result = torch.cat((output, output_mean)).swapaxes(0, 1)
            result_tensor = self.attn(result)
            weights_list.append(result_tensor)

        weights = torch.stack(weights_list)
        # normalize weights and swap axes to the output shape
        normalized_weights = F.softmax(weights, dim=2).swapaxes(1, 2)
        return normalized_weights

    def forward(self, x):
        bs, ln, fs = x.shape
        fc_output = self.fc(x.reshape(-1, fs))
        fc_output = fc_output.reshape(bs, ln, -1)

        # get weights form attention layer
        normalized_weights = self.get_attention(fc_output)

        # sum outputs weight-wise
        logits = torch.einsum("ijk,ijk->ik", fc_output, normalized_weights)

        return logits


class NewAttentionMLP(nn.Module):
    def __init__(self, model_config):
        super(NewAttentionMLP, self).__init__()

        input_size = int(model_config["input_size"])
        time_length = int(model_config["time_length"])
        output_size = int(model_config["output_size"])
        dropout = float(model_config["dropout"])
        hidden_size = int(model_config["hidden_size"])
        attention_size = int(model_config["attention_size"])
        num_layers = int(model_config["num_layers"])

        layers = [
            nn.LayerNorm(input_size),
            nn.Dropout(p=dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            layers.append(
                ResidualBlock(
                    nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                    )
                )
            )
        layers.append(
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, output_size),
            )
        )

        self.fc = nn.Sequential(*layers)

        self.attn = nn.Sequential(
            nn.LayerNorm(time_length + 1),
            nn.Dropout(p=dropout),
            nn.Linear(time_length + 1, attention_size),
            nn.ReLU(),
            nn.LayerNorm(attention_size),
            nn.Dropout(p=dropout),
            nn.Linear(attention_size, time_length),
        )

    def get_attention(self, outputs):
        # calculate mean over time
        outputs_mean = outputs.mean(1).unsqueeze(1)

        # add output's mean to the end of each output time dimension
        # as a reference for attention layer
        # and pass it to attention layer
        weights_list = []
        for output, output_mean in zip(outputs, outputs_mean):
            result = torch.cat((output, output_mean)).swapaxes(0, 1)
            result_tensor = self.attn(result)
            weights_list.append(result_tensor)

        weights = torch.stack(weights_list)
        # normalize weights and swap axes to the output shape
        normalized_weights = F.softmax(weights, dim=2).swapaxes(1, 2)
        return normalized_weights

    def forward(self, x):
        bs, ln, fs = x.shape
        fc_output = self.fc(x.reshape(-1, fs))
        fc_output = fc_output.reshape(bs, ln, -1)

        # get weights form attention layer
        normalized_weights = self.get_attention(fc_output)

        # sum outputs weight-wise
        logits = torch.einsum("ijk,ijk->ik", fc_output, normalized_weights)

        return logits


class LSTM(nn.Module):
    def __init__(self, model_config):
        super(LSTM, self).__init__()

        input_size = int(model_config["input_size"])
        output_size = int(model_config["output_size"])
        fc_dropout = float(model_config["fc_dropout"])
        hidden_size = int(model_config["hidden_size"])
        bidirectional = bool(model_config["bidirectional"])
        num_layers = int(model_config["num_layers"])

        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=fc_dropout),
            nn.Linear(2 * hidden_size if bidirectional else hidden_size, output_size),
        )

    def forward(self, x):
        lstm_output, _ = self.lstm(x)

        if self.bidirectional:
            out_forward = lstm_output[:, -1, : self.hidden_size]
            out_reverse = lstm_output[:, 0, self.hidden_size :]
            lstm_output = torch.cat((out_forward, out_reverse), 1)
        else:
            lstm_output = lstm_output[:, -1, :]

        fc_output = self.fc(lstm_output)
        return fc_output


class NoahLSTM(nn.Module):
    def __init__(self, model_config):
        super(NoahLSTM, self).__init__()

        input_size = int(model_config["input_size"])
        output_size = int(model_config["output_size"])
        fc_dropout = float(model_config["fc_dropout"])
        hidden_size = int(model_config["hidden_size"])
        bidirectional = bool(model_config["bidirectional"])
        num_layers = int(model_config["num_layers"])

        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=fc_dropout),
            nn.Linear(2 * hidden_size if bidirectional else hidden_size, output_size),
        )

    def forward(self, x):
        lstm_output, _ = self.lstm(x)

        lstm_output = torch.mean(lstm_output, dim=1)
        logits = self.fc(lstm_output)

        return logits


class Transformer(nn.Module):
    def __init__(self, model_config):
        super(Transformer, self).__init__()

        input_size = int(model_config["input_size"])
        output_size = int(model_config["output_size"])
        fc_dropout = float(model_config["fc_dropout"])
        head_hidden_size = int(model_config["head_hidden_size"])
        num_layers = int(model_config["num_layers"])
        num_heads = int(model_config["num_heads"])

        hidden_size = head_hidden_size * num_heads

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True
        )
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            transformer_encoder,
        ]
        self.transformer = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Dropout(p=fc_dropout), nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # x.shape = [Batch_size; time_len; n_channels]
        fc_output = self.transformer(x)
        fc_output = fc_output[:, -1, :]
        fc_output = self.fc(fc_output)
        return fc_output


class MeanTransformer(nn.Module):
    def __init__(self, model_config):
        super(MeanTransformer, self).__init__()

        input_size = int(model_config["input_size"])
        output_size = int(model_config["output_size"])
        fc_dropout = float(model_config["fc_dropout"])
        head_hidden_size = int(model_config["head_hidden_size"])
        num_layers = int(model_config["num_layers"])
        num_heads = int(model_config["num_heads"])

        hidden_size = head_hidden_size * num_heads

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True
        )
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            transformer_encoder,
        ]
        self.transformer = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Dropout(p=fc_dropout), nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # x.shape = [Batch_size; time_len; n_channels]
        fc_output = self.transformer(x)
        fc_output = torch.mean(fc_output, 1)
        fc_output = self.fc(fc_output)
        return fc_output
