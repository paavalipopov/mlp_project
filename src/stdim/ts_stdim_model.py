import os
import json
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from scipy.stats import randint, uniform, loguniform

from src.stdim.ts_stdim_experiments import STDIM_Experiment
from src.settings import LOGS_ROOT

from src.stdim.stdim_encoder_trainer import EncoderTrainer


def get_config(exp: STDIM_Experiment, data_shape):
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

        # get model config: probe, encoder and dataset params
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
        batch_size = randint.rvs(4, min(32, int(data_shape[0] / exp.n_splits) - 1))
        # batch_size = 32

        # pick random model hyperparameters

        # params of dataset reshapes
        model_config["datashape"] = {}
        # data_shape is [n_features; n_channels; time_len]
        # window_size=9 is minimal for the given kernels preset of NatureOneCNN
        model_config["datashape"]["window_size"] = randint.rvs(9, data_shape[2] // 5)
        # window shift determines how much the windows overlap
        model_config["datashape"]["window_shift"] = randint.rvs(
            1, model_config["datashape"]["window_size"]
        )

        # params of encoder
        model_config["encoder"] = {}
        # # it is 256 in MILC paper
        model_config["encoder"]["feature_size"] = randint.rvs(32, 256)
        # model_config["encoder"]["feature_size"] = 256
        # # it is 3e-4 in MILC paper
        model_config["encoder"]["lr"] = loguniform.rvs(1e-5, 1e-3)
        # model_config["encoder"]["lr"] = 3e-4
        # # data_shape is [n_features; n_channels; time_len]
        model_config["encoder"]["input_channels"] = data_shape[1]
        # convolution layers output size (depends on the windows size)
        model_config["encoder"]["conv_output_size"] = NatureOneCNN.get_conv_output_size(
            model_config["datashape"]["window_size"]
        )
        assert model_config["encoder"]["conv_output_size"] >= 1

        # params of probe
        model_config["probe"] = {}
        model_config["probe"]["input_size"] = model_config["encoder"]["feature_size"]
        model_config["probe"]["output_size"] = exp.n_classes
        model_config["probe"]["lr"] = loguniform.rvs(1e-5, 1e-3)
        # model_config["probe"]["lr"] = 3e-4

        print("Tuning model coonfig:")
        print(model_config)

    else:
        raise NotImplementedError()

    return int(batch_size), model_config


def get_encoder(exp: STDIM_Experiment, encoder_config: dict):
    if exp.pretraining == "NPT":
        # train encoder
        encoder = EncoderTrainer(
            encoder_config=encoder_config,
            dataset=exp.encoder_dataset,
            logpath=exp.config["runpath"],
            wandb_logger=exp.wandb_logger,
            batch_size=exp.batch_size,
        ).train_encoder()
        return encoder
    elif exp.pretraining == "FPT":
        raise NotImplementedError()
    elif exp.pretraining == "UFPT":
        raise NotImplementedError()

    raise NotImplementedError()


def get_probe(exp: STDIM_Experiment, probe_config: dict):
    return Probe(probe_config)


def get_criterion(exp: STDIM_Experiment):
    return nn.CrossEntropyLoss()


def get_optimizer(exp: STDIM_Experiment, encoder_config, probe_config):
    optimizer = optim.Adam(
        exp.probe.parameters(),
        lr=float(probe_config["lr"]),
    )
    # optimizer = optim.Adam(
    #     list(exp.probe.parameters()) + list(self.encoder.parameters()),
    #     lr=float(probe_config["lr"]),
    # )

    return optimizer


class Probe(nn.Module):
    def __init__(self, probe_config):
        super().__init__()
        self.model = nn.Linear(
            in_features=int(probe_config["input_size"]),
            out_features=int(probe_config["output_size"]),
        )

    def forward(self, x):
        return self.model(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NatureOneCNN(nn.Module):
    def init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module

    def __init__(self, encoder_config):
        super().__init__()

        self.feature_size = int(encoder_config["feature_size"])
        self.input_size = int(encoder_config["input_channels"])
        self.conv_output_size = int(encoder_config["conv_output_size"])

        init_ = lambda m: self.init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )
        self.flatten = Flatten()

        final_conv_size = 200 * self.conv_output_size
        # final_conv_shape = (200, self.conv_output_size)
        self.main = nn.Sequential(
            init_(nn.Conv1d(self.input_size, 64, 4, stride=1)),
            nn.ReLU(),
            init_(nn.Conv1d(64, 128, 4, stride=1)),
            nn.ReLU(),
            init_(nn.Conv1d(128, 200, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(final_conv_size, self.feature_size)),
        )

    @property
    def local_layer_depth(self):
        return self.main[4].out_channels

    @staticmethod
    def get_conv_output_size(conv_input_size):
        # https://www.baeldung.com/cs/convolutional-layer-size
        conv_output_size = conv_input_size - 4 + 1  # 1st layer
        conv_output_size = conv_output_size - 4 + 1  # 2nd layer
        conv_output_size = conv_output_size - 3 + 1  # 3rd layer

        return conv_output_size

    def forward(self, inputs, fmaps=False, five=False):
        f5 = self.main[:6](inputs)
        out = self.main[6:](f5)

        if five:
            return f5.permute(0, 2, 1)
        if fmaps:
            return {
                "f5": f5.permute(0, 2, 1),
                "out": out,
            }
        return out
