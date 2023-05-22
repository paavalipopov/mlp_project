import torch
from torch import nn
from random import uniform, randint

from omegaconf import OmegaConf

def test1():
    print("No args")
def test2(t):
    print("args", t)

def get_model(cfg):
    return MLP(cfg)

def default_HPs(cfg):
    # TODO: find decent default HPs
    cfg = {
        "dropout": 0.8,
        "hidden_size": 150,
        "num_layers": 2,
        "lr": 5e-4,

        "input_size": cfg.dataset.data_info.data_shape.main[2],
        "output_size": cfg.dataset.data_info.n_classes,
    }
    return OmegaConf.create(cfg)

def random_HPs(cfg):
    cfg = {
        "dropout": uniform(0.1, 0.9),
        "hidden_size": randint(32, 256),
        "num_layers": randint(0, 4),
        "lr": 10**uniform(-5, -3),

        "input_size": cfg.dataset.data_info.data_shape.main[2],
        "output_size": cfg.dataset.data_info.n_classes,
    }
    return OmegaConf.create(cfg)

class ResidualBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor):
        return self.block(x) + x

class MLP(nn.Module):
    def __init__(self, model_cfg):
        super(MLP, self).__init__()

        input_size = model_cfg.input_size
        output_size = model_cfg.output_size
        dropout = model_cfg.dropout
        hidden_size = model_cfg.hidden_size
        num_layers = model_cfg.num_layers

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
        bs, tl, fs = x.shape # [batch_size, time_length, input_feature_size]

        fc_output = self.fc(x.view(-1, fs))
        fc_output = fc_output.view(bs, tl, -1)

        logits = fc_output.mean(1)
        return logits