#pylint: disable=C0115,C0103,C0116,R1725,R0913
"""
Models for experiments
"""
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(
        self,
        input_len: int,
        fc_dropout: float = 0.5,
        hidden_size: int = 128,
        bidirectional: bool = False,
        **kwargs,
    ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(hidden_size=hidden_size, bidirectional=bidirectional, **kwargs)
        lstm_out = 2 * hidden_size * input_len if bidirectional else hidden_size * input_len
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(lstm_out),
            nn.Dropout(p=fc_dropout),
            nn.Linear(lstm_out, 2),
        )

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        fc_output = self.fc(lstm_output)
        return fc_output

class Transformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        input_len: int,
        fc_dropout: float = 0.5,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_heads: int = 8,
    ):
        super(Transformer, self).__init__()
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
            nn.Flatten(),
            nn.LayerNorm(input_len * hidden_size),
            nn.Dropout(p=fc_dropout),
            nn.Linear(input_len * hidden_size, 2),
        )

    def forward(self, x):
        fc_output = self.transformer(x)
        fc_output = self.fc(fc_output)
        return fc_output


class ResidualBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor):
        return self.block(x) + x


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout: float = 0.5,
        hidden_size: int = 128,
        num_layers: int = 0,
    ):
        super(MLP, self).__init__()
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
        fc_output = self.fc(x.reshape(-1, fs))
        fc_output = fc_output.reshape(bs, ln, -1).mean(1)  # .squeeze(1)
        return fc_output
