# pylint: disable=C0115,C0103,C0116,R1725,R0913
"""Models for experiments"""
import torch
from torch import nn
import torch.nn.functional as F


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
        # bs:  batch size
        # ln:  length in time, 295
        # fs: number of channels, 53
        fc_output = self.fc(x.view(-1, fs))
        fc_output = fc_output.view(bs, ln, -1)
        logits = fc_output.mean(1)
        return logits


class No_Res_MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout: float = 0.5,
        hidden_size: int = 128,
        num_layers: int = 0,
    ):
        super(No_Res_MLP, self).__init__()
        layers = [
            nn.LayerNorm(input_size),
            nn.Dropout(p=dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
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


class No_Ens_MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout: float = 0.5,
        hidden_size: int = 128,
        num_layers: int = 0,
    ):
        super(No_Ens_MLP, self).__init__()
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
        fc_output = self.fc(x.view(bs, -1))
        return fc_output


class Transposed_MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout: float = 0.5,
        hidden_size: int = 128,
        num_layers: int = 0,
    ):
        super(Transposed_MLP, self).__init__()
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
        fc_input = x.swapaxes(1, 2)
        # bs:  batch size
        # ln:  length in time, 295
        # fs: number of channels, 53
        fc_output = self.fc(fc_input.reshape(-1, ln))
        fc_output = fc_output.view(bs, fs, -1)
        logits = fc_output.mean(1)
        return logits


class LSTM(nn.Module):
    def __init__(
        self,
        output_size: int,
        fc_dropout: float = 0.5,
        hidden_size: int = 128,
        bidirectional: bool = False,
        **kwargs,
    ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            hidden_size=hidden_size, bidirectional=bidirectional, **kwargs
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


class Transformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
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
            nn.Dropout(p=fc_dropout), nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        fc_output = self.transformer(x)
        fc_output = fc_output[:, -1, :]
        fc_output = self.fc(fc_output)
        return fc_output


class AttentionMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        time_length: int,
        output_size: int,
        dropout: float = 0.5,
        hidden_size: int = 128,
        num_layers: int = 0,
    ):
        super(AttentionMLP, self).__init__()
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


class AnotherAttentionMLP(nn.Module):
    # not working
    def __init__(
        self,
        input_size: int,
        time_length: int,
        output_size: int,
        dropout: float = 0.5,
        hidden_size: int = 128,
        num_layers: int = 0,
    ):
        super(AnotherAttentionMLP, self).__init__()
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
            nn.Linear(time_length, 3 * hidden_size),
            nn.ReLU(),
            nn.Linear(3 * hidden_size, time_length),
        )

    def get_attention(self, outputs):
        # calculate mean over time
        outputs_mean = outputs.mean(1)

        # add output's mean to the end of each output time dimension
        # as a reference for attention layer
        # and pass it to attention layer
        weights_list = []
        for output, output_mean in zip(outputs, outputs_mean):
            # print("output shape: ", output.shape)
            # print("output_mean shape: ", output_mean.shape)
            result = torch.zeros(output.shape)
            for i in range(output.shape[0]):
                result[i] = torch.add(output[i], output_mean)
            # print("result shape: ", result.shape)
            result_tensor = self.attn(result.swapaxes(0, 1))
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


class MyLogisticRegression(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(MyLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        bs, ln, fs = x.shape
        # bs:  batch size
        # ln:  length in time, 295
        # fs: number of channels, 53

        logits = self.linear(x.view(bs, -1)).squeeze()
        return logits


class EnsembleLogisticRegression(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(EnsembleLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        bs, ln, fs = x.shape
        # bs:  batch size
        # ln:  length in time, 295
        # fs: number of channels, 53

        output = self.linear(x.view(-1, fs))
        output = output.view(bs, ln, -1)
        logits = output.mean(1).squeeze()
        return logits


class AnotherEnsembleLogisticRegression(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(AnotherEnsembleLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        bs, ln, fs = x.shape
        # bs:  batch size
        # ln:  length in time, 295
        # fs: number of channels, 53

        output = self.linear(x.view(-1, fs))
        output = output.view(bs, ln, -1)
        scores = torch.sigmoid(output)
        scores = scores.mean(1).squeeze()
        return scores


class MySVM(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(MySVM, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        bs, ln, fs = x.shape
        # bs:  batch size
        # ln:  length in time, 295
        # fs: number of channels, 53

        logits = self.linear(x.view(bs, -1)).squeeze()
        return logits


class EnsembleSVM(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(EnsembleSVM, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        bs, ln, fs = x.shape
        # bs:  batch size
        # ln:  length in time, 295
        # fs: number of channels, 53

        output = self.linear(x.view(-1, fs))
        output = output.view(bs, ln, -1)
        logits = output.mean(1).squeeze()
        return logits
