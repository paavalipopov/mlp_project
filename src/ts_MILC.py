# pylint: disable=C0115,C0103,C0116,R1725,R0913
""" wholeMILC model """

import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.utils.rnn as tn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NatureOneCNN(nn.Module):
    def init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module

    def __init__(
        self,
        input_channels,
        feature_size,
        no_downsample,
        fMRI_twoD,
        end_with_relu,
        method,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.feature_size = feature_size
        self.hidden_size = feature_size
        self.downsample = not no_downsample
        self.twoD = fMRI_twoD
        self.end_with_relu = end_with_relu
        self.method = method
        init_ = lambda m: self.init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )
        self.flatten = Flatten()

        if self.downsample:
            self.final_conv_size = 32 * 7 * 7
            self.final_conv_shape = (32, 7, 7)
            self.main = nn.Sequential(
                init_(nn.Conv2d(input_channels, 32, 8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 32, 3, stride=1)),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(self.final_conv_size, self.feature_size)),
                # nn.ReLU()
            )

        else:
            # problem with OASIS input is prolly here
            self.final_conv_size = 200 * 12
            self.final_conv_shape = (200, 12)
            print("feature_size = ", self.feature_size)
            self.main = nn.Sequential(
                init_(nn.Conv1d(input_channels, 64, 4, stride=1)),  # 0
                nn.ReLU(),
                init_(nn.Conv1d(64, 128, 4, stride=1)),
                nn.ReLU(),
                init_(nn.Conv1d(128, 200, 3, stride=1)),
                nn.ReLU(),
                Flatten(),  # 6
                init_(nn.Linear(self.final_conv_size, self.feature_size)),
                init_(nn.Conv1d(200, 128, 3, stride=1)),
                nn.ReLU(),
                # nn.ReLU()
            )

    def forward(self, inputs, fmaps=False, five=False):
        # print("inputs = ", inputs.shape)
        # inputs =  torch.Size([7, 53, 20]) for COBRE
        # inputs =  torch.Size([6, 53, 26]) for OASIS
        f5 = self.main[:6](inputs)
        # print("f5 = ", f5.shape)
        # f5 =  torch.Size([7, 200, 12]) for COBRE
        # f5 =  torch.Size([6, 200, 18]) for OASIS
        out = self.main[6:8](f5)
        f5 = self.main[8:](f5)

        if self.end_with_relu:
            assert self.method != "vae", "can't end with relu and use vae!"
            out = F.relu(out)
        if five:
            return f5.permute(0, 2, 1)
        if fmaps:
            return {
                "f5": f5.permute(0, 2, 1),
                # 'f7': f7.permute(0, 2, 1),
                "out": out,
            }
        return out


class subjLSTM(nn.Module):
    """Bidirectional LSTM for classifying subjects."""

    def __init__(
        self,
        device,
        embedding_dim,
        hidden_dim,
        num_layers=1,
        freeze_embeddings=True,
        gain=1,
    ):

        super(subjLSTM, self).__init__()
        self.gain = gain
        self.device = device
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.freeze_embeddings = freeze_embeddings

        self.lstm = nn.LSTM(
            self.embedding_dim,
            self.hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
        )

        # The linear layer that maps from hidden state space to tag space
        # self.decoder = nn.Sequential(
        #     nn.Linear(self.hidden_dim, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, 2)
        #
        # )
        self.init_weight()

    def init_hidden(self, batch_size, device):
        h0 = Variable(torch.zeros(2, batch_size, self.hidden_dim // 2, device=device))
        c0 = Variable(torch.zeros(2, batch_size, self.hidden_dim // 2, device=device))
        return (h0, c0)

    def init_weight(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        # for name, param in self.decoder.named_parameters():
        #     if 'weight' in name:
        #         nn.init.xavier_normal_(param, gain=self.gain)

    def forward(self, inputs, mode="train"):

        packed = tn.pack_sequence(inputs, enforce_sorted=False)

        self.hidden = self.init_hidden(len(inputs), packed.data.device)
        self.lstm.flatten_parameters()
        if mode == "eval" or mode == "test":
            with torch.no_grad():
                packed_out, self.hidden = self.lstm(packed, self.hidden)
        else:
            packed_out, self.hidden = self.lstm(packed, self.hidden)

        # output, lens = tn.pad_packed_sequence(packed_out, batch_first=True, total_length=total_length)
        outputs, lens = tn.pad_packed_sequence(packed_out, batch_first=True)
        # outputs = [line[:l] for line, l in zip(outputs, lens)]
        # outputs = [self.decoder(torch.cat((x[0, self.hidden_dim // 2:],
        #                                    x[-1, :self.hidden_dim // 2]), 0)) for x in outputs]
        return outputs


class combinedModel(nn.Module):
    """Bidirectional LSTM for classifying subjects."""

    def __init__(
        self,
        encoder,
        lstm,
        gain=0.1,
        PT="",
        exp="UFPT",
        device="cuda",
        oldpath="",
        complete_arc=False,
    ):

        super().__init__()
        self.encoder = encoder
        self.lstm = lstm
        self.gain = gain
        self.PT = PT
        self.exp = exp
        self.device = device
        self.oldpath = oldpath
        self.complete_arc = complete_arc
        self.attn = nn.Sequential(
            nn.Linear(2 * self.lstm.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.lstm.hidden_dim, self.lstm.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.lstm.hidden_dim, 2),
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(self.encoder.feature_size, self.lstm.hidden_dim),
        ).to(device)

        self.init_weight()
        if self.complete_arc == False:
            self.loadModels()

    def init_weight(self):
        for name, param in self.decoder.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.attn.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param, gain=self.gain)

    def loadModels(self):
        if self.PT in ["milc", "variable-attention", "two-loss-milc"]:
            if self.exp in ["UFPT", "FPT"]:
                print("in ufpt and fpt")
                if not self.complete_arc:
                    model_dict = torch.load(
                        os.path.join(self.oldpath, "encoder" + ".pt"),
                        map_location=self.device,
                    )
                    self.encoder.load_state_dict(model_dict)

                    model_dict = torch.load(
                        os.path.join(self.oldpath, "lstm" + ".pt"),
                        map_location=self.device,
                    )
                    self.lstm.load_state_dict(model_dict)
                    # self.model.lstm.to(self.device)

                    model_dict = torch.load(
                        os.path.join(self.oldpath, "attn" + ".pt"),
                        map_location=self.device,
                    )
                    self.attn.load_state_dict(model_dict)
                    # self.model.attn.to(self.device)
                else:
                    model_dict = torch.load(
                        os.path.join(self.oldpath, "best_full" + ".pth"),
                        map_location=self.device,
                    )
                    self.load_state_dict(model_dict)

    def get_attention(self, outputs):
        # print('in attention')
        weights_list = []
        for X in outputs:
            # t=time.time()
            # print("Raw output shape: ", X.shape)
            result = [torch.cat((X[i], X[-1]), 0) for i in range(X.shape[0])]
            # print("Attention result length: ", len(result))
            result = torch.stack(result)
            # print("Attention result shape after stack: ", result.shape)
            result_tensor = self.attn(result)
            # print("result_tensor shape: ", result_tensor.shape)
            weights_list.append(result_tensor)

        weights = torch.stack(weights_list)
        # print("weights shape: ", weights.shape)

        weights = weights.squeeze(2)
        # print("weights shape after squeeze: ", weights.shape)

        normalized_weights = F.softmax(weights, dim=1)
        # print("normalized_weights shape: ", normalized_weights.shape)
        # print(
        #     "normalized_weights shape after unsqueeze: ",
        #     normalized_weights.unsqueeze(1).shape,
        # )
        # print("outputs shape: ", outputs.shape)
        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)
        # print("attn_applied shape: ", attn_applied.shape)

        attn_applied = attn_applied.squeeze(1)
        # print("attn_applied shape after squeeze: ", attn_applied.shape)
        logits = self.decoder(attn_applied)
        # print("attention decoder ", time.time() - t)
        return logits

    def forward(self, sx, mode="train"):

        inputs = [self.encoder(x, fmaps=False) for x in sx]
        # print("Inputs len: ", len(inputs))
        # print("Inputs shape: ", inputs[0].shape)
        outputs = self.lstm(inputs, mode)
        # print("Output shape: ", outputs.shape)
        logits = self.get_attention(outputs)
        # print("Logits shape: ", logits.shape)

        return logits
