import os
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import torch

from captum.attr import visualization as viz
from captum.attr import (
    Saliency,
    IntegratedGradients,
    NoiseTunnel,
)

from src.settings import LOGS_ROOT, ASSETS_ROOT, UTCNOW
from src.data_load import load_ABIDE1, load_OASIS, load_FBIRN
from src.models import LSTM, Transformer, MLP, AttentionMLP

sns.set_theme(style="whitegrid", font_scale=2, rc={"figure.figsize": (18, 9)})


class Introspection:
    def __init__(
        self, dataset_name: str, model_name: str, image_path: str, model_path: str
    ) -> None:
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.image_path = image_path
        self.model_path = model_path

    def initialize_model(self, model) -> None:
        # model = LSTM(
        #     input_size=53,  # PRIOR
        #     input_len=156,  # PRIOR
        #     hidden_size=52,
        #     num_layers=3,
        #     batch_first=True,
        #     bidirectional=False,
        #     fc_dropout=0.2626756675371412,
        # )
        # checkpoint = torch.load(
        #     self.model_path, map_location=lambda storage, loc: storage
        # )
        # # print(checkpoint)
        # model.load_state_dict(checkpoint)
        # self.model = model.eval()
        self.model = model

    def introspection(self, i, feature, introspection_methods: set):
        feature = feature.astype(np.float32)
        feature = torch.tensor(feature).unsqueeze(0)
        feature.requires_grad = True

        cutoff = (feature.shape[1] * feature.shape[2]) // 20  # 5%
        time_range = feature.shape[1]

        for method in introspection_methods:
            if method == "saliency":
                saliency = Saliency(self.model)
                self.model.zero_grad()
                grads0 = saliency.attribute(feature, target=0)
                self.model.zero_grad()
                grads1 = saliency.attribute(feature, target=1)
            elif method == "ig":
                ig = IntegratedGradients(self.model)
                self.model.zero_grad()
                grads0, _ = ig.attribute(
                    inputs=feature,
                    target=0,
                    baselines=torch.zeros_like(feature),
                    return_convergence_delta=True,
                )
                self.model.zero_grad()
                grads1, _ = ig.attribute(
                    inputs=feature,
                    target=1,
                    baselines=torch.zeros_like(feature),
                    return_convergence_delta=True,
                )
            elif method == "ignt":
                ig = IntegratedGradients(self.model)
                nt = NoiseTunnel(ig)
                self.model.zero_grad()
                grads0, _ = nt.attribute(
                    inputs=feature,
                    target=0,
                    baselines=torch.zeros_like(feature),
                    return_convergence_delta=True,
                    nt_type="smoothgrad_sq",
                    nt_samples=5,
                    stdevs=0.2,
                )
                self.model.zero_grad()
                grads1, _ = nt.attribute(
                    inputs=feature,
                    target=1,
                    baselines=torch.zeros_like(feature),
                    return_convergence_delta=True,
                    nt_type="smoothgrad_sq",
                    nt_samples=5,
                    stdevs=0.2,
                )
            else:
                print(f"Method '{method}' is undefined")
                return

            fig, axs = plt.subplots(1, 1, figsize=(21, 9))
            # transpose to [num_features; time_len; 1]
            _ = viz.visualize_image_attr(
                np.transpose(grads0.cpu().detach().numpy(), (2, 1, 0)),
                np.transpose(feature.cpu().detach().numpy(), (2, 1, 0)),
                method="heat_map",
                cmap="inferno",
                show_colorbar=False,
                plt_fig_axis=(fig, axs),
                use_pyplot=False,
            )
            plt.savefig(
                self.image_path.joinpath(f"{method}/colormap/{i:04d}.0.png"),
                format="png",
                dpi=300,
            )
            plt.close()

            fig, axs = plt.subplots(1, 1, figsize=(21, 9))
            _ = viz.visualize_image_attr(
                np.transpose(grads1.cpu().detach().numpy(), (2, 1, 0)),
                np.transpose(feature.cpu().detach().numpy(), (2, 1, 0)),
                method="heat_map",
                cmap="inferno",
                show_colorbar=False,
                plt_fig_axis=(fig, axs),
                use_pyplot=False,
            )
            plt.savefig(
                self.image_path.joinpath(f"{method}/colormap/{i:04d}.1.png"),
                format="png",
                dpi=300,
            )
            plt.close()

            # bar charts
            threshold0 = np.sort(grads0.detach().numpy().ravel())[
                -cutoff
            ]  # get the nth largest value
            idx = grads0 < threshold0
            grads0[idx] = 0

            threshold1 = np.sort(grads1.detach().numpy().ravel())[
                -cutoff
            ]  # get the nth largest value
            idx = grads1 < threshold1
            grads1[idx] = 0

            plt.bar(
                range(time_range),
                np.sum(grads0.cpu().detach().numpy(), axis=(0, 2)),
                align="center",
                color="blue",
            )
            plt.xlim([0, time_range])
            plt.grid(False)
            plt.axis("off")
            plt.savefig(
                self.image_path.joinpath(f"{method}/barchart/{i:04d}.0.png"),
                format="png",
                dpi=300,
            )
            plt.close()

            plt.bar(
                range(time_range),
                np.sum(grads1.cpu().detach().numpy(), axis=(0, 2)),
                align="center",
                color="blue",
            )
            plt.xlim([0, time_range])
            plt.grid(False)
            plt.axis("off")
            plt.savefig(
                self.image_path.joinpath(f"{method}/barchart/{i:04d}.1.png"),
                format="png",
                dpi=300,
            )
            plt.close()

    def run_introspection(self, introspection_methods: set):
        if self.dataset_name == "oasis":
            features, _ = load_OASIS()
        elif self.dataset_name == "abide":
            features, _ = load_ABIDE1()
        elif self.dataset_name == "fbirn":
            features, _ = load_FBIRN()
        else:
            print(f"Dataset '{self.dataset_name}' is undefined")
            return
        features = np.swapaxes(features, 1, 2)  # [n_samples; seq_len; n_features]

        if "saliency" in introspection_methods:
            os.makedirs(self.image_path.joinpath("saliency/colormap"))
            os.makedirs(self.image_path.joinpath("saliency/barchart"))
        if "ig" in introspection_methods:
            os.makedirs(self.image_path.joinpath("ig/colormap"))
            os.makedirs(self.image_path.joinpath("ig/barchart"))
        if "ignt" in introspection_methods:
            os.makedirs(self.image_path.joinpath("ignt/colormap"))
            os.makedirs(self.image_path.joinpath("ignt/barchart"))

        for i, feature in enumerate(features):
            self.introspection(i, feature, introspection_methods)


# if __name__ == "__main__":
#     dataset_name = "oasis"
#     model_name = "lstm"
#     model_path = LOGS_ROOT.joinpath("220615.035350-lstm-oasis/k_0/0000/model.best.pth")
#     image_path = ASSETS_ROOT.joinpath(f"images/{UTCNOW}-{model_name}-{dataset_name}")

#     model = LSTM(
#         input_size=53,
#         input_len=156,
#         hidden_size=52,
#         num_layers=3,
#         batch_first=True,
#         bidirectional=False,
#         fc_dropout=0.2626756675371412,
#     )
#     checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
#     model.load_state_dict(checkpoint)
#     model = model.eval()

#     introspection = Introspection(
#         dataset_name=dataset_name,
#         model_name=model_name,
#         image_path=image_path,
#         model_path=model_path,
#     )
#     introspection.initialize_model(model=model)

#     introspection_methods = set(["saliency", "ig", "ignt"])
#     introspection.run_introspection(introspection_methods)

# if __name__ == "__main__":
#     dataset_name = "fbirn"
#     model_name = "mlp"
#     model_path = LOGS_ROOT.joinpath(
#         "220628.041515-ts-mlp-oasis-qFalse/0000/model.best.pth"
#     )
#     image_path = ASSETS_ROOT.joinpath(f"images/{UTCNOW}-{model_name}-{dataset_name}")

#     hidden_size = 142
#     num_layers = 2
#     dropout = 0.15847198018446662
#     lr = 0.0002222585782420201

#     model = AttentionMLP(
#         input_size=53,  # PRIOR
#         output_size=2,  # PRIOR
#         hidden_size=hidden_size,
#         num_layers=num_layers,
#         dropout=dropout,
#     )
#     checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
#     model.load_state_dict(checkpoint)
#     model = model.eval()

#     introspection = Introspection(
#         dataset_name=dataset_name,
#         model_name=model_name,
#         image_path=image_path,
#         model_path=model_path,
#     )
#     introspection.initialize_model(model=model)

#     introspection_methods = set(["saliency", "ig", "ignt"])
#     introspection.run_introspection(introspection_methods)

if __name__ == "__main__":
    dataset_name = "fbirn"
    model_name = "mlp"
    model_path = LOGS_ROOT.joinpath(
        "220629.030429-ts-mlp-oasis-qFalse/0000/model.best.pth"
    )
    image_path = ASSETS_ROOT.joinpath(f"images/{UTCNOW}-{model_name}-{dataset_name}")

    hidden_size = 128
    num_layers = 0
    dropout = 0.14665514458122644
    lr = 0.00027129837277095967

    model = MLP(
        input_size=53,  # PRIOR
        output_size=2,  # PRIOR
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    model = model.eval()

    introspection = Introspection(
        dataset_name=dataset_name,
        model_name=model_name,
        image_path=image_path,
        model_path=model_path,
    )
    introspection.initialize_model(model=model)

    introspection_methods = set(["saliency", "ig", "ignt"])
    introspection.run_introspection(introspection_methods)
