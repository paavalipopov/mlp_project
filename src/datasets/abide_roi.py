# pylint: disable=too-many-function-args, invalid-name, unused-argument
""" ABIDE ROI dataset loading script"""
import numpy as np
import pandas as pd

from omegaconf import DictConfig

from src.settings import DATA_ROOT


def load_data(
    cfg: DictConfig,
    dataset_path: str = DATA_ROOT.joinpath(
        "abide_roi/ABIDE1_AllData_871Subjects_region_shaefer200_316TP_onlytimeserieszscored.npz"
    ),
    labels_path: str = DATA_ROOT.joinpath("abide_roi/ABIDE1_region_labels_871.csv"),
):
    """
    Return ROI ABIDE data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("abide_roi/ABIDE1_AllData_871Subjects_region_shaefer200_316TP_onlytimeserieszscored.npz")
    - path to the dataset
    labels_path: str = DATA_ROOT.joinpath("abide_roi/ABIDE1_region_labels_871.csv")
    - path to labels

    Output:
    features, labels
    """

    # get data
    data = np.load(dataset_path)
    # print(data.shape)
    # >>> (871, 200, 316)

    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1
    # (871,)

    data = np.swapaxes(data, 1, 2)
    # data.shape = [n_samples, time_length, feature_size]

    return data, labels
