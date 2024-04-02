# pylint: disable=too-many-function-args, invalid-name
""" FBIRN ICA dataset loading script"""
import h5py
import numpy as np
import pandas as pd

from omegaconf import DictConfig

from src.settings import DATA_ROOT

from src.datasets.fbirn import load_data as load_fbirn
from src.datasets.bsnip import load_data as load_bsnip
from src.datasets.cobre import load_data as load_cobre

def pad(data_1, data_2):
    len_1 = data_1.shape[1]
    len_2 = data_2.shape[1]

    diff = np.abs(len_1 - len_2)

    if len_1 < len_2:
        data_1 = np.pad(data_1, ((0, 0), (0, diff), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
    elif len_2 < len_1:
        data_2 = np.pad(data_2, ((0, 0), (0, diff), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))

    return data_1, data_2

def load_data(
    cfg: DictConfig
):
    """
    Return FBIRN and BSNIP data

    Output:
    features with shape [n_samples, time_length, feature_size],
    labels
    """

    # get data
    data_1, labels_1 = load_bsnip(cfg)
    data_2, labels_2 = load_cobre(cfg)
    data_3, labels_3 = load_fbirn(cfg)

    data_1, data_2 = pad(data_1, data_2)
    data_1, data_3 = pad(data_1, data_3)

    data = np.concatenate((data_1, data_2, data_3))
    labels = np.concatenate((labels_1, labels_2, labels_3))

    return data, labels
