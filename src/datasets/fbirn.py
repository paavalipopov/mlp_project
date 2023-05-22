import h5py
import numpy as np
import pandas as pd

from src.settings import DATA_ROOT

def load_data(
    cfg,
    dataset_path: str = DATA_ROOT.joinpath("fbirn/FBIRN_AllData.h5"),
    indices_path: str = DATA_ROOT.joinpath("fbirn/correct_indices_GSP.csv"),
    labels_path: str = DATA_ROOT.joinpath("fbirn/labels_FBIRN_new.csv"),
):
    """
    Return FBIRN data

    Output:
    features with shape [subjects, time, components],
    labels
    """

    # get data
    hf = h5py.File(dataset_path, "r")
    data = hf.get("FBIRN_dataset")
    data = np.array(data)
    # data.shape = [311, 14000]

    # reshape data
    data = data.reshape(data.shape[0], 100, -1)
    # data.shape = [311, 100, 140]

    if cfg.dataset.filter_indices:
        # get correct indices/components
        indices = pd.read_csv(indices_path, header=None)
        idx = indices[0].values - 1
        data = data[:, idx, :]
        # data.shape = [311, 53, 140]

    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    data = np.swapaxes(data, 1, 2)

    return data, labels