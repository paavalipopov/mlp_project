#pylint: disable=C0103,C0115,C0116,R0913,E1121,C0301
"""
Functions for extracting dataset features and labels
"""
import h5py
import numpy as np
import pandas as pd

from src.settings import DATA_ROOT


def load_ABIDE1(
    dataset_path: str = DATA_ROOT.joinpath("abide/ABIDE1_AllData.h5"),
    indices_path: str = DATA_ROOT.joinpath("abide/correct_indices_GSP.csv"),
    labels_path: str = DATA_ROOT.joinpath("abide/labels_ABIDE1.csv"),
):
    """
    Return ABIDE1 data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("abide/ABIDE1_AllData.h5") - path to the dataset
    indices_path: str = DATA_ROOT.joinpath("abide/correct_indices_GSP.csv") - path to correct indices/components
    labels_path: str = DATA_ROOT.joinpath("abide/labels_ABIDE1.csv") - path to labels

    Output:
    features, labels
    """

    # get data
    hf = h5py.File(dataset_path, "r")
    data = hf.get("ABIDE1_dataset")
    data = np.array(data)
    # print(data.shape)
    # >>> (569, 14000)

    # reshape data
    num_subjects = data.shape[0]
    num_components = 100
    data = data.reshape(num_subjects, num_components, -1)
    # 569 - sessions - data.shape[0]
    # 100 - components - data.shape[1]
    # 140 - time points - data.shape[2]

    # get correct indices/components
    indices = pd.read_csv(indices_path, header=None)
    idx = indices[0].values - 1
    # filter the data: leave only correct components
    data = data[:, idx, :]

    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    return data, labels


def load_FBIRN(
    dataset_path: str = DATA_ROOT.joinpath("fbirn/FBIRN_AllData.h5"),
    indices_path: str = DATA_ROOT.joinpath("fbirn/correct_indices_GSP.csv"),
    labels_path: str = DATA_ROOT.joinpath("fbirn/labels_FBIRN_new.csv"),
):
    """
    Return FBIRN data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("fbirn/FBIRN_AllData.h5") - path to the dataset
    indices_path: str = DATA_ROOT.joinpath("fbirn/correct_indices_GSP.csv") - path to correct indices/components
    labels_path: str = DATA_ROOT.joinpath("fbirn/labels_FBIRN_new.csv") - path to labels

    Output:
    features, labels
    """

    # get data
    hf = h5py.File(dataset_path, "r")
    data = hf.get("FBIRN_dataset")
    data = np.array(data)
    # print(data.shape)
    # >>> (311, 14000)

    # reshape data
    num_subjects = data.shape[0]
    num_components = 100
    data = data.reshape(num_subjects, num_components, -1)
    # 311 - sessions - data.shape[0]
    # 100 - components - data.shape[1]
    # 140 - time points - data.shape[2]

    # get correct indices/components
    indices = pd.read_csv(indices_path, header=None)
    idx = indices[0].values - 1
    # filter the data: leave only correct components
    data = data[:, idx, :]

    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    return data, labels


def load_OASIS(
    only_first_sessions: bool = True,
    only_two_classes: bool = True,
    dataset_path: str = DATA_ROOT.joinpath("oasis/OASIS3_AllData_allsessions.npz"),
    indices_path: str = DATA_ROOT.joinpath("oasis/correct_indices_GSP.csv"),
    labels_path: str = DATA_ROOT.joinpath("oasis/labels_OASIS_6_classes.csv"),
    sessions_path: str = DATA_ROOT.joinpath("oasis/oasis_first_sessions_index.csv"),
):
    """
    Return OASIS data

    Input:
    only_first_sessions: bool = True - filter the second sessions from the output
    only_two_classes: bool = True - filter all classes except for 0 and 1
    dataset_path: str = DATA_ROOT.joinpath("oasis/OASIS3_AllData_allsessions.npz") - path to the dataset
    indices_path: str = DATA_ROOT.joinpath("oasis/correct_indices_GSP.csv") - path to correct indices/components
    labels_path: str = DATA_ROOT.joinpath("oasis/labels_OASIS_6_classes.csv") - path to labels
    sessions_path: str = DATA_ROOT.joinpath("oasis/oasis_first_sessions_index.csv") - path to indices of the first sessions

    Output:
    features, labels
    """

    data = np.load(dataset_path)
    # 2826 - sessions - data.shape[0]
    # 100 - components - data.shape[1]
    # 160 - time points - data.shape[2]

    # get correct indices/components
    indices = pd.read_csv(indices_path, header=None)
    idx = indices[0].values - 1

    # filter the data: leave only correct components and the first 156 time points
    # (not all subjects have all 160 time points)
    data = data[:, idx, :156]

    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    if only_first_sessions:
        # leave only first sessions
        sessions = pd.read_csv(sessions_path, header=None)
        first_session = sessions[0].values - 1

        data = data[first_session, :, :]
        # 912 - sessions - data.shape[0] - only first session
        labels = labels[first_session]

    if only_two_classes:
        # leave subjects of class 0 and 1 only
        filter_array = []
        for label in labels:
            if label in (0, 1):
                filter_array.append(True)
            else:
                filter_array.append(False)

        data = data[filter_array, :, :]
        # 2559 - sessions - data.shape[0] - subjects of class 0 and 1
        # 823 - sessions - data.shape[0] - if only first sessions are considered
        labels = labels[filter_array]

    return data, labels


def load_balanced_OASIS():
    """
    Return 320 balanced OASIS subjects (classes 0 and 1 only)

    Output:
    features, labels
    """

    features, labels = load_OASIS(only_first_sessions = True, only_two_classes = True)

    # for 651 subjects with label 0
    filter_array_0 = []
    # for 172 subjects with label 1
    filter_array_1 = []

    for label in labels:
        if label == 0:
            filter_array_0.append(True)
            filter_array_1.append(False)
        else:
            filter_array_0.append(False)
            filter_array_1.append(True)

    # copy subjects to separate arrays
    features_0 = features[filter_array_0]
    labels_0 = labels[filter_array_0]
    features_1 = features[filter_array_1]
    labels_1 = labels[filter_array_1]

    # balance the arrays
    features_0 = features_0[:160]
    labels_0 = labels_0[:160]
    features_1 = features_1[:160]
    labels_1 = labels_1[:160]

    features = np.concatenate((features_0, features_1), axis=0)
    labels = np.concatenate((labels_0, labels_1), axis=0)

    return features, labels
