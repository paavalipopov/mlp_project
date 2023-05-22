# pylint: disable=invalid-name, too-many-function-args
"""Functions for extracting dataset features and labels"""
from importlib import import_module

import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold

from omegaconf import OmegaConf, open_dict


def data_factory(cfg):
    """
    Return dict of raw data based on config
    Dict contains
    {
        "main" dataset,
        "additional test" datasets,
    }
    datasets are dicts of
    {
        "TS": TS-data of shape [subjects, time, components],
        "labels": labels
    } if model is TS or unspecified;
    {
        "FNC": FNC-data of shape [subjects, components, components],
        "labels": labels
    } if FNC;
    {
        "FNC": FNC-data of shape [subjects, flattened_upper_FNC_triangle],
        "labels": labels
    } if tri-FNC;
    {
        "TS": TS-data of shape [subjects, time, components],
        "FNC": FNC-data of shape [subjects, components, components],
        "labels": labels
    } if TS-FNC;

    "TS" data is z-scored over time if config says so
    "FNC" is obtained using Pearson correlation coefficients
    """

    # load dataset
    data = {}
    data_shape = {}

    try:
        dataset_module = import_module(f"src.datasets.{cfg.dataset.name}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"No module named '{cfg.dataset.name}' \
                                  found in 'src.datasets'. Check if dataset name \
                                  in config file and its module name are the same"
        ) from e

    try:
        ts_data, labels = dataset_module.load_data(cfg)
        n_classes = np.unique(labels).shape[0]
    except AttributeError as e:
        raise AttributeError(
            f"'src.datasets.{cfg.dataset.name}' has no function\
                             'load_data'. Is the function misnamed/not defined?"
        ) from e

    # select tuning holdout (if needed)
    if cfg.exp.single_HP and cfg.exp.tuning_holdout:
        assert (
            cfg.exp.tuning_split is not None
        ), "you must specify 'exp.tuning_split' if \
                 'exp.tuning_holdout' is set to True"
        assert isinstance(cfg.exp.tuning_split, int)

        skf = StratifiedKFold(
            n_splits=cfg.exp.tuning_split, shuffle=True, random_state=42
        )
        CV_folds = list(skf.split(ts_data, labels))
        train_index, test_index = CV_folds[0]
        if cfg.exp.mode == "tune":
            ts_data = ts_data[test_index]
            labels = labels[test_index]
        else:
            ts_data = ts_data[train_index]
            labels = labels[train_index]

    # z-score the data over time
    if cfg.dataset.zscore:
        ts_data = stats.zscore(ts_data, axis=1)

    # derive FNC data, if needed
    if "data_type" not in cfg.model or cfg.model.data_type == "TS":
        data["main"] = {"TS": ts_data, "labels": labels}
        data_shape["main"] = ts_data.shape
    elif cfg.model.data_type in ["FNC", "tri-FNC", "TS-FNC"]:
        pearson = np.zeros((ts_data.shape[0], ts_data.shape[2], ts_data.shape[2]))
        for i in range(ts_data.shape[0]):
            pearson[i, :, :] = np.corrcoef(ts_data[i, :, :], rowvar=False)

        if cfg.model.data_type == "FNC":
            data["main"] = {"FNC": pearson, "labels": labels}
            data_shape["main"] = pearson.shape
        elif cfg.model.data_type == "tri-FNC":
            tril_inx = np.tril_indices(pearson.shape[1])
            triangle = np.zeros((pearson.shape[0], tril_inx[0].shape[0]))
            for i in range(triangle.shape[0]):
                triangle[i] = pearson[i][tril_inx]
            data["main"] = {"FNC": triangle, "labels": labels}
            data_shape["main"] = triangle.shape
        elif cfg.model.data_type == "TS-FNC":
            data["main"] = {"TS": ts_data, "FNC": pearson, "labels": labels}
            data_shape["main"] = {"TS": ts_data.shape, "FNC": pearson.shape}

    # TODO: add additional test datasets and appropriate conditions
    # TODO: add dataset's cropping for time-length-sensitive models

    data_info = OmegaConf.create(
        {
            "data_shape": data_shape,
            "n_classes": n_classes,
        }
    )

    with open_dict(cfg):
        cfg.dataset.data_info = data_info

    return data


def data_postfactory(cfg, model_cfg, original_data):
    """
    Post-process the raw dataset data according to model_cfg
    """
    if "require_data_postproc" not in cfg.model or not cfg.model.require_data_postproc:
        data = original_data
    else:
        try:
            model_module = import_module(f"src.models.{cfg.model.name}")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"No module named '{cfg.model.name}' \
                                    found in 'src.models'. Check if model name \
                                    in config file and its module name are the same"
            ) from e

        try:
            data_postproc = model_module.data_postproc
        except AttributeError as e:
            raise AttributeError(
                f"'src.models.{cfg.model.name}' has no function\
                                'data_postproc'. Is the function misnamed/not defined?"
            ) from e

        data = data_postproc(original_data, model_cfg)

    return data
