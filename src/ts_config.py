import os
import json
import pandas as pd

from src.scripts.ts_dl_experiments import Experiment
from src.settings import LOGS_ROOT


def get_tune_config(exp: Experiment, random_seed):
    from scipy.stats import randint, uniform, loguniform

    model_config = {}

    # pick random hyperparameters
    if exp.model in [
        "mlp",
        "wide_mlp",
        "deep_mlp",
        "attention_mlp",
        "new_attention_mlp",
        "meta_mlp",
        "pe_mlp",
        "pe_att_mlp",
    ]:
        model_config["hidden_size"] = randint.rvs(32, 256, random_state=random_seed)
        model_config["num_layers"] = randint.rvs(0, 4, random_state=random_seed)
        model_config["dropout"] = uniform.rvs(0.1, 0.9, random_state=random_seed)

        if exp.model == "wide_mlp":
            model_config["hidden_size"] = randint.rvs(
                256, 1024, random_state=random_seed
            )
        elif exp.model == "deep_mlp":
            model_config["num_layers"] = randint.rvs(4, 20, random_state=random_seed)

        if exp.model == "attention_mlp":
            model_config["time_length"] = exp.data_shape[1]
        elif exp.model in ["new_attention_mlp", "pe_att_mlp"]:
            model_config["time_length"] = exp.data_shape[1]
            model_config["attention_size"] = randint.rvs(
                32, 256, random_state=random_seed
            )

        model_config["input_size"] = exp.data_shape[2]
        model_config["output_size"] = exp.n_classes

    elif exp.model in ["lstm", "noah_lstm"]:
        model_config["hidden_size"] = randint.rvs(32, 256, random_state=random_seed)
        model_config["num_layers"] = randint.rvs(1, 4, random_state=random_seed)
        model_config["bidirectional"] = bool(randint.rvs(0, 1, random_state=random_seed))
        model_config["fc_dropout"] = uniform.rvs(0.1, 0.9, random_state=random_seed)

        model_config["input_size"] = exp.data_shape[2]
        model_config["output_size"] = exp.n_classes
    elif exp.model in [
        "transformer",
        "mean_transformer",
        "first_transformer",
        "pe_transformer",
    ]:
        model_config["head_hidden_size"] = randint.rvs(4, 128, random_state=random_seed)
        model_config["num_heads"] = randint.rvs(1, 4, random_state=random_seed)
        model_config["num_layers"] = randint.rvs(1, 4, random_state=random_seed)
        model_config["fc_dropout"] = uniform.rvs(0.1, 0.9, random_state=random_seed)

        model_config["input_size"] = exp.data_shape[2]
        model_config["output_size"] = exp.n_classes
    elif exp.model == "window_mlp":
        model_config["input_size"] = exp.data_shape[2]
        model_config["output_size"] = exp.n_classes

        model_config["hidden_size"] = randint.rvs(32, 256, random_state=random_seed)
        model_config["num_layers"] = randint.rvs(0, 4, random_state=random_seed)
        model_config["dropout"] = uniform.rvs(0.1, 0.9, random_state=random_seed)

        model_config["decoder"] = {}
        model_config["decoder"]["type"] = exp.model_decoder
        if exp.model_decoder == "lstm":
            model_config["decoder"]["hidden_size"] = randint.rvs(
                32, 256, random_state=random_seed
            )
            model_config["decoder"]["num_layers"] = randint.rvs(
                1, 4, random_state=random_seed
            )
            model_config["decoder"]["bidirectional"] = True
        elif exp.model_decoder == "tf":
            model_config["decoder"]["head_hidden_size"] = randint.rvs(
                4, 128, random_state=random_seed
            )
            model_config["decoder"]["num_heads"] = randint.rvs(
                1, 4, random_state=random_seed
            )
            model_config["decoder"]["num_layers"] = randint.rvs(
                1, 4, random_state=random_seed
            )

        model_config["mode"] = exp.model_mode

        model_config["datashape"] = {}
        model_config["datashape"]["window_size"] = randint.rvs(
            5, exp.data_shape[1] // 5, random_state=random_seed
        )
        # window shift determines how much the windows overlap
        model_config["datashape"]["window_shift"] = randint.rvs(
            2, model_config["datashape"]["window_size"], random_state=random_seed
        )
    elif exp.model == "mlp_tf":
        model_config["input_size"] = exp.data_shape[2]
        model_config["output_size"] = exp.n_classes

        model_config["hidden_size"] = randint.rvs(32, 256, random_state=random_seed)
        model_config["num_layers"] = randint.rvs(0, 4, random_state=random_seed)
        model_config["dropout"] = uniform.rvs(0.1, 0.9, random_state=random_seed)

        model_config["decoder"] = {}
        model_config["decoder"]["type"] = "tf"
        model_config["decoder"]["head_hidden_size"] = randint.rvs(
            4, 128, random_state=random_seed
        )
        model_config["decoder"]["num_heads"] = randint.rvs(
            1, 4, random_state=random_seed
        )
        model_config["decoder"]["num_layers"] = randint.rvs(
            1, 4, random_state=random_seed
        )

    else:
        raise NotImplementedError()

    model_config["lr"] = loguniform.rvs(1e-5, 1e-3, random_state=random_seed)

    print("Tuning config:")
    print(model_config)

    return model_config


def get_best_config(exp: Experiment):
    model_config = {}

    # find and load the best tuned model
    runs_files = []

    searched_dir = exp.project_name.split("-")
    searched_dir = "-".join(searched_dir[1:3])
    serached_dir = f"tune-{searched_dir}"
    if exp.project_prefix != exp.utcnow:
        serached_dir = f"{exp.project_prefix}-{serached_dir}"
    print(f"Searching trained model in {LOGS_ROOT}/*{serached_dir}")
    for logdir in os.listdir(LOGS_ROOT):
        if serached_dir in logdir:
            runs_files.append(os.path.join(LOGS_ROOT, logdir))

    # if multiple run files found, choose the latest
    runs_file = sorted(runs_files)[-1]
    print(f"Using best model from {runs_file}")

    # get model config
    df = pd.read_csv(f"{runs_file}/runs.csv", delimiter=",", index_col=False)
    # pick hyperparams of a model with the highest test_score
    best_config_path = df.loc[df["score"].idxmax()].to_dict()
    best_config_path = best_config_path["path_to_config"]
    with open(best_config_path, "r") as fp:
        model_config = json.load(fp)

    print("Loaded model config:")
    print(model_config)

    return model_config
