import gc
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from custom.config_types import CONFIG_TYPES
from custom.helper import inference_fn
from custom.runner.v01.train import retransform_regression_target
from logger import Logger
from pytorch_pfn_extras.config import Config
from util import load_yaml, seed_everything

logger = Logger(name="inference")


def load_feature_df(pre_eval_config: dict, name: str) -> Any:
    filepath_for_features_df = (
        Path(pre_eval_config["global"]["resources"])
        / "output"
        / pre_eval_config["fe"]["out_dir"]
        / pre_eval_config["fe"]["dataset"]
        / f"{name}.pkl"
    )
    return joblib.load(filepath_for_features_df)


def make_sequences(df: pd.DataFrame, group_key: str, group_values: list[str]) -> Any:
    with logger.time_log("make_sequences"):
        grouped = df.groupby(group_key, sort=False)
        sequences = [torch.tensor(group[group_values].to_numpy()) for _, group in grouped]
    return sequences


def set_model_config(pre_eval_config: dict, feature_names, auxiliary_names):
    # model
    if pre_eval_config["nn"]["model"]["type"].startswith("CustomLSTMModel"):
        pre_eval_config["nn"]["model"]["input_size1"] = len(feature_names)
        pre_eval_config["nn"]["model"]["input_size2"] = len(auxiliary_names)
        pre_eval_config["nn"]["model"]["output_size"] = len(pre_eval_config["nn"]["feature"]["target_names"])
        return pre_eval_config

    elif pre_eval_config["nn"]["model"]["type"].startswith("CustomTransformerModel"):
        pre_eval_config["nn"]["model"]["input_size_src"] = len(feature_names)
        pre_eval_config["nn"]["model"]["input_size_tgt"] = len(auxiliary_names)
        pre_eval_config["nn"]["model"]["output_size"] = len(pre_eval_config["nn"]["feature"]["target_names"])
        return pre_eval_config
    else:
        raise NotImplementedError()


def set_config(pre_eval_config: dict, test_feature_df: pd.DataFrame) -> Config:
    # update parameters
    pre_eval_config["nn"]["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f'device : {pre_eval_config["nn"]["device"]}')

    # set feature names
    feature_names = pre_eval_config["nn"]["feature"]["feature_names"]
    feature_names = (
        feature_names if feature_names != "???" else [x for x in test_feature_df.columns if x.startswith("f_")]
    )
    auxiliary_names = pre_eval_config["nn"]["feature"]["auxiliary_names"]
    auxiliary_names = (
        auxiliary_names
        if auxiliary_names != "???"
        else [x for x in test_feature_df.columns if x.startswith("f_d") or x.startswith("f_t")]
    )
    logger.info(f"feature_names : {feature_names}")
    logger.info(f"auxiliary_names : {auxiliary_names}")

    # target (fix)
    lower_target_d = 60

    # test sequences
    pre_eval_config["nn"]["dataset"]["test"]["feature_seqs"] = make_sequences(
        df=test_feature_df.query(f"d < {lower_target_d}"),
        group_key="uid",
        group_values=feature_names,
    )
    pre_eval_config["nn"]["dataset"]["test"]["auxiliary_seqs"] = make_sequences(
        df=test_feature_df.query(f"d >= {lower_target_d}"),
        group_key="uid",
        group_values=auxiliary_names,
    )

    # model
    pre_eval_config = set_model_config(pre_eval_config, feature_names, auxiliary_names)

    # check
    assert len(pre_eval_config["nn"]["dataset"]["test"]["auxiliary_seqs"]) == len(
        pre_eval_config["nn"]["dataset"]["test"]["feature_seqs"]
    )

    return Config(pre_eval_config, types=CONFIG_TYPES)


def inference_loop(pre_eval_config: dict, test_data: Any, loop_name: str, out_dir: Path) -> None:
    # eval config
    config = set_config(pre_eval_config, test_data)
    model = config["/nn/model"]
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{loop_name}.pth"
    state = torch.load(model_path)
    model.load_state_dict(state)

    outputs = inference_fn(config=config, model=model)

    del model, state
    gc.collect()
    torch.cuda.empty_cache()

    return outputs


def inference_fold(pre_eval_config: dict, df: pd.DataFrame, out_dir: Path) -> None:
    num_fold = pre_eval_config["cv"]["num_fold"]
    valid_folds = pre_eval_config["cv"]["valid_folds"]

    outputs = []
    for i_fold in range(num_fold):
        if i_fold not in valid_folds:
            continue

        with logger.time_log(f"fold {i_fold}"):
            i_outputs = inference_loop(
                pre_eval_config=pre_eval_config, test_data=df, loop_name=f"fold_{i_fold}", out_dir=out_dir
            )
        outputs.append(i_outputs)
    mean_outputs = np.mean(outputs, axis=0)

    mean_outputs = retransform_regression_target(
        config=Config(pre_eval_config), outputs=mean_outputs, data=df.query("d >= 60")
    )
    return mean_outputs


def run():
    pre_eval_config = load_yaml()
    seed_everything(pre_eval_config["global"]["seed"])
    out_dir = (
        Path(pre_eval_config["global"]["resources"])
        / "output"
        / pre_eval_config["nn"]["out_dir"]
        / pre_eval_config["fe"]["dataset"]
    )

    feature_df = load_feature_df(pre_eval_config=pre_eval_config, name="test_feature_df")
    with logger.time_log("test_fold"):
        test_outputs = inference_fold(pre_eval_config, df=feature_df, out_dir=out_dir)

    logger.debug(f"shape : {test_outputs.shape}")
    joblib.dump(test_outputs, out_dir / "test_outputs.pkl")


if __name__ == "__main__":
    run()
