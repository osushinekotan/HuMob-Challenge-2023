import math
from pathlib import Path

import joblib
import pandas as pd
import torch
from custom.config_types import CONFIG_TYPES
from custom.helper import train_fn, valid_fn
from logger import Logger
from pytorch_pfn_extras.config import Config
from util import load_yaml

logger = Logger(name="fe")

# set config
pre_eval_config = load_yaml()
# config = Config(pre_eval_config, types=CONFIG_TYPES)


def load_feature_df(pre_eval_config, name):
    filepath_for_features_df = (
        Path(pre_eval_config["global"]["resources"])
        / "output"
        / pre_eval_config["fe"]["out_dir"]
        / pre_eval_config["fe"]["dataset"]
        / f"{name}.pkl"
    )
    return joblib.load(filepath_for_features_df)


def make_sequences(df: pd.DataFrame, group_key: str, group_values: list[str]):
    with logger.time_log("make_sequences"):
        grouped = df.groupby(group_key, sort=False)
        sequences = [torch.tensor(group[group_values].to_numpy()) for _, group in grouped]
    return sequences


def calc_steps(
    train_length: int,
    batch_size: int,
    max_epochs: int,
    gradient_accumulation_steps: int | None,
) -> tuple[int, int]:
    iters_per_epoch = math.floor(train_length / batch_size)
    training_steps = (
        iters_per_epoch if gradient_accumulation_steps is None else iters_per_epoch // gradient_accumulation_steps
    ) * max_epochs
    return training_steps


def set_config(pre_eval_config, train_feature_df, valid_feature_df):
    # update parameters
    pre_eval_config["nn"]["scheduler"]["num_training_steps"] = calc_steps(
        train_length=len(train_feature_df),
        batch_size=pre_eval_config["nn"]["dataloader"]["train"]["batch_size"],
        max_epochs=pre_eval_config["nn"]["max_epochs"],
        gradient_accumulation_steps=pre_eval_config["nn"]["gradient_accumulation_steps"],
    )

    # set feature names
    feature_names = pre_eval_config["nn"]["feature"]["feature_names"]
    feature_names = (
        feature_names if feature_names != "???" else [x for x in train_feature_df.columns if x.startswith("f_")]
    )

    # train sequences
    pre_eval_config["nn"]["dataset"]["train"]["feature_seqs"] = make_sequences(
        df=train_feature_df,
        group_key="uid",
        group_values=feature_names,
    )
    pre_eval_config["nn"]["dataset"]["train"]["auxiliary_seqs"] = make_sequences(
        df=train_feature_df.query("d >= 60"),
        group_key="uid",
        group_values=pre_eval_config["nn"]["feature"]["auxiliary_names"],
    )
    pre_eval_config["nn"]["dataset"]["train"]["target_seqs"] = make_sequences(
        df=train_feature_df.query("d >= 60"),
        group_key="uid",
        group_values=pre_eval_config["nn"]["feature"]["target_names"],
    )

    # valid sequences
    pre_eval_config["nn"]["dataset"]["valid"]["feature_seqs"] = make_sequences(
        df=valid_feature_df,
        group_key="uid",
        group_values=feature_names,
    )
    pre_eval_config["nn"]["dataset"]["valid"]["auxiliary_seqs"] = make_sequences(
        df=valid_feature_df.query("d >= 60"),
        group_key="uid",
        group_values=pre_eval_config["nn"]["feature"]["auxiliary_names"],
    )
    pre_eval_config["nn"]["dataset"]["valid"]["target_seqs"] = make_sequences(
        df=valid_feature_df.query("d >= 60"),
        group_key="uid",
        group_values=pre_eval_config["nn"]["feature"]["target_names"],
    )

    # check
    assert (
        len(pre_eval_config["nn"]["dataset"]["valid"]["target_seqs"])
        == len(pre_eval_config["nn"]["dataset"]["valid"]["auxiliary_seqs"])
        == len(pre_eval_config["nn"]["dataset"]["valid"]["feature_seqs"])
    )
    assert (
        len(pre_eval_config["nn"]["dataset"]["train"]["target_seqs"])
        == len(pre_eval_config["nn"]["dataset"]["train"]["auxiliary_seqs"])
        == len(pre_eval_config["nn"]["dataset"]["train"]["feature_seqs"])
    )

    return Config(config, types=CONFIG_TYPES)


def train_loop(config, df):
    train_fn(config=config, wandb_logger=None)


raw_train_feature_df = load_feature_df(pre_eval_config=pre_eval_config, name="train_feature_df")

num_fold = pre_eval_config["cv"]["num_fold"]
valid_folds = pre_eval_config["cv"]["valid_folds"]
config = set_config(pre_eval_config, train_feature_df, valid_feature_df)
