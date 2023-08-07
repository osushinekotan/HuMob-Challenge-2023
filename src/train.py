import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from custom.config_types import CONFIG_TYPES
from custom.helper import train_fn, valid_fn
from logger import Logger
from pytorch_pfn_extras.config import Config
from util import load_yaml

logger = Logger(name="fe")


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
    return training_steps, iters_per_epoch


def set_config(pre_eval_config, train_feature_df, valid_feature_df):
    # update parameters
    num_training_steps, iters_per_epoch = calc_steps(
        train_length=len(train_feature_df),
        batch_size=pre_eval_config["dataloader"]["train"]["batch_size"],
        max_epochs=pre_eval_config["nn"]["max_epochs"],
        gradient_accumulation_steps=pre_eval_config["nn"]["gradient_accumulation_steps"],
    )
    pre_eval_config["nn"]["num_training_steps"] = num_training_steps
    pre_eval_config["nn"]["iters_per_epoch"] = iters_per_epoch

    # set feature names
    feature_names = pre_eval_config["nn"]["feature"]["feature_names"]
    feature_names = (
        feature_names if feature_names != "???" else [x for x in train_feature_df.columns if x.startswith("f_")]
    )

    # target
    lower_target_d = 60

    # train sequences
    pre_eval_config["nn"]["dataset"]["train"]["feature_seqs"] = make_sequences(
        df=train_feature_df,
        group_key="uid",
        group_values=feature_names,
    )
    pre_eval_config["nn"]["dataset"]["train"]["auxiliary_seqs"] = make_sequences(
        df=train_feature_df.query(f"d >= {lower_target_d}"),
        group_key="uid",
        group_values=pre_eval_config["nn"]["feature"]["auxiliary_names"],
    )
    pre_eval_config["nn"]["dataset"]["train"]["target_seqs"] = make_sequences(
        df=train_feature_df.query(f"d >= {lower_target_d}"),
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
        df=valid_feature_df.query(f"d >= {lower_target_d}"),
        group_key="uid",
        group_values=pre_eval_config["nn"]["feature"]["auxiliary_names"],
    )
    pre_eval_config["nn"]["dataset"]["valid"]["target_seqs"] = make_sequences(
        df=valid_feature_df.query(f"d >= {lower_target_d}"),
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

    return Config(pre_eval_config, types=CONFIG_TYPES)


def train_loop(pre_eval_config, train_data, valid_data, loop_name, wandb_logger=None):
    # eval config
    config = set_config(pre_eval_config, train_data, valid_data)
    metrics = config["/nn/metrics"]  # get metrics
    model = config["/nn/model"]
    max_epochs = pre_eval_config["nn"]["max_epochs"]
    out_dir = Path(config["/global/resources"]) / "output" / config["nn/out_dir"]
    
    

    total_step = 0  # to record total step accross epochs
    best_score = -np.inf
    for epoch in range(max_epochs):
        # train
        tr_output = train_fn(
            config=config,
            model=model,
            total_step=total_step,
            wandb_logger=wandb_logger,
        )
        loss, step = tr_output["loss"], tr_output["step"]
        total_step += step

        # valid
        va_output = valid_fn(config=config, model=model)
        eval_score = metrics(targets=va_output["targets"], outputs=va_output["outputs"])

        # logs
        logs = {
            "epoch": epoch,
            "eval_score": eval_score,
            "train_loss_epoch": loss.item(),
            "valid_loss_epoch": va_output["loss"].item(),
        }
        logger.info(logs)
        if wandb_logger is not None:
            wandb_logger.log(logs)

        if best_score < eval_score:
            best_score = eval_score
            logger.info(f"epoch {epoch} - best score: {best_score:.4f} model 🌈")

            torch.save(model.state_dict(), out_dir / f"{loop_name}.pth")  # save model weight
            joblib.dump(va_output, out_dir / f"{loop_name}.pkl")  # save outputs


def train_fold(pre_eval_config, df):
    num_fold = pre_eval_config["cv"]["num_fold"]
    valid_folds = pre_eval_config["cv"]["valid_folds"]

    for i_fold in num_fold:
        if i_fold not in valid_folds:
            continue

        with logger.time_log(f"fold {i_fold}"):
            train_feature_df = df[df["fold"] != i_fold].reset_index(drop=True)
            valid_feature_df = df[df["fold"] == i_fold].reset_index(drop=True)

            train_loop(
                config=pre_eval_config,
                train_data=train_feature_df,
                valid_data=valid_feature_df,
                loop_name=f"fold_{i_fold}",
            )


def main():
    pre_eval_config = load_yaml()
    feature_df = load_feature_df(pre_eval_config=pre_eval_config, name="train_feature_df")
    train_fold(pre_eval_config, df=feature_df)
