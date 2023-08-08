import gc
import math
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from custom.config_types import CONFIG_TYPES
from custom.helper import train_fn, valid_fn
from logger import Logger
from pytorch_pfn_extras.config import Config
from util import load_yaml, seed_everything

import wandb

logger = Logger(name="train")
wandb.login(key=os.environ["WANDB_KEY"])  # need wandb account


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


def set_config(pre_eval_config: dict, train_feature_df: pd.DataFrame, valid_feature_df: pd.DataFrame) -> Config:
    # update parameters
    num_training_steps, iters_per_epoch = calc_steps(
        train_length=(train_feature_df["uid"].nunique()),
        batch_size=pre_eval_config["nn"]["dataloader"]["train"]["batch_size"],
        max_epochs=pre_eval_config["nn"]["max_epochs"],
        gradient_accumulation_steps=pre_eval_config["nn"]["gradient_accumulation_steps"],
    )
    print(num_training_steps)
    pre_eval_config["nn"]["num_training_steps"] = num_training_steps
    pre_eval_config["nn"]["iters_per_epoch"] = iters_per_epoch
    pre_eval_config["nn"]["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set feature names
    feature_names = pre_eval_config["nn"]["feature"]["feature_names"]
    feature_names = (
        feature_names if feature_names != "???" else [x for x in train_feature_df.columns if x.startswith("f_")]
    )

    # target (fix)
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

    # model
    pre_eval_config["nn"]["model"]["input_size1"] = len(feature_names)
    pre_eval_config["nn"]["model"]["input_size2"] = len(pre_eval_config["nn"]["feature"]["auxiliary_names"])
    pre_eval_config["nn"]["model"]["output_size"] = len(pre_eval_config["nn"]["feature"]["target_names"])

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


def train_loop(pre_eval_config: dict, train_data: Any, valid_data: Any, loop_name: str) -> None:
    # eval config
    config = set_config(pre_eval_config, train_data, valid_data)
    metrics = config["/nn/metrics"]  # get metrics
    model = config["/nn/model"]
    train_dataloader = config["/nn/dataloader/train"]
    valid_dataloader = config["/nn/dataloader/valid"]
    criterion = config["/nn/criterion"]
    optimizer = config["/nn/optimizer"]
    scheduler = config["/nn/scheduler"]

    max_epochs = pre_eval_config["nn"]["max_epochs"]
    out_dir = Path(config["/global/resources"]) / "output" / config["nn/out_dir"]

    # setup
    wandb.init(
        project=config["/global/project"],
        name=config["/nn/out_dir"],
        group=loop_name,
        job_type="train",
        anonymous=None,
        reinit=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    total_step = 0  # to record total step accross epochs
    best_score = -np.inf
    for epoch in range(max_epochs):
        # train
        tr_output = train_fn(
            config=config,
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            total_step=total_step,
            wandb_logger=wandb,
        )
        loss, step = tr_output["loss"], tr_output["step"]
        total_step += step

        # valid
        va_output = valid_fn(
            config=config,
            model=model,
            dataloader=valid_dataloader,
        )
        eval_score = metrics(va_output["outputs"], va_output["targets"])

        # logs
        logs = {
            "epoch": epoch,
            "eval_score": float(eval_score),
            "train_loss_epoch": loss.item(),
            "valid_loss_epoch": va_output["loss"].item(),
        }
        logger.info(logs)
        wandb.log(logs)

        if best_score < eval_score:
            best_score = eval_score
            logger.info(f"epoch {epoch} - best score: {best_score:.4f} model 🌈")

            torch.save(model.state_dict(), out_dir / f"{loop_name}.pth")  # save model weight
            joblib.dump(va_output, out_dir / f"{loop_name}.pkl")  # save outputs

    wandb.finish(quiet=True)
    torch.cuda.empty_cache()
    gc.collect()
    best_val_outputs = joblib.load(out_dir / f"{loop_name}.pkl")
    return best_val_outputs


def train_fold(pre_eval_config: dict, df: pd.DataFrame) -> None:
    num_fold = pre_eval_config["cv"]["num_fold"]
    valid_folds = pre_eval_config["cv"]["valid_folds"]
    oof_outputs = []
    for i_fold in range(num_fold):
        if i_fold not in valid_folds:
            continue

        with logger.time_log(f"fold {i_fold}"):
            train_feature_df = df[df["fold"] != i_fold].reset_index(drop=True)
            valid_feature_df = df[df["fold"] == i_fold].reset_index(drop=True)

            best_outputs = train_loop(
                pre_eval_config=pre_eval_config,
                train_data=train_feature_df,
                valid_data=valid_feature_df,
                loop_name=f"fold_{i_fold}",
            )
            oof_outputs.append(best_outputs)
    return oof_outputs


def main() -> None:
    pre_eval_config = load_yaml()
    seed_everything(pre_eval_config["global"]["seed"])
    out_dir = Path(pre_eval_config["global"]["resources"]) / "output" / pre_eval_config["nn"]["out_dir"]

    feature_df = load_feature_df(pre_eval_config=pre_eval_config, name="train_feature_df")
    with logger.time_log("train_fold"):
        oof_outputs = train_fold(pre_eval_config, df=feature_df)
    joblib.dump(oof_outputs, out_dir / "oof_outputs.pkl")


if __name__ == "__main__":
    main()
