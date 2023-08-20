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


def set_config(pre_eval_config: dict, train_feature_df: pd.DataFrame, valid_feature_df: pd.DataFrame) -> Config:
    # update parameters
    num_training_steps, iters_per_epoch = calc_steps(
        train_length=(train_feature_df["uid"].nunique()),
        batch_size=pre_eval_config["nn"]["dataloader"]["train"]["batch_size"],
        max_epochs=pre_eval_config["nn"]["max_epochs"],
        gradient_accumulation_steps=pre_eval_config["nn"]["gradient_accumulation_steps"],
    )
    logger.debug(f"num_training_steps: {num_training_steps}")

    pre_eval_config["nn"]["num_training_steps"] = num_training_steps
    pre_eval_config["nn"]["iters_per_epoch"] = iters_per_epoch
    pre_eval_config["nn"]["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f'device : {pre_eval_config["nn"]["device"]}')

    # set feature names
    feature_names = pre_eval_config["nn"]["feature"]["feature_names"]
    feature_names = (
        feature_names if feature_names != "???" else [x for x in train_feature_df.columns if x.startswith("f_")]
    )
    auxiliary_names = pre_eval_config["nn"]["feature"]["auxiliary_names"]
    auxiliary_names = (
        auxiliary_names
        if auxiliary_names != "???"
        else [x for x in train_feature_df.columns if x.startswith("f_d") or x.startswith("f_t")]
    )
    logger.info(f"feature_names : {feature_names}")
    logger.info(f"auxiliary_names : {auxiliary_names}")

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
        group_values=auxiliary_names,
    )
    pre_eval_config["nn"]["dataset"]["train"]["target_seqs"] = make_sequences(
        df=train_feature_df.query(f"d >= {lower_target_d}"),
        group_key="uid",
        group_values=pre_eval_config["nn"]["feature"]["target_names"],
    )
    logger.debug(f'featuer_seqs:\n\n{pre_eval_config["nn"]["dataset"]["train"]["feature_seqs"][0][:10]}')
    logger.debug(f'auxiliary_seqs:\n\n{pre_eval_config["nn"]["dataset"]["train"]["auxiliary_seqs"][0][:10]}')
    logger.debug(f'target_seqs:\n\n{pre_eval_config["nn"]["dataset"]["train"]["target_seqs"][0][:10]}')

    # valid sequences
    pre_eval_config["nn"]["dataset"]["valid"]["feature_seqs"] = make_sequences(
        df=valid_feature_df.query(f"d < {lower_target_d}"),
        group_key="uid",
        group_values=feature_names,
    )
    pre_eval_config["nn"]["dataset"]["valid"]["auxiliary_seqs"] = make_sequences(
        df=valid_feature_df.query(f"d >= {lower_target_d}"),
        group_key="uid",
        group_values=auxiliary_names,
    )
    pre_eval_config["nn"]["dataset"]["valid"]["target_seqs"] = make_sequences(
        df=valid_feature_df.query(f"d >= {lower_target_d}"),
        group_key="uid",
        group_values=pre_eval_config["nn"]["feature"]["target_names"],
    )

    # model
    pre_eval_config = set_model_config(pre_eval_config, feature_names, auxiliary_names)

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


def judge_best_or_not(best_score: dict | float, eval_score: dict) -> bool:
    if best_score == -np.inf:
        return True

    for k in eval_score.keys():
        if best_score[k] < eval_score[k]:
            return True  # ひとつで評価指標でよくなればOK
    return False


def get_wandb_run_name(config: Config) -> str:
    name = f'{config["/nn/out_dir"]}_{config["/fe/dataset"]}'
    name = f"debug_{name}" if config["global/debug"] else name
    return name


def retransform_regression_target(config: Config, outputs: np.ndarray, data=None) -> np.ndarray:
    if config["/fe/regression_target_transform"] == "log":
        outputs = np.exp(outputs).astype(int)
        outputs[outputs < 1] = 1
        outputs[outputs > 200] = 200
        return outputs

    elif config["/fe/regression_target_transform"] == "mean_diff":
        assert len(data) == len(outputs)
        outputs = outputs + data[["x_mean", "y_mean"]].values
        outputs[outputs < 1] = 1
        outputs[outputs > 200] = 200
        return outputs.astype(int)

    elif config["/fe/regression_target_transform"] == "z_score":
        assert len(data) == len(outputs)
        outputs = (outputs * data[["x_std", "y_std"]].values) + data[["x_mean", "y_mean"]].values
        outputs[outputs < 1] = 1
        outputs[outputs > 200] = 200
        return outputs.astype(int)

    else:
        raise NotImplementedError()


def train_loop(pre_eval_config: dict, train_data: Any, valid_data: Any, loop_name: str, out_dir: Path) -> None:
    # eval config
    config = set_config(pre_eval_config, train_data, valid_data)
    wandb.login(key=os.environ["WANDB_KEY"])  # need wandb account

    # model setting
    metrics = config["/nn/metrics"]
    model = config["/nn/model"]
    train_dataloader = config["/nn/dataloader/train"]
    valid_dataloader = config["/nn/dataloader/valid"]
    criterion = config["/nn/criterion"]
    optimizer = config["/nn/optimizer"]
    scheduler = config["/nn/scheduler"]
    max_epochs = pre_eval_config["nn"]["max_epochs"]

    logger.debug(f"len_traindataloader : {len(train_dataloader)}, len_validdataloader : {len(valid_dataloader)}")

    # setup
    wandb.init(
        project=config["/global/project"],
        name=get_wandb_run_name(config),
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
        logger.debug(f'outputs: {va_output["outputs"].shape}')
        with logger.time_log("calc metrics"):
            eval_score = metrics(
                output=retransform_regression_target(
                    config=config, outputs=va_output["outputs"], data=valid_data.query("d >= 60")
                ),
                target=valid_data.query("d >= 60")[["original_x", "original_y"]].to_numpy(),
                info=valid_data[["d", "t"]].query("d >= 60").to_numpy(),
            )

        # logs
        logs = {
            "epoch": epoch,
            "train_loss_epoch": loss.item(),
            "valid_loss_epoch": va_output["loss"].item(),
        }
        if isinstance(eval_score, dict):
            logs = {**logs, **eval_score}  # update
        else:
            logs["eval_score"] = eval_score

        logger.info(logs)
        wandb.log(logs)

        if judge_best_or_not(best_score=best_score, eval_score=eval_score):
            best_score = eval_score
            logger.info(f"epoch {epoch} - best score: {best_score} model 🌈")

            torch.save(model.state_dict(), out_dir / f"{loop_name}.pth")  # save model weight
            joblib.dump(va_output, out_dir / f"{loop_name}.pkl")  # save outputs

    wandb.finish(quiet=True)
    torch.cuda.empty_cache()
    gc.collect()
    best_val_outputs = joblib.load(out_dir / f"{loop_name}.pkl")
    return best_val_outputs


def train_fold(pre_eval_config: dict, df: pd.DataFrame, out_dir: Path) -> None:
    num_fold = pre_eval_config["cv"]["num_fold"]
    valid_folds = pre_eval_config["cv"]["valid_folds"]
    oof_outputs = []
    for i_fold in range(num_fold):
        if i_fold not in valid_folds:
            continue

        with logger.time_log(f"fold {i_fold}"):
            train_feature_df = df[df["fold"] != i_fold].reset_index(drop=True)
            valid_feature_df = df[df["fold"] == i_fold].reset_index(drop=True)

            logger.debug(f"train_feature: {train_feature_df.shape}, valid_feature: {valid_feature_df.shape}")

            best_outputs = train_loop(
                pre_eval_config=pre_eval_config,
                train_data=train_feature_df,
                valid_data=valid_feature_df,
                loop_name=f"fold_{i_fold}",
                out_dir=out_dir,
            )
            oof_outputs.append(best_outputs["outputs"])
    return np.concatenate(oof_outputs, axis=0)


def run() -> None:
    pre_eval_config = load_yaml()
    seed_everything(pre_eval_config["global"]["seed"])
    out_dir = (
        Path(pre_eval_config["global"]["resources"])
        / "output"
        / pre_eval_config["nn"]["out_dir"]
        / pre_eval_config["fe"]["dataset"]
    )

    feature_df = load_feature_df(pre_eval_config=pre_eval_config, name="train_feature_df")
    with logger.time_log("train_fold"):
        oof_outputs = train_fold(pre_eval_config, df=feature_df, out_dir=out_dir)
    joblib.dump(oof_outputs, out_dir / "oof_outputs.pkl")


if __name__ == "__main__":
    run()
