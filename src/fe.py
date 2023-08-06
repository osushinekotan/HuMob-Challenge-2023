import hashlib
from functools import cached_property, wraps
from pathlib import Path
from typing import Callable

import joblib
import pandas as pd
import torch
from custom.config_types import CONFIG_TYPES
from logger import Logger
from pytorch_pfn_extras.config import Config
from util import load_yaml, reduce_mem_usage, sort_df_numpy

logger = Logger(name="fe")

# set config
pre_eval_config = load_yaml()
config = Config(pre_eval_config, types=CONFIG_TYPES)

# set const
DEBUG = config["/global/debug"]


class TaskDatset:
    def __init__(self, config, overwrite=False) -> None:
        self.config = config
        self.dirpath = Path(config["/global/resources"]) / "input"
        self.dataset_name = config["/fe/dataset"]

        self.raw_train_filepath = self.dirpath / f"{self.dataset_name}_raw_train.parquet"
        self.raw_test_filepath = self.dirpath / f"{self.dataset_name}_raw_test.parquet"

        self.overwrite = overwrite

    @property
    def raw_train_data(self):
        if self.raw_train_filepath.is_file() and (not self.overwrite):
            return pd.read_parquet(self.raw_train_filepath)

        uids = self.raw_data.query("x != 999")["uid"].unique()
        raw_train_df = self.raw_data[self.raw_data["uid"].isin(uids)].reset_index(drop=True)
        raw_train_df.to_parquet(self.raw_train_filepath)
        return raw_train_df

    @property
    def raw_test_data(self):
        if self.raw_test_filepath.is_file() and (not self.overwrite):
            return pd.read_parquet(self.raw_test_filepath)

        uids = self.raw_data.query("x == 999")["uid"].unique()
        raw_test_df = self.raw_data[self.raw_data["uid"].isin(uids)].reset_index(drop=True)
        raw_test_df.to_parquet(self.raw_test_filepath)
        return raw_test_df

    @cached_property
    def raw_data(self):
        return read_parquet_from_csv(
            filepath=self.dirpath / f"{self.dataset_name}.csv.gz",
            dirpath=self.dirpath,
            process_fns=[reduce_mem_usage, sort_df_numpy],
            overwrite=self.config["/fe/overwrite"],
        )

    @property
    def poi_data(self):
        return read_parquet_from_csv(filepath=self.dirpath / "cell_POIcat.csv.gz", dirpath=self.dirpath)


def read_parquet_from_csv(
    filepath: Path,
    dirpath: Path,
    process_fns: list[Callable] | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    name = filepath.name.split(".")[0]
    parquet_filepath = dirpath / f"{name}.parquet"
    if parquet_filepath.is_file() and (not overwrite):
        logger.info(f"load parquet file ({str(filepath)})")
        return pd.read_parquet(parquet_filepath)

    logger.info(f"load csv & convert to parquet ({str(filepath)})")
    df = pd.read_csv(filepath)

    if process_fns is not None:
        for fn in process_fns:
            logger.info(f"excute {fn.__name__}")
            df = fn(df)

    df.to_parquet(parquet_filepath)
    return df


def cache(out_dir: Path, overwrite: bool = False, no_cache: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if no_cache:
                return func(*args, **kwargs)

            extractor = args[1]

            extractor_name = extractor.__class__.__name__
            hash_input = extractor_name + str(extractor.__dict__)

            # use hash
            extractor_id = hashlib.sha256(hash_input.encode()).hexdigest()
            filename = f"{extractor_name}_{extractor_id}"
            cache_file = out_dir / f"{filename}.pkl"

            if cache_file.exists() and not overwrite:
                logger.debug(f"use cache : {filename}")
                result = joblib.load(cache_file)
            else:
                result = func(*args, **kwargs)
                joblib.dump(result, cache_file)
            return result

        return wrapper

    return decorator


def make_features(config, df, overwrite=False):
    extractors = config["fe/extractors"]
    out_dir = Path(config["/global/resources"]) / "output" / config["fe/out_dir"]

    @cache(out_dir=out_dir, overwrite=overwrite)
    def _extract(df, extractor):
        with logger.time_log(target=extractor.__class__.__name__):
            return extractor(df)

    features_df = pd.concat([df] + [_extract(df, extractor) for extractor in extractors], axis=1)
    return features_df


# load data
task_dataset = TaskDatset(config=config, overwrite=True)
raw_train_df = task_dataset.raw_train_data
poi_df = task_dataset.poi_data

if DEBUG:
    user_ids = raw_train_df["uid"].sample(100, random_state=config["/global/seed"]).tolist()
    raw_train_df = raw_train_df[raw_train_df["uid"].isin(user_ids)].reset_index(drop=True)

# feature engineering
train_df = make_features(config=config, df=raw_train_df, overwrite=True)


def make_sequences(df: pd.DataFrame, group_key: str, group_values: list[str]):
    grouped = df.groupby(group_key, sort=False)
    sequences = [torch.tensor(group[group_values].to_numpy()) for _, group in grouped]
    return sequences


# feature_names = [x for x in train_df.columns if x.startswith("f_")]
feature_seqs = make_sequences(df=train_df, group_key="uid", group_values=["d", "t"])
auxiliary_seqs = make_sequences(
    df=train_df.query("d >= 60"), group_key="uid", group_values=["d", "t"]
)  # features for prediction zone

target_seqs = make_sequences(
    df=train_df.query("d >= 60"),
    group_key="uid",
    group_values=["x", "y"],
)  # target is x & y over 60 zone

assert len(feature_seqs) == len(target_seqs) == len(auxiliary_seqs)
print("OK")
