import hashlib
from functools import cached_property, wraps
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
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
    def __init__(self, config) -> None:
        self.config = config
        self.dirpath = Path(config["/global/resources"]) / "input"
        self.dataset_name = config["/fe/dataset"]

        self.raw_train_filepath = self.dirpath / f"{self.dataset_name}_raw_train.parquet"
        self.raw_test_filepath = self.dirpath / f"{self.dataset_name}_raw_test.parquet"

    @property
    def raw_train_data(self):
        if self.raw_train_filepath.is_file():
            return pd.read_parquet(self.raw_train_filepath)

        raw_train_df = self.raw_data.query("x != 999").reset_index(drop=True)
        raw_train_df.to_parquet(self.raw_train_filepath)
        return raw_train_df

    @property
    def raw_test_data(self):
        if self.raw_test_filepath.is_file():
            return pd.read_parquet(self.raw_test_filepath)

        raw_test_df = self.raw_data.query("x == 999")
        raw_test_df.to_parquet(self.raw_test_filepath).reset_index(drop=True)
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


def make_features(config, df):
    extractors = config["fe/extractors"]
    out_dir = Path(config["/global/resources"]) / "output" / config["fe/out_dir"]

    @cache(out_dir=out_dir, overwrite=config["/fe/overwrite"])
    def _extract(df, extractor):
        with logger.time_log(target=extractor.__class__.__name__):
            return extractor(df)

    features_df = pd.concat([df] + [_extract(df, extractor) for extractor in extractors], axis=1)
    return features_df


class TrainValidDataset:
    def __init__(self, config, uids):
        self.config = config
        out_dir = Path(config["/global/resources"]) / "output" / config["fe/out_dir"]
        self.train_filepath = out_dir / "train_feaures_df.pkl"
        self.valid_filepath = out_dir / "valid_features_df.pkl"

        self.uids = uids

    @cached_property
    def valid_uids(self):
        valid_uids = (
            pd.Series(np.unique(self.uids))
            .sample(self.config["/cv/n_valid_uids"], random_state=self.config["/global/seed"])
            .tolist()
        )
        return valid_uids

    def load_valid_data(self, df):
        if self.valid_filepath.is_file():
            return joblib.load(self.train_filepath)

        valid_df = df[df["uid"].isin(self.valid_uids)].reset_index(drop=True)
        joblib.dump(valid_df, self.valid_filepath)
        return valid_df

    def load_train_data(self, df):
        if self.train_filepath.is_file():
            return joblib.load(self.train_filepath)

        train_df = df[~df["uid"].isin(self.valid_uids)].reset_index(drop=True)
        joblib.dump(train_df, self.train_filepath)
        return train_df


# load data
task_dataset = TaskDatset(config=config)
raw_train_df = task_dataset.raw_train_data
poi_df = task_dataset.poi_data

if DEBUG:
    user_ids = raw_train_df["uid"].sample(100, random_state=config["/global/seed"]).tolist()
    raw_train_df = raw_train_df[raw_train_df["uid"].isin(user_ids)].reset_index(drop=True)

# feature engineering
train_df = make_features(config=config, df=raw_train_df)
train_valid_dataset = TrainValidDataset(config=config, uids=train_df["uids"])
valid_df = train_valid_dataset.load_valid_data(df=train_df)
train_df = train_valid_dataset.load_train_data(df=train_df)


print("OK")
