import hashlib
from functools import cached_property, wraps
from pathlib import Path
from typing import Callable

import joblib
import pandas as pd
from custom.config_types import CONFIG_TYPES
from logger import Logger
from pytorch_pfn_extras.config import Config
from util import load_yaml, reduce_mem_usage, sort_df_numpy

logger = Logger(name="fe")


class TaskDatset:
    def __init__(self, config: Config, overwrite: bool = False) -> None:
        self.config = config
        self.dirpath = Path(config["/global/resources"]) / "input"
        self.dataset_name = config["/fe/dataset"]

        self.raw_train_filepath = self.dirpath / f"{self.dataset_name}_raw_train.parquet"
        self.raw_test_filepath = self.dirpath / f"{self.dataset_name}_raw_test.parquet"

        self.overwrite = overwrite

    @property
    def raw_train_data(self) -> pd.DataFrame:
        if self.raw_train_filepath.is_file() and (not self.overwrite):
            logger.info(f"read_parquet : {self.raw_train_filepath}")
            return pd.read_parquet(self.raw_train_filepath)

        with logger.time_log(target="raw_train_data"):
            uids = self.raw_data.query("x == 999")["uid"].unique()
            raw_train_df = self.raw_data[~self.raw_data["uid"].isin(uids)].reset_index(drop=True)
            raw_train_df.to_parquet(self.raw_train_filepath)

        return raw_train_df

    @property
    def raw_test_data(self) -> pd.DataFrame:
        if self.raw_test_filepath.is_file() and (not self.overwrite):
            logger.info(f"read_parquet : {self.raw_test_filepath}")
            return pd.read_parquet(self.raw_test_filepath)

        with logger.time_log(target="raw_test_data"):
            uids = self.raw_data.query("x == 999")["uid"].unique()
            raw_test_df = self.raw_data[self.raw_data["uid"].isin(uids)].reset_index(drop=True)
            raw_test_df.to_parquet(self.raw_test_filepath)
        return raw_test_df

    @cached_property
    def raw_data(self) -> pd.DataFrame:
        return read_parquet_from_csv(
            filepath=self.dirpath / f"{self.dataset_name}.csv.gz",
            dirpath=self.dirpath,
            process_fns=[reduce_mem_usage, sort_df_numpy],
            overwrite=False,
        )

    @property
    def poi_data(self) -> pd.DataFrame:
        return read_parquet_from_csv(filepath=self.dirpath / "cell_POIcat.csv.gz", dirpath=self.dirpath)


def convert_debug_train_df(df: pd.DataFrame, n_uids: int = 100, random_state: int | None = None):
    user_ids = df["uid"].sample(n_uids, random_state=random_state).tolist()
    debug_df = df[df["uid"].isin(user_ids)].reset_index(drop=True)
    return debug_df


def read_parquet_from_csv(
    filepath: Path,
    dirpath: Path,
    process_fns: list[Callable] | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    name = filepath.name.split(".")[0]
    parquet_filepath = dirpath / f"{name}.parquet"
    if parquet_filepath.is_file() and (not overwrite):
        logger.info(f"read_parquet : ({str(filepath)})")
        return pd.read_parquet(parquet_filepath)

    logger.info(f"load csv & convert to parquet ({str(filepath)})")
    df = pd.read_csv(filepath)

    if process_fns is not None:
        for fn in process_fns:
            logger.info(f"excute {fn.__name__}")
            df = fn(df)

    df = df.reset_index(drop=True)
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


def make_features(
    config: Config,
    df: pd.DataFrame,
    overwrite: bool = False,
    no_cache: bool = False,
    name: str = "",
) -> pd.DataFrame:
    extractors = config["/fe/extractors"]
    dataset_name = config["/fe/dataset"]

    # set dir
    out_dir = Path(config["/global/resources"]) / "output" / config["fe/out_dir"] / dataset_name
    logger.debug(f"make_features: overwrite={overwrite}, no_cache={no_cache}")

    # cache for final output
    filepath_for_features_df = out_dir / f"{name}.pkl"
    if filepath_for_features_df.is_file() and (not overwrite):
        logger.info(f"load : {filepath_for_features_df}")
        return joblib.load(filepath_for_features_df)

    # feature engineering
    @cache(out_dir=out_dir, overwrite=overwrite, no_cache=no_cache)
    def _extract(df, extractor):
        with logger.time_log(target=extractor.__class__.__name__):
            return extractor(df)

    features_df = pd.concat([df] + [_extract(df, extractor) for extractor in extractors], axis=1)
    joblib.dump(features_df, filepath_for_features_df)
    return features_df


def add_fold_index(config: Config, df: pd.DataFrame) -> pd.DataFrame:
    with logger.time_log("add_fold"):
        df["fold"] = -1
        cv = config["/cv/strategy"]
        for i_fold, (tr_idx, va_idx) in enumerate(cv.split(X=df, y=df["fold"], groups=df["uid"])):
            df.loc[va_idx, "fold"] = i_fold
    return df


def run() -> None:
    # set config
    pre_eval_config = load_yaml()
    config = Config(pre_eval_config, types=CONFIG_TYPES)

    # set const
    DEBUG = config["/global/debug"]
    DEBUG_N_UIDS = 100

    # load data
    task_dataset = TaskDatset(config=config, overwrite=True)
    raw_train_df = task_dataset.raw_train_data
    raw_test_df = task_dataset.raw_test_data

    if DEBUG:
        raw_train_df = convert_debug_train_df(
            df=raw_train_df,
            n_uids=DEBUG_N_UIDS,
            random_state=config["/global/seed"],
        )
        raw_test_df = convert_debug_train_df(
            df=raw_test_df,
            n_uids=100,
            random_state=config["/global/seed"],
        )

    # add fold index
    raw_train_df = add_fold_index(config=config, df=raw_train_df)

    # feature engineering
    train_feature_df = make_features(
        config=config,
        df=raw_train_df,
        overwrite=True,
        name="train_feature_df",
    )
    test_feature_df = make_features(
        config=config,
        df=raw_test_df,
        overwrite=True,
        name="test_feature_df",
    )

    # check
    logger.debug(f"train_feature_df : {train_feature_df.shape}, test_feature_df : {test_feature_df.shape}")
    logger.debug(f"train_uids : {train_feature_df['uid'].nunique()}, test_uids : {test_feature_df['uid'].nunique()}")

    assert len(train_feature_df.query("x == 999")) == 0
    assert test_feature_df.query("x == 999")["uid"].nunique() == test_feature_df["uid"].nunique()
    logger.debug(f"\nfeatures\n\n{train_feature_df}")


if __name__ == "__main__":
    run()
