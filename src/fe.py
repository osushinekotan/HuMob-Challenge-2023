from pathlib import Path
from typing import Callable

import pandas as pd
from custom.config_types import CONFIG_TYPES
from logger import Logger
from pytorch_pfn_extras.config import Config
from util import load_yaml, reduce_mem_usage, sort_df_numpy

logger = Logger(name="fe")

pre_eval_config = load_yaml()
config = Config(pre_eval_config, types=CONFIG_TYPES)

RESOURCES = Path(config["/global/resources"])
INPUT = RESOURCES / "input"

DATESET = config["/fe/dataset"]
DEBUG = config["/global/debug"]


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


# feature engineering
def make_features(config, df):
    extractors = config["fe/extractors"]

    def _extract(df, extractor):
        with logger.time_log(target=extractor.__class__.__name__):
            return extractor(df)

    features_df = pd.concat([df] + [_extract(df, extractor) for extractor in extractors], axis=1)
    return features_df


# load data
raw_train_df = read_parquet_from_csv(
    filepath=INPUT / f"{DATESET}.csv.gz",
    dirpath=INPUT,
    process_fns=[reduce_mem_usage, sort_df_numpy],
    overwrite=config["/fe/overwrite"],
)
poi_df = read_parquet_from_csv(filepath=INPUT / "cell_POIcat.csv.gz", dirpath=INPUT)

if DEBUG:
    user_ids = raw_train_df["uid"].sample(100, random_state=config["/global/seed"]).tolist()
    raw_train_df = raw_train_df[raw_train_df["uid"].isin(user_ids)].reset_index(drop=True)

train_df = make_features(config=config, df=raw_train_df)
