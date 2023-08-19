import hashlib
from functools import cached_property, wraps
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import polars as pl
from custom.config_types import CONFIG_TYPES
from custom.util import generate_adjacency_matrix, sort_df_numpy
from logger import Logger
from pytorch_pfn_extras.config import Config
from tqdm import tqdm
from util import load_yaml, reduce_mem_usage

logger = Logger(name="fe")
tqdm.pandas()


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

    @property
    def mesh_map(self):
        x = list(range(1, 201))
        y = list(range(1, 201))
        mesh_map_df = pd.DataFrame(
            {
                "x": np.repeat(x, len(x)),
                "y": np.tile(y, len(y)),
                "mesh_id": range(len(x) * len(y)),
            }
        )
        return mesh_map_df


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


def add_original_raw_targets(df: pd.DataFrame) -> pd.DataFrame:
    df["original_x"] = df["x"].copy()
    df["original_y"] = df["y"].copy()
    return df


def transform_regression_target(config: Config, df: pd.DataFrame) -> pd.DataFrame:
    target_mask = np.array(df["original_x"] != 999, dtype=bool)
    if config["/fe/regression_target_transform"] == "log":
        df.loc[target_mask, ["x", "y"]] = np.log(df.loc[target_mask, ["original_x", "original_y"]].to_numpy())
        return df

    else:
        raise NotImplementedError()


def make_features(
    config: Config,
    df: pd.DataFrame,
    overwrite: bool = False,
    no_cache: bool = False,
) -> pd.DataFrame:
    extractors = config["/fe/extractors"]
    dataset_name = config["/fe/dataset"]

    # set dir
    out_dir = Path(config["/global/resources"]) / "output" / config["fe/out_dir"] / dataset_name
    logger.debug(f"make_features: overwrite={overwrite}, no_cache={no_cache}")

    # feature engineering
    @cache(out_dir=out_dir, overwrite=overwrite, no_cache=no_cache)
    def _extract(df, extractor):
        with logger.time_log(target=extractor.__class__.__name__):
            return reduce_mem_usage(extractor(df))

    return pd.concat([df] + [_extract(df, extractor) for extractor in extractors], axis=1)


def save_features(config, features_df, name):
    # set dir
    out_dir = Path(config["/global/resources"]) / "output" / config["fe/out_dir"] / config["/fe/dataset"]
    filepath_for_features_df = out_dir / f"{name}.pkl"
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(features_df, filepath_for_features_df)
    logger.info(f"complete save_features! ({filepath_for_features_df})")
    return features_df


def add_fold_index(config: Config, df: pd.DataFrame) -> pd.DataFrame:
    with logger.time_log("add_fold"):
        df["fold"] = -1
        cv = config["/cv/strategy"]
        for i_fold, (tr_idx, va_idx) in enumerate(cv.split(X=df, y=df["fold"], groups=df["uid"])):
            df.loc[va_idx, "fold"] = i_fold
        logger.info(f"df : {df.shape}")
    return df


def make_poi_cat_count_pivot_table(config, df):
    poi_cat_cnt_df = pd.pivot_table(df, columns=["POIcategory"], index=["x", "y"]).fillna(0)
    poi_cat_cnt_df.columns = [f"POIcat_{x[1]:02}_count" for x in poi_cat_cnt_df.columns]

    decomposer = config["/fe/poi_decomposer"]
    decomposed_poi_df = pd.DataFrame(
        decomposer.fit_transform(poi_cat_cnt_df.to_numpy()),
        columns=[f"POI_d{x:02}" for x in range(decomposer.n_components)],
    )
    return pd.concat(
        [
            poi_cat_cnt_df.astype(np.int16).reset_index(),
            decomposed_poi_df,
        ],
        axis=1,
    )


def join_poi_to_task_data_batched(config, task_df, poi_df, batch_size=1000):
    pivot_df_pl = pl.from_pandas(make_poi_cat_count_pivot_table(config, poi_df))
    n = len(task_df)

    return (
        pd.concat(
            [
                pl.from_pandas(task_df.iloc[i : i + batch_size])
                .join(pivot_df_pl, on=["x", "y"], how="left")
                .to_pandas()
                for i in tqdm(range(0, n, batch_size), desc="join poi")
            ],
            axis=0,
        )
        .fillna(0)
        .reset_index(drop=True)
    )


def add_poi_features(config, df, poi_df, batch_size=100000):
    merged_df = join_poi_to_task_data_batched(config=config, task_df=df, poi_df=poi_df, batch_size=batch_size)
    merged_df.columns = [f"f_{x}" if x.startswith("POIcat_") else x for x in merged_df.columns]

    node_colmns = [x for x in merged_df.columns if x.startswith("f_")]
    node_df = merged_df[node_colmns].copy()
    node_df.columns = [f"fn{x[1:]}" for x in node_df.columns]  # central node feature prefix : fn_

    # reduce mem
    merged_df = reduce_mem_usage(merged_df)
    node_df = reduce_mem_usage(node_df)

    return pd.concat([merged_df, node_df], axis=1)


def scaling(config, train_feature_df, test_feature_df):
    n = len(train_feature_df)

    all_df = pd.concat([train_feature_df, test_feature_df]).reset_index()
    feature_cols = [x for x in all_df.columns if x.startswith("f_")]
    nofeature_cols = [x for x in all_df.columns if not x.startswith("f_")]

    assert sorted(feature_cols + nofeature_cols) == sorted(all_df.columns)

    scaler = config["/fe/scaling"]
    scaled_df = pd.DataFrame(scaler.fit_transform(all_df[feature_cols]), columns=feature_cols)

    all_df = pd.concat([all_df[nofeature_cols], scaled_df], axis=1)
    scaled_train_feature_df = all_df.iloc[:n].reset_index(drop=True)
    scaled_test_feature_df = all_df[n:].reset_index(drop=True)

    assert len(train_feature_df) == len(scaled_train_feature_df)
    assert len(test_feature_df) == len(scaled_test_feature_df)

    return scaled_train_feature_df.fillna(0), scaled_test_feature_df.fillna(0)


def assign_d_cycle_number(config, df):
    cycles = config["/fe/cycles"]
    for cycle in cycles:
        df[f"cycle_{cycle:02}"] = np.array([x // cycle for x in df["d"]], dtype=np.int16)
    return df


def get_neighbor_mesh_id(mesh_id, adj_matrix, padding_value=-1, max_neighbor_num=8):
    neighbor_mesh_ids = np.nonzero(adj_matrix[mesh_id])[0].tolist()
    if (d := max_neighbor_num - len(neighbor_mesh_ids)) > 0:
        return neighbor_mesh_ids + [padding_value] * d
    return neighbor_mesh_ids


def make_mesh_map_df(adj_matrix, padding_value=-1):
    x = list(range(1, 201))
    y = list(range(1, 201))
    mesh_map_df = pd.DataFrame(
        {
            "x": np.repeat(x, len(x)),
            "y": np.tile(y, len(y)),
            "mesh_id": range(len(x) * len(y)),
        }
    )
    max_neighbor_num = adj_matrix.sum(axis=1).max()
    mesh_map_df["neighbor_mesh_ids"] = mesh_map_df["mesh_id"].progress_apply(
        lambda m: get_neighbor_mesh_id(m, adj_matrix, padding_value=padding_value, max_neighbor_num=max_neighbor_num)
    )
    return mesh_map_df


def make_node_features(mesh_map_df, poi_df):
    poi_cat_cnt_df = pd.pivot_table(poi_df, columns=["POIcategory"], index=["x", "y"]).fillna(0)
    poi_cat_cnt_df.columns = [f"POIcat_{x[1]:02}_count" for x in poi_cat_cnt_df.columns]
    node_feature = (
        pd.merge(
            mesh_map_df.drop("neighbor_mesh_ids", axis=1),
            poi_cat_cnt_df.reset_index(),
            on=["x", "y"],
            how="left",
        )
        .drop(["x", "y", "mesh_id"], axis=1)
        .fillna(0)
        .astype(np.int16)
        .to_numpy()
    )
    return node_feature


def load_features(config, name):
    # set dir
    out_dir = Path(config["/global/resources"]) / "output" / config["fe/out_dir"] / config["/fe/dataset"]
    filepath_for_features_df = out_dir / f"{name}.pkl"
    return joblib.load(filepath_for_features_df)


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
    poi_df = task_dataset.poi_data
    mesh_map_df = task_dataset.mesh_map

    # adjacency node
    with logger.time_log("make_mesh_map_df"):
        adj_matrix = generate_adjacency_matrix(n=200)
        mesh_map_df = make_mesh_map_df(adj_matrix, padding_value=config["/global/padding_value"])
        save_features(config, features_df=mesh_map_df, name="mesh_map_df")

        node_features = make_node_features(mesh_map_df, poi_df=poi_df)
        save_features(config, features_df=node_features, name="node_features")  # NOTE : not df

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

    # add mesh_id
    raw_train_df = pd.merge(raw_train_df, mesh_map_df, how="left", on=["x", "y"])
    raw_test_df = pd.merge(raw_test_df, mesh_map_df, how="left", on=["x", "y"])

    # add POIcat features (f_*)
    if config["/fe/use_poi_features"]:
        with logger.time_log("make poi features"):
            raw_train_df = add_poi_features(config=config, df=raw_train_df, poi_df=poi_df)
            raw_test_df = add_poi_features(config=config, df=raw_test_df, poi_df=poi_df)

    # assign cycle number
    raw_train_df = assign_d_cycle_number(config, df=raw_train_df)
    raw_test_df = assign_d_cycle_number(config, df=raw_test_df)

    # copy original target
    raw_train_df = add_original_raw_targets(raw_train_df)
    raw_test_df = add_original_raw_targets(raw_test_df)

    # add fold index
    raw_train_df = add_fold_index(config=config, df=raw_train_df)

    if config["/fe/overwrite"]:
        # target enginineering
        train_feature_df = transform_regression_target(config=config, df=raw_train_df)
        test_feature_df = transform_regression_target(config=config, df=raw_test_df)

        # feature engineering
        train_feature_df = make_features(
            config=config,
            df=raw_train_df,
            overwrite=True,
        )
        test_feature_df = make_features(
            config=config,
            df=raw_test_df,
            overwrite=True,
        )

        # scaling
        train_feature_df, test_feature_df = scaling(
            config=config,
            train_feature_df=train_feature_df,
            test_feature_df=test_feature_df,
        )

        # save features
        save_features(config=config, features_df=train_feature_df, name="train_feature_df")
        save_features(config=config, features_df=test_feature_df, name="test_feature_df")

    else:
        train_feature_df = load_features(config=config, name="train_feature_df")
        test_feature_df = load_features(config=config, name="test_feature_df")

    # check
    logger.debug(f"train_feature_df : {train_feature_df.shape}, test_feature_df : {test_feature_df.shape}")
    logger.debug(f"train_uids : {train_feature_df['uid'].nunique()}, test_uids : {test_feature_df['uid'].nunique()}")

    assert len(train_feature_df.query("x == 999")) == 0
    assert test_feature_df.query("x == 999")["uid"].nunique() == test_feature_df["uid"].nunique()
    logger.debug(f"\nfeatures\n\n{train_feature_df}")


if __name__ == "__main__":
    run()
