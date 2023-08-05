import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_yaml(filepath: Path = "/workspace/src/conf/custom.yaml") -> dict:
    """load yaml as dict

    Args:
        filepath (Path, optional): yaml filepath. Defaults to "/workspace/src/conf/custom.yaml", config file.

    Returns:
        dict: yaml as dict
    """
    with open(filepath) as f:
        return yaml.safe_load(f)


def unzip(zip_filepath: Path, extract_dir: Path | None = None) -> None:
    if extract_dir is None:
        extract_dir = Path(zip_filepath.parent)
    output_dir = extract_dir / zip_filepath.stem
    if not output_dir.is_dir():
        shutil.unpack_archive(zip_filepath, extract_dir)


def save_json(filepath: Path, json_file: dict) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            json_file,
            f,
            indent=4,
            sort_keys=True,
            ensure_ascii=False,
        )


def load_json(filepath: Path) -> dict:
    with open(filepath) as f:
        dic = json.load(f)
    return dic


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    if verbose:
        cols = df.columns[df.columns.duplicated()]
        if len(cols) == 0:
            print("There are no duplicated columns")
        else:
            print(f"duplicated culumns are {df.columns[df.columns.duplicated()]}")
    df = df.loc[:, ~df.columns.duplicated()]  # TODO : ダブってるカラムを見つける
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    dfs = []
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    dfs.append(df[col].astype(np.int8))
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    dfs.append(df[col].astype(np.int16))
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    dfs.append(df[col].astype(np.int32))
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    dfs.append(df[col].astype(np.int64))
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    dfs.append(df[col].astype(np.float16))
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    dfs.append(df[col].astype(np.float32))
                else:
                    dfs.append(df[col].astype(np.float64))
        else:
            dfs.append(df[col])

    df_out = pd.concat(dfs, axis=1)
    if verbose:
        end_mem = df_out.memory_usage().sum() / 1024**2
        num_reduction = str(100 * (start_mem - end_mem) / start_mem)
        print(f"Mem. usage decreased to {str(end_mem)[:3]}Mb:  {num_reduction[:2]}% reduction")  # noqa
    return df_out


def sort_df_numpy(df):
    # get columns
    uid_col = "uid"
    d_col = "d"
    t_col = "t"

    # get columns index
    uid_idx = df.columns.get_loc(uid_col)
    d_idx = df.columns.get_loc(d_col)
    t_idx = df.columns.get_loc(t_col)

    # to numpy from pandas
    arr = df.values
    cols = df.columns
    del df
    gc.collect()

    # calc uid
    unique_uid, counts_uid = np.unique(arr[:, uid_idx], return_counts=True)
    nunique_uid_dict = dict(zip(unique_uid, counts_uid))
    nunique_uid = np.vectorize(nunique_uid_dict.get)(arr[:, uid_idx])

    # concat sort target
    sorting_array = np.column_stack((nunique_uid, arr))

    # sort
    sorted_array = sorting_array[
        np.lexsort(
            (
                sorting_array[:, t_idx + 1],
                sorting_array[:, d_idx + 1],
                sorting_array[:, uid_idx + 1],
                sorting_array[:, 0],
            )
        )
    ]
    sorted_df = pd.DataFrame(sorted_array[:, 1:], columns=cols)
    return sorted_df


def sort_df(df):
    return (
        df.assign(nunique_uid=df["uid"].map(df["uid"].value_counts()))
        .sort_values(["nunique_uid", "uid", "d", "t"])
        .reset_index(drop=True)
        .drop("nunique_uid", axis=1)
    )
