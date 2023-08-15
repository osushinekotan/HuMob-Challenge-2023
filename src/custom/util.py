import gc

import numpy as np
import pandas as pd


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
