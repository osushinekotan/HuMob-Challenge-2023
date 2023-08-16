import gc

import numpy as np
import pandas as pd
from tqdm import tqdm


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


def get_neighbors(i, j, n):
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    neighbors = []
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < n and 0 <= nj < n:
            neighbors.append(nj * n + ni)
    return neighbors


def generate_adjacency_matrix(n=200):
    adj_matrix = np.zeros((n * n, n * n), dtype=int)

    for i in tqdm(range(n)):
        for j in range(n):
            mesh_id = j * n + i
            for neighbor_id in get_neighbors(i, j, n):
                adj_matrix[mesh_id][neighbor_id] = 1

    return adj_matrix


def get_kth_adjacency(adj_matrix, k):
    if k >= 2:
        kth_adj = np.linalg.matrix_power(adj_matrix, k)
        kth_adj[kth_adj > 1] = 1
        return kth_adj
    return adj_matrix
