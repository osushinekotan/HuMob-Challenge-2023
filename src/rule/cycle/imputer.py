import os
from itertools import product

import pandas as pd
from pandarallel import pandarallel

os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

pandarallel.initialize(progress_bar=True, use_memory_fs=False)


class CycleImputer:
    def __init__(
        self,
        group_keys=["uid", "dayofweek", "t"],
        agg_method="median",
    ):
        self.group_keys = group_keys
        self.agg_method = agg_method

        self.agg_df = None
        self.base_df = None
        self.agg_cols = [f"x_{agg_method}", f"y_{agg_method}"]

    def aggregate(self, df, d=60):
        self.agg_df = df[df["d"] < d].reset_index(drop=True).groupby(self.group_keys)[["x", "y"]].agg([self.agg_method])
        self.agg_df.columns = [f"{x[0]}_{x[1]}" for x in self.agg_df.columns]
        self.agg_df.reset_index(inplace=True)

    def make_base(self, df):
        lists = [range(48) if x == "t" else df[x].unique() for x in self.group_keys]

        self.base_df = self.create_combinations_df(lists, columns=self.group_keys)

        if self.agg_df is None:
            self.aggregate(df)

        self.base_df = pd.merge(self.base_df, self.agg_df, on=self.group_keys, how="left")
        self.base_df = self.base_df.sort_values(self.group_keys).reset_index(drop=True)

    def impute(self, df, cycle_groups=[["uid", "dayofweek"]], T=2):
        # TODO: to bidirectional

        if self.base_df is None:
            self.make_base(df)

        fill_missing = lambda x: self.fill_missing(group=x, T=T)  # noqa
        df_filled = self.base_df.copy()

        for cycle_group in cycle_groups:
            if df_filled[f"x_{self.agg_method}"].isnull().sum() == 0:
                return df_filled
            print(f"{cycle_group}")
            df_filled = df_filled.groupby(cycle_group, sort=False).parallel_apply(fill_missing).reset_index(drop=True)
            df_filled = df_filled.sort_values(self.group_keys, ascending=False).reset_index(
                drop=True
            )  # reverse direction
            df_filled = df_filled.groupby(cycle_group, sort=False).parallel_apply(fill_missing).reset_index(drop=True)

        # 各uidの集計
        uid_agg_df = self.base_df.groupby("uid")[[f"x_{self.agg_method}", f"y_{self.agg_method}"]].agg(self.agg_method)

        # uidをインデックスとしてセット
        df_filled.set_index("uid", inplace=True)

        # 残った欠損値を各uidの平均で埋める
        df_filled[f"x_{self.agg_method}"].fillna(uid_agg_df[f"x_{self.agg_method}"], inplace=True)
        df_filled[f"y_{self.agg_method}"].fillna(uid_agg_df[f"y_{self.agg_method}"], inplace=True)

        assert df_filled[f"x_{self.agg_method}"].isnull().sum() == 0
        return df_filled.reset_index()

    def fill_missing(self, group, T=2):
        # T期以内の連続の欠損をt-1期の値で埋める
        group_filled = group.copy()
        for col in [f"x_{self.agg_method}", f"y_{self.agg_method}"]:
            missing_streak = 0
            for i in range(1, len(group_filled)):
                if pd.isna(group_filled[col].iloc[i]):
                    missing_streak += 1
                    if missing_streak <= T:
                        group_filled.loc[group_filled.index[i], col] = group_filled[col].iloc[i - 1]
                else:
                    missing_streak = 0
        return group_filled

    @staticmethod
    def create_combinations_df(lists, columns):
        """
        与えられた複数のリストのすべての組み合わせを持つDataFrameを作成する関数

        Parameters:
        lists : リストのリスト
            各リストは、組み合わせを作成するための要素を含む
        columns : list
            DataFrameのカラム名を指定するリスト

        Returns:
        df : DataFrame
            すべての組み合わせを持つDataFrame
        """

        if len(lists) != len(columns):
            raise ValueError("The number of lists must match the number of columns")

        combinations = list(product(*lists))

        df = pd.DataFrame(combinations, columns=columns)
        return df
