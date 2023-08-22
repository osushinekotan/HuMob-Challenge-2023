from typing import Callable

import pandas as pd


class GroupedDiffFeatureExtractor:
    def __init__(
        self,
        group_key: str | list = "uid",
        group_values: list[str] = ["t", "d"],
        intervals: list[int] = [1, 2],
    ):
        self.group_key = group_key
        self.group_values = group_values
        self.intervals = intervals
        if isinstance(group_key, list):
            self.group_key_name = "_".join(group_key)
        else:
            self.group_key_name = group_key

    def __call__(self, df):
        cols = [
            {v: f"{v}_grpby_{self.group_key_name}_diff_{interval}" for v in self.group_values}
            for interval in self.intervals
        ]
        out_df = pd.concat(
            [
                df.groupby(self.group_key)[self.group_values].diff(interval).rename(columns=col)
                for interval, col in zip(self.intervals, cols)
            ],
            axis=1,
        )
        return out_df.add_prefix("f_")


class GroupedShiftFeatureExtractor:
    def __init__(
        self,
        group_key: str | list = "uid",
        group_values: list[str] = ["t", "d"],
        intervals: list[int] = [1, 2],
    ):
        self.group_key = group_key
        self.group_values = group_values
        self.intervals = intervals
        if isinstance(group_key, list):
            self.group_key_name = "_".join(group_key)
        else:
            self.group_key_name = group_key

    def __call__(self, df):
        cols = [
            {v: f"{v}_grpby_{self.group_key_name}_shift_{interval}" for v in self.group_values}
            for interval in self.intervals
        ]
        out_df = pd.concat(
            [
                df.groupby(self.group_key)[self.group_values].shift(interval).rename(columns=col)
                for interval, col in zip(self.intervals, cols)
            ],
            axis=1,
        )
        return out_df.add_prefix("f_")


class GroupedSimpleFeatureExtoractor:
    def __init__(
        self,
        group_key: str | list = "uid",
        group_values: list[str] = ["t", "d"],
        agg_methods: list[str, Callable] = ["mean", "min", "max"],
    ):
        self.group_key = group_key
        self.group_values = group_values
        self.agg_methods = agg_methods

        if isinstance(group_key, list):
            self.group_key_name = "_".join(group_key)
        else:
            self.group_key_name = group_key

    def aggregate(self, df):
        return df.groupby(self.group_key)[self.group_values].agg(self.agg_methods)

    def __call__(self, df):
        agg_df = self.aggregate(df=df)
        agg_df.columns = [f"{x[0]}_grpby_{self.group_key_name}_agg_{x[1]}" for x in agg_df.columns]
        return (
            pd.merge(
                df[self.group_key],
                agg_df,
                how="left",
                left_on=self.group_key,
                right_index=True,
            )
            .drop(self.group_key, axis=1)
            .add_prefix("f_")
        )


class D60MaskGroupedSimpleFeatureExtoractor(GroupedSimpleFeatureExtoractor):
    def aggregate(self, df):
        return df.query("d < 60").groupby(self.group_key)[self.group_values].agg(self.agg_methods)


class TimeGroupedSimpleFeatureExtoractor:
    def __init__(
        self,
        group_key: str | list = "uid",
        group_values: list[str] = ["t", "d"],
        agg_methods: list[str, Callable] = ["mean", "min", "max"],
        time_range: dict[str, list] = {"d": [0, 8], "t": [0, 20]},
    ):
        self.group_key = group_key
        self.group_values = group_values
        self.time_range = time_range
        self.agg_methods = agg_methods

        if isinstance(group_key, list):
            self.group_key_name = "_".join(group_key)
        else:
            self.group_key_name = group_key

        self.d_range = list(range(*time_range["d"])) if "d" in self.time_range else None
        self.t_range = list(range(*time_range["t"])) if "t" in self.time_range else None

        self.time_range_name = self.format_dict(time_range)

    @staticmethod
    def format_dict(d):
        result = []
        for key, values in d.items():
            result.append(f"{key}{values[0]}_{values[1]}")
        return "_".join(result)

    def __call__(self, df):
        selected_df = df[df["d"].isin(self.d_range)].reset_index(drop=True) if self.d_range is not None else df.copy()
        selected_df = df[df["t"].isin(self.t_range)].reset_index(drop=True) if self.t_range is not None else selected_df

        agg_df = selected_df.groupby(self.group_key)[self.group_values].agg(self.agg_methods)
        agg_df.columns = [
            f"{x[0]}_grpby_{self.group_key_name}_agg_{x[1]}_{self.time_range_name}" for x in agg_df.columns
        ]
        return (
            pd.merge(
                df[self.group_key],
                agg_df,
                how="left",
                left_on=self.group_key,
                right_index=True,
            )
            .drop(self.group_key, axis=1)
            .add_prefix("f_")
        )


class RawFeatureExtractor:
    def __init__(self, use_columns) -> None:
        self.use_columns = use_columns

    def __call__(self, df):
        return df[self.use_columns].add_prefix("f_")
