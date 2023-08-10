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
        return out_df.add_prefix("f_").fillna(-1)


class RawFeatureExtractor:
    def __init__(self, use_columns) -> None:
        self.use_columns = use_columns

    def __call__(self, df):
        return df[self.use_columns].add_prefix("f_")
