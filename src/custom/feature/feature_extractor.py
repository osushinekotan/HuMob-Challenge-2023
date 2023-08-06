import pandas as pd


class GroupedDiffFeatureExtractor:
    def __init__(
        self,
        group_key: str = "uid",
        group_values: list[str] = ["t", "d"],
        intervals: list[int] = [1, 2],
    ):
        self.group_key = group_key
        self.group_values = group_values
        self.intervals = intervals

    def __call__(self, df):
        cols = [
            {v: f"{v}_grpby_{self.group_key}_diff_{interval}" for v in self.group_values} for interval in self.intervals
        ]
        out_df = pd.concat(
            [
                df.groupby(self.group_key)[self.group_values].diff(interval).rename(columns=col)
                for interval, col in zip(self.intervals, cols)
            ],
            axis=1,
        )
        return out_df.add_prefix("f_").fillna(-1)
