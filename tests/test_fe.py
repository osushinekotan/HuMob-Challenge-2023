import pandas as pd

from src.util import sort_df, sort_df_numpy


def test_sort_df():
    df = pd.DataFrame({"uid": [1, 2, 2, 1, 3], "d": [5, 4, 5, 3, 4], "t": [7, 6, 6, 5, 7], "x": [1, 2, 3, 4, 5]})

    result_original = sort_df(df)
    result_numpy = sort_df_numpy(df)

    print(result_original)
    print(result_numpy)
    assert result_original.equals(result_numpy), "Results do not match!"
