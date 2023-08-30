# ライブラリをインポート
from pathlib import Path
from typing import Callable

import dash
import numpy as np
import pandas as pd
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output

HOME = Path("/workspace")
RESOURCES = HOME / "resources"
INPUT = RESOURCES / "input"

DEBUG = True
lower = 0.1
upper = 0.9

# Global cache for datasets
datasets_cache = {}

# Dashアプリケーションを作成
app = dash.Dash(__name__)


def read_parquet_from_csv(
    filepath: Path,
    dirpath: Path,
    process_fns: list[Callable] | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    name = filepath.name.split(".")[0]
    parquet_filepath = dirpath / f"{name}.parquet"
    if parquet_filepath.is_file() and (not overwrite):
        return pd.read_parquet(parquet_filepath)

    df = pd.read_csv(filepath)

    if process_fns is not None:
        for fn in process_fns:
            df = fn(df)

    df = df.reset_index(drop=True)
    df.to_parquet(parquet_filepath)
    return df


def load_and_process_data():
    # if dataset already in cache, return it
    if "main_df" in datasets_cache:
        return datasets_cache["main_df"]

    # otherwise, load and preprocess it
    df = read_parquet_from_csv(filepath=INPUT / "task1_dataset.csv.gz", dirpath=INPUT)

    if DEBUG:
        uids = df.query("x == 999")["uid"].unique()
        df = df[~df["uid"].isin(uids)].reset_index(drop=True)
        user_ids = pd.Series(df["uid"].unique()).sample(5, random_state=None).tolist() + [64902, 28678]
        df = df[df["uid"].isin(user_ids)]

    df["time"] = (df["d"].astype(str).str.zfill(2) + df["t"].astype(str).str.zfill(2)).astype(int)
    df["x_point"] = df["x"] + (upper - lower) * np.random.rand() + lower
    df["y_point"] = df["y"] + (upper - lower) * np.random.rand() + lower
    df["uid"] = df["uid"].astype(str)
    df = df.sort_values("time").reset_index(drop=True)

    # store in cache before returning
    datasets_cache["main_df"] = df
    return df


df = load_and_process_data()

app.layout = html.Div(
    [
        html.H1("XYLine"),
        dcc.Dropdown(
            id="uid-dropdown",
            options=[{"label": uid, "value": uid} for uid in df["uid"].unique()],
            value=[df["uid"].unique()[0]],
            multi=True,
        ),
        # 折れ線グラフ用のGraph
        dcc.Graph(id="x-line"),
        dcc.Graph(id="y-line"),
    ]
)


@app.callback(
    [Output("x-line", "figure"), Output("y-line", "figure")],
    [Input("uid-dropdown", "value")],
)
def update_graph(selected_uids):
    dff = df[df["uid"].isin(selected_uids)].sort_values("time")

    # 新しく追加する折れ線グラフの作成
    x_line_fig = px.line(
        dff,
        x="time",
        y="x",
        color="uid",
        labels={"x": "Time", "value": "Position Value", "variable": "Axis"},
        title="X",
    )
    x_line_fig.update_traces(mode="lines+markers")

    y_line_fig = px.line(
        dff,
        x="time",
        y="y",
        color="uid",
        labels={"x": "Time", "value": "Position Value", "variable": "Axis"},
        title="Y",
    )
    y_line_fig.update_traces(mode="lines+markers")
    return x_line_fig, y_line_fig


# サーバを起動
if __name__ == "__main__":
    app.run_server(debug=DEBUG)
