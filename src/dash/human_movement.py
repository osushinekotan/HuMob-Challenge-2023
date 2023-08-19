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
        user_ids = df["uid"].sample(100, random_state=None)
        df = df[df["uid"].isin(user_ids)]

    df["time"] = df["d"].astype(str).str.zfill(2) + "-" + df["t"].astype(str).str.zfill(2)
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
        html.H1("Movement of uid(s) over time"),
        dcc.Dropdown(
            id="uid-dropdown",
            options=[{"label": uid, "value": uid} for uid in df["uid"].unique()],
            value=[df["uid"].unique()[0]],
            multi=True,
        ),
        dcc.Slider(
            id="time-slider",
            min=0,
            max=len(df["time"].unique()) - 1,
            value=0,
            marks={i: "" for i in range(len(df["time"].unique()))},
            step=1,
        ),
        # 散布図用のGraph
        dcc.Graph(id="movement-graph"),
        # 折れ線グラフ用のGraph
        dcc.Graph(id="line-graph"),
    ]
)


@app.callback(
    [Output("movement-graph", "figure"), Output("line-graph", "figure")],
    [Input("uid-dropdown", "value"), Input("time-slider", "value")],
)
def update_graph(selected_uids, slider_value):
    # スライダーの値から対応するtimeを取得
    selected_time = df["time"].unique()[slider_value]

    dff = df[df["uid"].isin(selected_uids)]
    dff = dff[dff["time"] <= selected_time]  # time列を用いてフィルタリング

    # px.scatterを呼び出す際に直接customdataをセット
    fig = px.scatter(
        dff,
        x="x_point",
        y="y_point",
        color="uid",
        size_max=15,
        animation_frame="time",
        animation_group="uid",
        hover_data={
            "x_point": False,  # x_pointの表示を無効化
            "y_point": False,  # y_pointの表示を無効化
            "x": True,  # xの表示を有効化
            "y": True,  # yの表示を有効化
        },
    )

    # 図の大きさを調整し、点が被らないようにマーカーの大きさと透明度を調整
    fig.update_layout(
        plot_bgcolor="white",
        autosize=False,
        width=1600,
        height=1200,
        showlegend=True,
    )
    fig.update_traces(marker=dict(size=8, opacity=1.0), selector=dict(mode="markers"))

    # x軸とy軸の範囲をuidの最大x,y~最小x,yの範囲で限定し、tickの位置にグリッド線を引く
    fig.update_xaxes(
        range=[dff["x"].min() - 1, dff["x"].max() + 1],
        tick0=0,
        dtick=1,
        showticklabels=False,
        gridcolor="LightGray",
        zerolinecolor="LightGray",
        gridwidth=1,
    )
    fig.update_yaxes(
        range=[dff["y"].min() - 1, dff["y"].max() + 1],
        tick0=0,
        dtick=1,
        showticklabels=False,
        gridcolor="LightGray",
        zerolinecolor="LightGray",
        gridwidth=1,
    )

    # 新しく追加する折れ線グラフの作成
    line_fig = px.line(
        dff,
        x="time",
        y=["x", "y"],
        color="uid",
        labels={"x": "Time", "value": "Position Value", "variable": "Axis"},
        title="Movement over Time",
    )
    line_fig.update_traces(mode="lines+markers")
    line_fig.update_layout(xaxis_type="category")

    return fig, line_fig


# サーバを起動
if __name__ == "__main__":
    app.run_server(debug=DEBUG)
