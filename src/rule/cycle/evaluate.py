import numpy as np
import pandas as pd
from pandarallel import pandarallel
from pytorch_pfn_extras.config import Config
from rule.cycle.imputer import CycleImputer
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import geobleu

pandarallel.initialize(progress_bar=True, use_memory_fs=False)


def sampling(df, n, seed=0):
    uids = pd.Series(df["uid"].unique()).sample(n=n, random_state=seed)
    return df[df["uid"].isin(uids)].reset_index(drop=True)


def assign_day_of_week(df, task=1):
    df["dayofweek"] = (df["d"] % 7).astype(int).values
    df["weekend"] = df["dayofweek"].isin([6, 0]).values

    if task == 1:
        df["weekend"] = np.array(df["weekend"] + df["d"].isin([37, 36]), dtype=bool)  # + holoday
    return df


def assign_t_labe(df):
    morning = {k: 0 for k in list(range(12, 36))}
    midnight = {k: 1 for k in list(range(36, 48)) + list(range(0, 12))}
    t_label_mapping = {**morning, **midnight}
    df["t_label"] = df["t"].map(t_label_mapping).values
    return df


def assign_detailed_t_label(df):
    division = 48 // 12
    result_dict = {i: i // division for i in range(48)}
    df["detailed_t_label"] = df["t"].map(result_dict).values
    return df


def preprocess(df, task):
    task = 1 if task == "task1_dataset" else 0
    assign_funcs = [
        lambda x: assign_day_of_week(x, task=task),
        assign_t_labe,
        assign_detailed_t_label,
    ]

    for func in assign_funcs:
        df = func(df)
    return df


def make_valid_df(imputer, raw_df, filled_df):
    preds_df = raw_df.query("d >= 60").dropna().reset_index(drop=True)  # valid
    preds_df = pd.merge(preds_df, filled_df, on=imputer.group_keys, how="left")
    return preds_df


def make_eval_inputs(imputer, preds_df):
    reference = preds_df[["uid", "d", "t", "x", "y"]]
    generated = preds_df[["uid", "d", "t"] + imputer.agg_cols]
    generated.columns = reference.columns
    return reference, generated


def calc_metrics(reference, generated, max_eval=100):
    eval_uids = reference["uid"].unique()[:max_eval]

    geobleu_score = 0
    dtw_score = 0

    for uid in tqdm(eval_uids):
        a_generated = generated.loc[generated["uid"] == uid, ["d", "t", "x", "y"]].values.tolist()
        a_reference = reference.loc[reference["uid"] == uid, ["d", "t", "x", "y"]].values.tolist()

        geobleu_score += geobleu.calc_geobleu(a_generated, a_reference, processes=4)
        dtw_score += geobleu.calc_dtw(a_generated, a_reference, processes=4)

    geobleu_score = geobleu_score / len(eval_uids)
    dtw_score = dtw_score / len(eval_uids)
    rmse = mean_squared_error(y_true=reference[["x", "y"]].values, y_pred=generated[["x", "y"]].values, squared=False)

    scores = dict(
        geobleu_score=geobleu_score,
        dtw_score=dtw_score,
        rmse=rmse,
    )
    return scores


def make_preds_df(df, imputer, task_dataset, cycle_groups, T):
    df = preprocess(df, task=task_dataset)
    filled_df = imputer.impute(df=df, cycle_groups=cycle_groups, T=T)
    preds_df = make_valid_df(imputer=imputer, raw_df=df, filled_df=filled_df)
    return preds_df


def run(pre_eval_config):
    config = Config(config=pre_eval_config)
    task_dataset = config["cycle/task_dataset"]

    filepath = f"/workspace/resources/input/{task_dataset}_raw_train.parquet"

    df = pd.read_parquet(filepath)
    df = sampling(df=df, n=config["cycle/eval/eval_uid_num"], seed=config["global/seed"])
    imputer = CycleImputer(group_keys=config["cycle/group_keys"], agg_method=config["cycle/agg_method"])
    preds_df = make_preds_df(
        df,
        imputer=imputer,
        task_dataset=config["cycle/task_dataset"],
        cycle_groups=config["cycle/cycle_groups"],
        T=config["cycle/T"],
    )
    reference, generated = make_eval_inputs(imputer, preds_df)
    scores = calc_metrics(reference, generated, max_eval=config["cycle/eval/eval_uid_num"])
    print(scores)
