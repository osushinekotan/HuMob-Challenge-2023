import pandas as pd
from imputer import CycleImputer
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import geobleu


def sampling(df, n, seed=0):
    uids = pd.Series(df["uid"].unique()).sample(n=n, random_state=seed)
    return df[df["uid"].isin(uids)].reset_index(drop=True)


def assign_day_of_week(df):
    df["dayofweek"] = (df["d"] % 7).astype(int)
    df["weekend"] = df["dayofweek"].isin([6, 0])
    return df


def assign_t_labe(df):
    morning = {k: 0 for k in list(range(12, 36))}
    midnight = {k: 1 for k in list(range(36, 48)) + list(range(0, 12))}
    t_label_mapping = {**morning, **midnight}
    df["t_label"] = df["t"].map(t_label_mapping)
    return df


def assign_detailed_t_label(df):
    division = 48 // 12  # 48を12で割った値
    result_dict = {i: i // division for i in range(48)}
    df["detailed_t_label"] = df["t"].map(result_dict)
    return df


def preprocess(df):
    assign_funcs = [
        assign_day_of_week,
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

        geobleu_score += geobleu.calc_geobleu(a_generated, a_reference, processes=3)
        dtw_score += geobleu.calc_dtw(a_generated, a_reference, processes=3)

    geobleu_score = geobleu_score / len(eval_uids)
    dtw_score = dtw_score / len(eval_uids)
    rmse = mean_squared_error(y_true=reference[["x", "y"]].values, y_pred=generated[["x", "y"]].values, squared=False)

    scores = dict(
        geobleu_score=geobleu_score,
        dtw_score=dtw_score,
        rmse=rmse,
    )
    return scores


def main(task_dataset, eval_uid_num, group_keys, agg_method, cycle_groups, T, seed):
    filepath = f"/workspace/resources/input/{task_dataset}_raw_train.parquet"

    df = pd.read_parquet(filepath)
    df = sampling(df=df, n=eval_uid_num, seed=seed)
    df = preprocess(df)

    imputer = CycleImputer(group_keys=group_keys, agg_method=agg_method)
    filled_df = imputer.impute(df=df, cycle_groups=cycle_groups, T=T)

    preds_df = make_valid_df(imputer=imputer, raw_df=df, filled_df=filled_df)
    reference, generated = make_eval_inputs(imputer, preds_df)
    scores = calc_metrics(reference, generated, max_eval=eval_uid_num)
    print(scores)


if __name__ == "__main__":
    # TODO : make config file
    task_dataset = "task1_dataset"
    eval_uid_num = 1000
    seed = 0
    group_keys = ["uid", "dayofweek", "t_label", "t"]
    agg_method = "median"
    cycle_groups = [["uid", "dayofweek"]]
    T = 24

    main(
        task_dataset=task_dataset,
        eval_uid_num=eval_uid_num,
        group_keys=group_keys,
        agg_method=agg_method,
        cycle_groups=cycle_groups,
        T=T,
        seed=seed,
    )
