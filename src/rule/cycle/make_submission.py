from pathlib import Path

import pandas as pd
from pytorch_pfn_extras.config import Config
from rule.cycle.evaluate import make_preds_df
from rule.cycle.imputer import CycleImputer


def run(pre_eval_config):
    config = Config(config=pre_eval_config)
    task_dataset = config["cycle/task_dataset"]

    filepath = f"/workspace/resources/input/{task_dataset}_raw_test.parquet"

    df = pd.read_parquet(filepath)
    print(df)

    imputer = CycleImputer(group_keys=config["cycle/group_keys"], agg_method=config["cycle/agg_method"])
    preds_df = make_preds_df(
        df,
        imputer=imputer,
        task_dataset=config["cycle/task_dataset"],
        cycle_groups=config["cycle/cycle_groups"],
        T=config["cycle/T"],
    )
    generated = preds_df[["uid", "d", "t"] + imputer.agg_cols]
    generated.columns = ["uid", "d", "t", "x", "y"]

    print(generated)

    out_dir = Path(f"/workspace/resources/output/rule/{config['cycle/name']}")
    out_dir.mkdir(parents=True, exist_ok=True)

    submission_filepath = out_dir / f"osushineko_{task_dataset[:5]}_humob.csv.gz"
    generated.to_csv(submission_filepath, index=False)
