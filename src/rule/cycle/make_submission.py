import pandas as pd
from pytorch_pfn_extras.config import Config
from rule.cycle.evaluate import make_preds_df
from rule.cycle.imputer import CycleImputer
from util import load_yaml


def run():
    config = load_yaml("/workspace/src/conf/rule.yaml")
    config = Config(config=config)
    task_dataset = config["cycle/task_dataset"]

    filepath = f"/workspace/resources/input/{task_dataset}_raw_train.parquet"

    df = pd.read_parquet(filepath)
    imputer = CycleImputer(group_keys=config["cycle/group_keys"], agg_method=config["cycle/agg_method"])
    preds_df = make_preds_df(
        df,
        imputer=imputer,
        task_dataset=config["cycle/task_dataset"],
        cycle_groups=config["cycle/cycle_groups"],
        T=config["cycle/T"],
    )
    print(preds_df)


if __name__ == "__main__":
    run()
