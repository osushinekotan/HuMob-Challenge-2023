from pathlib import Path

import pandas as pd
from pandarallel import pandarallel
from pytorch_pfn_extras.config import Config
from rule.cycle.evaluate import make_preds_df
from rule.cycle.imputer import CycleImputer
from tqdm import tqdm

pandarallel.initialize(progress_bar=False, use_memory_fs=False)


def process_batch(batch, config):
    imputer = CycleImputer(group_keys=config["cycle/group_keys"], agg_method=config["cycle/agg_method"])
    preds_df = make_preds_df(
        batch,
        imputer=imputer,
        task_dataset=config["cycle/task_dataset"],
        cycle_groups=config["cycle/cycle_groups"],
        T=config["cycle/T"],
    )
    generated = preds_df[["uid", "d", "t"] + imputer.agg_cols]
    generated.columns = ["uid", "d", "t", "x", "y"]

    return generated


def run(pre_eval_config):
    config = Config(config=pre_eval_config)
    task_dataset = config["cycle/task_dataset"]

    filepath = f"/workspace/resources/input/{task_dataset}_raw_test.parquet"

    df = pd.read_parquet(filepath)
    print(df)

    grouped = [group for _, group in df.groupby("uid")]

    batch_size = config["cycle/inference/batch_size"]
    num_batches = (len(grouped) + batch_size - 1) // batch_size

    generated = [
        process_batch(pd.concat(grouped[i * batch_size : (i + 1) * batch_size]).reset_index(drop=True), config)
        for i in tqdm(range(num_batches), desc="Processing batches")
    ]
    generated = pd.concat(generated).reset_index(drop=True).astype(int)

    print(generated)

    out_dir = Path(f"/workspace/resources/output/rule/{config['cycle/name']}")
    out_dir.mkdir(parents=True, exist_ok=True)

    submission_filepath = out_dir / f"osushineko_{task_dataset[:5]}_humob.csv.gz"
    generated.to_csv(submission_filepath, index=False)
