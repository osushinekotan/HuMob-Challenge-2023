from pathlib import Path

import joblib
import pandas as pd
from logger import Logger

logger = Logger(name="make_submission")


def run(pre_eval_config):
    out_dir = Path(pre_eval_config["global"]["resources"]) / "output"

    test_predictions = joblib.load(
        out_dir / pre_eval_config["nn"]["out_dir"] / pre_eval_config["fe"]["dataset"] / "test_outputs.pkl"
    )
    test_df = (
        joblib.load(
            out_dir / pre_eval_config["fe"]["out_dir"] / pre_eval_config["fe"]["dataset"] / "test_feature_df.pkl"
        )
        .query("d>=60")
        .reset_index(drop=True)
    )

    logger.debug(f"test_predictions : {test_predictions.shape}")
    logger.debug(f"test_df : {test_df.shape}")
    submission_df = pd.concat([test_df[["uid", "d", "t"]], pd.DataFrame(test_predictions, columns=["x", "y"])], axis=1)
    assert submission_df.isnull().sum().sum() == 0

    logger.debug(f"\nsubmission_df\n\n{submission_df}")

    task_name = pre_eval_config["fe"]["dataset"].split("_")[0]
    submission_filepath = (
        out_dir
        / pre_eval_config["nn"]["out_dir"]
        / pre_eval_config["fe"]["dataset"]
        / f"osushineko_{task_name}midterm_humob.csv.gz"
    )
    submission_df.astype({"uid": int}).to_csv(
        submission_filepath,
        compression="gzip",
        index=None,
    )
