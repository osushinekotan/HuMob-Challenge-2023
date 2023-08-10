from pathlib import Path

import joblib
import pandas as pd
from logger import Logger
from util import load_yaml

logger = Logger(name="make_submission")


def postprocess(test_predictions):
    return test_predictions.astype(int)


def run():
    pre_eval_config = load_yaml()
    out_dir = Path(pre_eval_config["global"]["resources"]) / "output"

    test_predictions = joblib.load(
        out_dir / pre_eval_config["nn"]["out_dir"] / pre_eval_config["fe"]["dataset"] / "test_outputs.pkl"
    )
    test_predictions = postprocess(test_predictions)
    test_df = (
        joblib.load(
            out_dir / pre_eval_config["fe"]["out_dir"] / pre_eval_config["fe"]["dataset"] / "test_feature_df.pkl"
        )
        .query("d>=60")
        .reset_index(drop=True)
    )
    submission_df = pd.concat([test_df[["uid", "d", "t"]], pd.DataFrame(test_predictions, columns=["x", "y"])], axis=1)
    assert submission_df.isnull().sum().sum() == 0

    logger.debug(f"\nsubmission_df\n\n{submission_df}")


if __name__ == "__main__":
    with logger.time_log():
        run()
