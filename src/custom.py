from pathlib import Path

import click
from util import load_yaml


@click.command()
@click.option("--fe", is_flag=True, help="Run fe.run() if set.")
@click.option("--train", is_flag=True, help="Run train.run() if set.")
@click.option("--inference", is_flag=True, help="Run inference.run() if set.")
@click.option("--make_submission", is_flag=True, help="Run make_submission.run() if set.")
@click.option("--config_file", type=str, default="000_task2", help="Path to the configuration file.")
def run_v01(fe, train, inference, make_submission, config_file):
    """Run selected functions based on the provided flags."""
    from custom.runner.v01 import fe as fe_module
    from custom.runner.v01 import inference as inference_module
    from custom.runner.v01 import make_submission as make_submission_module
    from custom.runner.v01 import train as train_module

    config_filepath = Path(f"/workspace/conf/customs/{config_file}.yaml")
    pre_eval_config = load_yaml(config_filepath)

    if fe:
        fe_module.run(pre_eval_config)
    if train:
        train_module.run(pre_eval_config)
    if inference:
        inference_module.run(pre_eval_config)
    if make_submission:
        make_submission_module.run(pre_eval_config)


if __name__ == "__main__":
    run_v01()
