from pathlib import Path

import click
from util import load_yaml


@click.command()
@click.option("--evaluation", is_flag=True, help="Run evaluate.run() if set.")
@click.option("--make_submission", is_flag=True, help="Run make_submission.run() if set.")
@click.option("--config_file", type=str, default="000_task2", help="Path to the configuration file.")
def run_rule(evaluation, make_submission, config_file):
    """Run selected functions based on the provided flags."""
    from rule.cycle import evaluate as eval_module
    from rule.cycle import make_submission as submit_module

    config_filepath = Path(f"/workspace/conf/rules/{config_file}.yaml")
    pre_eval_config = load_yaml(config_filepath)

    if evaluation:
        eval_module.run(pre_eval_config)
    if make_submission:
        submit_module.run(pre_eval_config)


if __name__ == "__main__":
    run_rule()
