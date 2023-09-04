import click


@click.command()
@click.option("--fe", is_flag=True, help="Run fe.run() if set.")
@click.option("--train", is_flag=True, help="Run train.run() if set.")
@click.option("--inference", is_flag=True, help="Run inference.run() if set.")
@click.option("--make_submission", is_flag=True, help="Run make_submission.run() if set.")
def run_v01(**kwargs):
    """Run selected functions based on the provided flags."""
    from custom.runner.v01 import fe, inference, make_submission, train

    if kwargs.get("fe"):
        fe.run()
    if kwargs.get("train"):
        train.run()
    if kwargs.get("inference"):
        inference.run()
    if kwargs.get("make_submission"):
        make_submission.run()


if __name__ == "__main__":
    run_v01()
