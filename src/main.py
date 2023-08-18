def run():
    from custom.runner.v02 import fe, inference, make_submission, train

    # fe.run()
    train.run()
    inference.run()
    make_submission.run()


if __name__ == "__main__":
    run()
