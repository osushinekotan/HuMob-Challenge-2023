def v01():
    from custom.runner.v01 import fe, inference, make_submission, train

    fe.run()
    train.run()
    inference.run()
    make_submission.run()


if __name__ == "__main__":
    v01()
