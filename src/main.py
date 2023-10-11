import os ,  argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from experiment_enums import experimentsAll


def main():
    """
    The main function that parses arguments
    :return:
    """
    parser = argparse.ArgumentParser(description="run experiment using the provided experiment enums.")
    args = parser.parse_args()
    experiment_setup(args)


def experiment_setup(args: argparse.Namespace) -> None:
    """
    This function sets up the experiment and runs it
    :param args: dictionary arguments from user
    :return: None
    """
    experiments = experimentsAll
    for experiment in experiments:
        # experiment.run(logging_frequency=1)
        experiment.evaluate()


if __name__ == "__main__":
    main()
