import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Framework for training, evaluating, and printing results."
    )
    parser.add_argument(
        "--train", action="store_true", default=False, help="Train the model"
    )
    parser.add_argument(
        "--evaluate", action="store_true", default=False, help="Evaluate the model"
    )
    parser.add_argument(
        "--print-results", action="store_true", default=True, help="Print results"
    )

    args = parser.parse_args()

    if args.train:
        from src.train_evo import train_evo
        from src.train_celoe import train_celoe

        train_evo()
        train_celoe()

    if args.evaluate:
        from src.evaluations import evaluate_logical_explainers

        evaluate_logical_explainers()

    if args.print_results:
        from src.print_results import print_results

        print_results()
