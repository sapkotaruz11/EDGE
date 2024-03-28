import argparse
import os


# Custom validation function for datasets argument
def validate_datasets(value):
    valid_datasets = ["Mutag", "AIFB"]  # Add more valid datasets as needed
    datasets = value.split()
    for dataset in datasets:
        if dataset not in valid_datasets:
            raise argparse.ArgumentTypeError(f"Invalid dataset: {dataset}.")
    return datasets[0]


# Default list of datasets
default_datasets = ["mutag", "aifb"]


# Default list of explainers
default_explainers = ["EvoLearner", "SubGraphX", "PGExplainer", "CELOE"]


def validate_explainers(values):
    valid_explainers = [
        "EvoLearner",
        "SubGraphX",
        "PGExplainer",
        "CELOE",
    ]  # Add more valid explainers as needed
    explainers = values.split()
    for explainer in explainers:
        if explainer not in valid_explainers:
            raise argparse.ArgumentTypeError(f"Invalid explainer: {explainer}.")
    return explainers[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Framework for training and Evaluation different Explainers on Heterogenous Data."
    )
    parser.add_argument(
        "--train",
        action="store_true",  # Use action='store_false' if you want False as default
        help="Specify to print results",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        type=validate_datasets,
        default=default_datasets,
        help=f"Specify the datasets to use, separated by spaces (default: {', '.join(default_datasets)})",
    )

    # Add argument for explainers
    parser.add_argument(
        "--explainers",
        nargs="+",
        type=validate_explainers,
        default=default_explainers,
        help=f"Specify the explainers to use, separated by spaces (default: {', '.join(default_explainers)})",
    )

    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of runs to execute (default: 5).",
    )
    # Add the argument for printing results
    parser.add_argument(
        "--print_results",
        action="store_true",  # Use action='store_false' if you want False as default
        help="Specify to print results",
    )

    args = parser.parse_args()
    print("Datasets:", args.datasets)
    print("Explainers", args.explainers)
    if args.train:
        from src.explainers_runner import run_explainers

        for dataset in args.datasets:
            run_explainers(
                explainers=args.explainers, dataset=dataset, no_of_runs=args.num_runs
            )

    if args.print_results:
        from src.utils.print_results import print_results

        print_results()
