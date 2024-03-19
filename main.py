import argparse
import os

from src.gnn_explainers.trainer import train_gnn
from src.utils.create_lp import create_lp_aifb, create_lp_mutag


def get_default_datasets():
    # Return a list of all default datasets
    return ["Mutag", "AIFB"]  # Update this list when there are more datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Framework for training and Evaluation different Explainers on Heterogenous Data."
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specify the datasets to use, separated by spaces (e.g., Mutag AIFB)",
    )

    # Add the argument for printing results
    parser.add_argument(
        "--print_results",
        action="store_true",  # Use action='store_false' if you want False as default
        help="Specify to print results",
    )

    args = parser.parse_args()

    datasets = (
        [dataset.lower() for dataset in args.datasets]
        if args.datasets
        else [d.lower() for d in get_default_datasets()]
    )
    print(datasets)
    from src.explainers_runner import run_explainers

    for dataset in datasets:
        run_explainers(dataset=dataset)

    if args.print_results:
        from src.utils.print_results import print_results

        print_results()
