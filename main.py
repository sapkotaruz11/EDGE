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
        "--train",
        action="store_true",  # Use action='store_false' if you want False as default
        help="Specify to print results",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specify the datasets to use, separated by spaces (e.g., Mutag AIFB)",
    )
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs to execute (default: 5).")
    # Add the argument for printing results
    parser.add_argument(
        "--print_results",
        action="store_true",  # Use action='store_false' if you want False as default
        help="Specify to print results",
    )

    args = parser.parse_args()
    no_of_runs = args.num_runs
    if args.train:
        datasets = (
            [dataset.lower() for dataset in args.datasets]
            if args.datasets
            else [d.lower() for d in get_default_datasets()]
        )
        from src.explainers_runner import run_explainers
    
        for dataset in datasets:
            run_explainers(dataset=dataset, no_of_runs = no_of_runs)

    if args.print_results:
        from src.utils.print_results import print_results

        print_results()
