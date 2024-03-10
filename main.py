import argparse
import os

from src.gnn_explainers.trainer import train_gnn
from src.utils.create_lp import create_lp_aifb, create_lp_mutag
from src.runner import main_runner


def get_default_models():
    # Return a list of all default models
    return ["PGExplainer", "SubGraphX", "EVO", "CELOE"]


def get_default_datasets():
    # Return a list of all default datasets
    return ["Mutag", "AIFB"]  # Update this list when there are more datasets


def get_model_name_mapping():
    # Mapping from lowercase model names and their aliases to their required format
    return {
        "evo": "EVO",
        "evolearner": "EVO",
        "celoe": "CELOE",
        "pgexplainer": "PGExplainer",
        "subgraphx": "SubGraphX",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Framework for training and Evaluation different Explainers on Heterogenous Data."
    )
    parser.add_argument(
        "--retrain-models",
        action="store_true",
        help="Flag to retrain models even if they are already trained",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="Specify the models to train, separated by spaces (e.g., PGExplainer SubGraphX EvoLearner CELOE)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specify the datasets to use, separated by spaces (e.g., Mutag AIFB)",
    )
    parser.add_argument(
        "--evaluation_type",
        type=str,
        choices=["micro", "macro"],
        default="micro",
        help="Specify the evaluation type: 'micro' or 'macro'",
    )
    # Add the argument for printing results
    parser.add_argument(
        "--print_results",
        action="store_true",  # Use action='store_false' if you want False as default
        help="Specify to print results",
    )

    args = parser.parse_args()
    # Use default models and datasets if none are provided
    models = (
        [model.lower() for model in args.models]
        if args.models
        else [m.lower() for m in get_default_models()]
    )
    datasets = (
        [dataset.lower() for dataset in args.datasets]
        if args.datasets
        else [d.lower() for d in get_default_datasets()]
    )

    main_runner(models=models, datasets=datasets)
    if args.print_results:
        from src.utils.print_results import print_results

        print_results()
