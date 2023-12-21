import argparse
import os
from src.gnn_explainers.trainer import train_gnn
from src.utils.create_lp import create_lp_aifb, create_lp_mutag


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

    # Handling training for specified models and datasets
    if args.retrain_models and models and datasets:
        for dataset in datasets:
            model_path = f"trained_models/{dataset}_trained.pt"
            lp_path = f"configs/{dataset}.json"

            # Check if trained model exists
            if not os.path.exists(model_path):
                train_gnn(dataset=dataset)

            # Check if learning problem exists
            if not os.path.exists(lp_path):
                if dataset == "aifb":
                    create_lp_aifb()
                elif dataset == "mutag":
                    create_lp_mutag()

        for model in models:
            if model == "pgexplainer" or model == "subgraphx":
                for dataset in datasets:
                    if model == "pgexplainer":
                        from src.gnn_explainers.HeteroPG_explainer import explain_PG

                        explain_PG(dataset=dataset, print_explainer_loss=True)

                    if model == "subgraphx":
                        from src.gnn_explainers.HeteroSubGraphX_explainer import (
                            explain_SGX,
                        )

                        explain_SGX(dataset=dataset)

            if model == "evolearner" or model == "evo":
                from src.logical_explainers.EvoLearner import train_evo, train_evo_fid

                train_evo(kgs=datasets)
                train_evo_fid(kgs=datasets)

            if model == "celoe":
                from src.logical_explainers.CELOE import train_celoe, train_celoe_fid

                train_celoe(kgs=datasets, use_heur=False)
                train_celoe_fid(kgs=datasets, use_heur=False)

    model_name_mapping = get_model_name_mapping()
    # Segregate models into logical and GNN categories
    logical_models = [
        model_name_mapping.get(model, model)
        for model in models
        if model in ["evo", "evolearner", "celoe"]
    ]
    gnn_models = [
        model_name_mapping.get(model, model)
        for model in models
        if model in ["pgexplainer", "subgraphx"]
    ]

    # Call evaluation functions with the filtered model lists and datasets
    # Handling evaluation
    if args.evaluation_type == "macro":
        from src.evaluations_macro import (
            evaluate_gnn_explainers,
            evaluate_logical_explainers,
        )

        if logical_models:
            evaluate_logical_explainers(explainers=logical_models, KGs=datasets)
        if gnn_models:
            evaluate_gnn_explainers(explainers=gnn_models, datasets=datasets)

    else:  # Default to micro evaluation
        from src.evaluations import evaluate_gnn_explainers, evaluate_logical_explainers

        if logical_models:
            evaluate_logical_explainers(explainers=logical_models, KGs=datasets)
        if gnn_models:
            evaluate_gnn_explainers(explainers=gnn_models, datasets=datasets)

    # Always print results at the end
    from src.print_results import print_results

    print_results()
