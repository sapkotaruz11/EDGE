import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Framework for training and Evaluation different Explainers on Heterogenous Data."
    )
    parser.add_argument(
        "--train_gnn",
        action="store_true",
        default=False,
        help="Train the GNN Explainer models",
    )
    parser.add_argument(
        "--train_logical",
        action="store_true",
        default=False,
        help="Train the Logical Approachs",
    )
    parser.add_argument(
        "--evaluate", action="store_true", default=False, help="Evaluate the models"
    )
    parser.add_argument(
        "--evaluate_macro",
        action="store_true",
        default=False,
        help="Evaluate the models with macro averaging",
    )
    parser.add_argument(
        "--print-results", action="store_true", default=True, help="Print results"
    )

    args = parser.parse_args()

    if args.train_gnn:
        from src.gnn_explainers.HeteroPG_explainer import explain_PG
        from src.gnn_explainers.HeteroSubGraphX_explainer import explain_SGX

        explain_PG(dataset="aifb", print_explainer_loss=True)
        explain_PG(dataset="mutag", print_explainer_loss=True)

        explain_SGX(dataset="aifb")
        explain_SGX(dataset="mutag")

    if args.train_logical:
        from src.logical_explainers.EvoLearner import train_evo, train_evo_fid

        train_evo()
        train_evo_fid()

        from src.logical_explainers.CELOE import train_celoe, train_celoe_fid

        train_celoe(use_heur=False)
        train_celoe_fid(use_heur=False)

    if args.evaluate:
        from src.evaluations import evaluate_gnn_explainers, evaluate_logical_explainers

        evaluate_logical_explainers()
        evaluate_gnn_explainers()

    if args.evaluate_macro:
        from src.evaluations_macro import (
            evaluate_gnn_explainers,
            evaluate_logical_explainers,
        )

        evaluate_logical_explainers()
        evaluate_gnn_explainers()

    if args.print_results:
        from src.print_results import print_results

        print_results()
