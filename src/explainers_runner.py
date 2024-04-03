from src.Explainer import Explainer
import json
import os


def run_explainers(
    dataset, explainers, print_explainer_loss=True, no_of_runs=5, model="RGCN"
):
    print(f"Running explainers for {no_of_runs} runs for dataset {dataset}")
    performances = {}
    preds = {}
    predictions_data = {}

    for i in range(no_of_runs):
        print(f"Starting the  Run", i + 1)
        performances[i] = {}
        preds[i] = {}
        my_explainer = Explainer(
            explainers=explainers, dataset=dataset, model_name=model
        )
        for explainer in explainers:
            if explainer == "PGExplainer":
                performances[i][explainer] = my_explainer.evaluations.get(explainer, {})
            elif explainer == "SubGraphX":
                performances[i][explainer] = my_explainer.evaluations.get(explainer, {})
            elif explainer == "EvoLearner":
                performances[i][explainer] = my_explainer.evaluations.get(explainer, {})
                preds[i][explainer] = my_explainer.explanations.get(explainer, {})
            elif explainer == "CELOE":
                performances[i][explainer] = my_explainer.evaluations.get(explainer, {})
                preds[i][explainer] = my_explainer.explanations.get(explainer, {})

        predictions_data[i] = my_explainer.pred_df.to_json()

    for explainer in explainers:
        file_path_evaluations = (
            f"results/evaluations/{model}/{explainer}/{dataset}.json"
        )
        # Get the directory path from the file path
        directory = os.path.dirname(file_path_evaluations)
        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path_evaluations, "w") as json_file:
            json.dump(
                {f"Run_{i + 1}": performances[i][explainer] for i in range(no_of_runs)},
                json_file,
                indent=2,
            )

        if explainer not in ["PGExplainer", "SubGraphX"]:
            file_path_predictions = (
                f"results/predictions/{model}/{explainer}/{dataset}.json"
            )
            directory = os.path.dirname(file_path_predictions)
            # Create the directory if it doesn't exist
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(file_path_predictions, "w") as json_file:
                json.dump(
                    {f"Run_{i + 1}": preds[i][explainer] for i in range(no_of_runs)},
                    json_file,
                    indent=2,
                )

    file_path_predictions_df = f"results/predictions/{model}/{dataset}.json"
    directory = os.path.dirname(file_path_predictions_df)
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path_predictions_df, "w") as json_file:
        json.dump(predictions_data, json_file, indent=2)


# def run_explainers(dataset, explainers, print_explainer_loss=True, no_of_runs=5):
#     print(f"Running explainers  for {no_of_runs} runs. for dataset {dataset} ")
#     pg_performance = {}
#     sgx_performance = {}
#     evo_performance = {}
#     celoe_performance = {}
#     evo_preds = {}
#     celoe_preds = {}
#     predicition_data = {}

#     for i in range(no_of_runs):
#         my_explainer = Explainer(explainers=explainers, dataset="mutag")
#         evo_performance[i] = my_explainer.evaluations["EvoLearner"]
#         pg_performance[i] = my_explainer.evaluations["PGExplainer"]
#         celoe_performance[i] = my_explainer.evaluations["CELOE"]
#         sgx_performance[i] = my_explainer.evaluations["PGExplainer"]
#         evo_preds[i] = my_explainer.explanations["EvoLearner"]
#         celoe_preds[i] = my_explainer.explanations["CELOE"]
#         predicition_data[i] = my_explainer.pred_df.to_json()

#     file_path_evaluations_pg = f"results/evaluations/PGExplainer/{dataset}.json"
#     with open(file_path_evaluations_pg, "w") as json_file:
#         json.dump(pg_performance, json_file, indent=2)

#     file_path_evaluations_sgx = f"results/evaluations/SubGraphX/{dataset}.json"
#     with open(file_path_evaluations_sgx, "w") as json_file:
#         json.dump(sgx_performance, json_file, indent=2)

#     file_path_predictions_evo = f"results/predictions/EVO/{dataset}.json"
#     file_path_evaluations_evo = f"results/evaluations/EVO/{dataset}.json"

#     with open(file_path_predictions_evo, "w") as json_file:
#         json.dump(evo_preds, json_file, indent=2)

#     with open(file_path_evaluations_evo, "w") as json_file:
#         json.dump(evo_performance, json_file, indent=2)

#     file_path_predictions_celoe = f"results/predictions/CELOE/{dataset}.json"
#     file_path_evaluations_celoe = f"results/evaluations/CELOE/{dataset}.json"

#     with open(file_path_predictions_celoe, "w") as json_file:
#         json.dump(celoe_preds, json_file, indent=2)

#     with open(file_path_evaluations_celoe, "w") as json_file:
#         json.dump(celoe_performance, json_file, indent=2)
#     file_path_predicitions = f"results/predictions/{dataset}.json"
#     with open(file_path_predicitions, "w") as json_file:
#         json.dump(predicition_data, json_file, indent=2)
