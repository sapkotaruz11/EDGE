import json
import os
from collections import defaultdict

from ontolearn.metrics import F1, Accuracy, Precision, Recall

EPSILON = 1e-10  # Small constant to avoid division by zero


def calculate_macro_metrics(dict1, dict2):
    classes = set(dict1.values()) | set(dict2.values())
    class_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for key in dict1.keys():
        true_class = dict1[key]
        if key in dict2:
            predicted_class = dict2[key]
            if true_class == predicted_class:
                class_metrics[true_class]["tp"] += 1
            else:
                class_metrics[true_class]["fn"] += 1
                class_metrics[predicted_class]["fp"] += 1
        else:
            class_metrics[true_class]["fn"] += 1

    for key in dict2.keys():
        if key not in dict1:
            predicted_class = dict2[key]
            class_metrics[predicted_class]["fp"] += 1

    macro_precision = 0
    macro_recall = 0
    macro_f1_score = 0
    macro_jaccard_similarity = 0

    for cls in classes:
        tp = class_metrics[cls]["tp"]
        fp = class_metrics[cls]["fp"]
        fn = class_metrics[cls]["fn"]

        precision = tp / (tp + fp + EPSILON)
        recall = tp / (tp + fn + EPSILON)
        f1_score = 2 * (precision * recall) / (precision + recall + EPSILON)
        jaccard_similarity = tp / (tp + fp + fn + EPSILON)

        macro_precision += precision
        macro_recall += recall
        macro_f1_score += f1_score
        macro_jaccard_similarity += jaccard_similarity

    num_classes = len(classes)
    macro_precision /= num_classes
    macro_recall /= num_classes
    macro_f1_score /= num_classes
    macro_jaccard_similarity /= num_classes

    return macro_precision, macro_recall, macro_f1_score, macro_jaccard_similarity


def calculate_metrics(dict1, dict2):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for key in dict1.keys():
        if key in dict2:
            if dict1[key] == dict2[key]:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            false_negatives += 1

    for key in dict2.keys():
        if key not in dict1:
            false_positives += 1

    precision = true_positives / (true_positives + false_positives + EPSILON)
    recall = true_positives / (true_positives + false_negatives + EPSILON)
    f1_score = 2 * (precision * recall) / (precision + recall + EPSILON)
    jaccard_similarity = true_positives / (
        true_positives + false_positives + false_negatives + EPSILON
    )

    return precision, recall, f1_score, jaccard_similarity


def evaluate_gnn_explainers(datasets=["aifb"], explainers=["PGExplainer"], PATH=None):
    if explainers is None:
        explainers = ["PGExplainer", "SubGraphX"]
    for explainer in explainers:
        results = {}
        if datasets is None:
            datasets = ["aifb", "mutag"]
        for dataset in datasets:
            # Specify the path to your JSON file
            file_path = f"results/predictions/{explainer}/{dataset}.json"

            # Open and read the JSON file
            with open(file_path, "r") as file:
                predictions_data = json.load(file)

            exp_preds_dict = {
                key: value["exp_preds"] for key, value in predictions_data.items()
            }
            gnn_pred_dict = {
                key: value["gnn_pred"] for key, value in predictions_data.items()
            }
            gts_dict = {key: value["gts"] for key, value in predictions_data.items()}
            entity_dict = {
                key: value["entity"] for key, value in predictions_data.items()
            }

            # Calculate macro scores for precision, recall, F1 score, and Jaccard similarity
            (
                macro_precision,
                macro_recall,
                macro_f1_score,
                macro_jaccard_similarity,
            ) = calculate_macro_metrics(gts_dict, gnn_pred_dict)
            precision, recall, f1_score, jaccard_similarity = calculate_metrics(
                gts_dict, gnn_pred_dict
            )
            dataset_evals_macro = {
                "Model": "Hetero-RGCN-macro",
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1_score": macro_f1_score,
                "macro_jaccard_similarity": macro_jaccard_similarity,
            }
            dataset_evals = {
                "Model": "Hetero-RGCN",
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "jaccard_similarity": jaccard_similarity,
            }

            (
                macro_precision,
                macro_recall,
                macro_f1_score,
                macro_jaccard_similarity,
            ) = calculate_macro_metrics(gnn_pred_dict, exp_preds_dict)
            precision, recall, f1_score, jaccard_similarity = calculate_metrics(
                gnn_pred_dict, exp_preds_dict
            )

            dataset_evals_macro_fid = {
                "Model": explainer,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1_score": macro_f1_score,
                "macro_jaccard_similarity": macro_jaccard_similarity,
            }

            dataset_evals_fid = {
                "Model": explainer,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "jaccard_similarity": jaccard_similarity,
            }

            nested_dict = {
                "eval_pred": dataset_evals,
                "macro_eval_pred": dataset_evals_macro,
                "eval_fid": dataset_evals_fid,
                "eval_macro_fid": dataset_evals_macro_fid,
            }
            results[dataset] = nested_dict

        file_path = f"results/evaluations/{explainer}.json"

        with open(file_path, "w") as json_file:
            json.dump(results, json_file, indent=2)


def evalate_logical_explainers(explainers=["EVO", "CELOE"], KGs=["mutag", "aifb"]):
    explainers = ["EVO", "CELOE"]
    results = {}
    for explainer in explainers:
        KGs = ["mutag", "aifb"]

        for kg in KGs:
            file_path = f"results/predictions/{explainer}/{kg}.json"

            with open(file_path, "r") as file:
                predictions_data = json.load(file)
            EPSILON = 1e-10  # Small constant to avoid division by zero
            precision = 0
            recall = 0
            f1_score = 0
            jaccard_similarity = 0

            for learning_problem, examples in predictions_data.items():
                concept_individuals = set(examples["concept_individuals"])
                pos = set(examples["positive_examples"])
                neg = set(examples["positive_examples"])

                true_positives = len(pos.intersection(concept_individuals))
                false_negatives = len(pos.difference(concept_individuals))
                false_positives = len(neg.intersection(concept_individuals))
                precision += true_positives / (
                    true_positives + false_positives + EPSILON
                )
                recall += true_positives / (true_positives + false_negatives + EPSILON)
                f1_score += 2 * (precision * recall) / (precision + recall + EPSILON)
                jaccard_similarity += true_positives / (
                    true_positives + false_positives + false_negatives + EPSILON
                )

            macro_eval_pred = {
                "Model": explainer,
                "macro_precision": precision,
                "macro_recall": recall,
                "macro_f1_score": f1_score,
                "macro_jaccard_similarity": jaccard_similarity,
            }

            file_path = f"results/predictions/{explainer}/{kg}_gnn_preds.json"

            with open(file_path, "r") as file:
                predictions_data = json.load(file)
            EPSILON = 1e-10  # Small constant to avoid division by zero
            precision = 0
            recall = 0
            f1_score = 0
            jaccard_similarity = 0

            for learning_problem, examples in predictions_data.items():
                concept_individuals = set(examples["concept_individuals"])
                pos = set(examples["positive_examples"])
                neg = set(examples["positive_examples"])

                true_positives = len(pos.intersection(concept_individuals))
                false_negatives = len(pos.difference(concept_individuals))
                false_positives = len(neg.intersection(concept_individuals))
                precision += true_positives / (
                    true_positives + false_positives + EPSILON
                )
                recall += true_positives / (true_positives + false_negatives + EPSILON)
                f1_score += 2 * (precision * recall) / (precision + recall + EPSILON)
                jaccard_similarity += true_positives / (
                    true_positives + false_positives + false_negatives + EPSILON
                )

            macro_eval_fed = {
                "Model": explainer,
                "macro_precision": precision,
                "macro_recall": recall,
                "macro_f1_score": f1_score,
                "macro_jaccard_similarity": jaccard_similarity,
            }

            results[kg] = {
                "macro_eval_fed": macro_eval_fed,
                "macro_eval_pred": macro_eval_pred,
            }

        file_path = f"results/evaluations/{explainer}.json"

        with open(file_path, "w") as json_file:
            json.dump(results, json_file, indent=2)
