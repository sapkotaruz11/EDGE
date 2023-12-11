import json
from collections import defaultdict


def calculate_macro_metrics(y_true, y_pred, no_result_value=-1, EPSILON=1e-10):
    class_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    # Calculate per-class metrics
    for true_class, predicted_class in zip(y_true, y_pred):
        if true_class != no_result_value and predicted_class != no_result_value:
            if true_class == predicted_class:
                class_metrics[true_class]["tp"] += 1
            else:
                class_metrics[true_class]["fn"] += 1
                class_metrics[predicted_class]["fp"] += 1
        elif true_class != no_result_value:
            # Consider "no result" as both false positive and false negative
            class_metrics[true_class]["fn"] += 1
            # class_metrics[no_result_value]["fp"] += 1

    for predicted_class in set(y_pred) - set(y_true):
        if predicted_class != no_result_value:
            class_metrics[predicted_class]["fp"] += 1

    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    macro_jaccard = 0.0

    # Calculate macro-average metrics
    for cls in class_metrics:
        tp = class_metrics[cls]["tp"]
        fp = class_metrics[cls]["fp"]
        fn = class_metrics[cls]["fn"]

        precision = tp / (tp + fp + EPSILON) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn + EPSILON) if tp + fn > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall + EPSILON)
            if precision + recall > 0
            else 0.0
        )
        jaccard = tp / (tp + fp + fn + EPSILON) if tp + fp + fn > 0 else 0.0

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        macro_jaccard += jaccard

    num_classes = len(class_metrics)

    macro_precision /= num_classes
    macro_recall /= num_classes
    macro_f1 /= num_classes
    macro_jaccard /= num_classes

    return macro_precision, macro_recall, macro_f1, macro_jaccard


import json
from typing import List, Optional


def evaluate_gnn_explainers(
    datasets: Optional[List[str]] = None,
    explainers: Optional[List[str]] = None,
):
    # Default values if not provided
    if explainers is None:
        explainers = ["PGExplainer", "SubGraphX"]
    if datasets is None:
        datasets = ["aifb", "mutag"]

    for explainer in explainers:
        results = {}

        for dataset in datasets:
            # Specify the path to your JSON file
            file_path = f"results/predictions/{explainer}/{dataset}.json"

            # Open and read the JSON file
            with open(file_path, "r") as file:
                predictions_data = json.load(file)

            exp_preds_list = [value["exp_preds"] for value in predictions_data.values()]
            gnn_pred_list = [value["gnn_pred"] for value in predictions_data.values()]
            gts_list = [value["gts"] for value in predictions_data.values()]

            # Calculate macro scores for precision, recall, F1 score, and Jaccard similarity
            (
                precision_gnn,
                recall_gnn,
                f1_score_gnn,
                jaccard_similarity_gnn,
            ) = calculate_macro_metrics(gts_list, gnn_pred_list)

            (
                precision_exp,
                recall_exp,
                f1_score_exp,
                jaccard_similarity_exp,
            ) = calculate_macro_metrics(gnn_pred_list, exp_preds_list)

            (
                precision_fid,
                recall_fid,
                f1_score_fid,
                jaccard_similarity_fid,
            ) = calculate_macro_metrics(gts_list, exp_preds_list)

            # Evaluation metrics for Prediction Accuracy
            eval_preds = {
                "Model": "Hetero-RGCN",
                "Metric": "Prediction Accuracy",
                "evaluations" : "macro",
                "precision": precision_gnn,
                "recall": recall_gnn,
                "f1_score": f1_score_gnn,
                "jaccard_similarity": jaccard_similarity_gnn,
            }

            # Evaluation metrics for Explanation Fidelity
            eval_fids = {
                "Model": explainer,
                "Metric": "Explanation Fidelity",
                "evaluations" : "macro",
                "precision": precision_fid,
                "recall": recall_fid,
                "f1_score": f1_score_fid,
                "jaccard_similarity": jaccard_similarity_fid,
            }

            # Evaluation metrics for Explanation Accuracy
            eval_exp_acc = {
                "Model": explainer,
                "Metric": "Explanation Accuracy",
                "evaluations" : "macro",
                "precision": precision_exp,
                "recall": recall_exp,
                "f1_score": f1_score_exp,
                "jaccard_similarity": jaccard_similarity_exp,
            }

            # Create a nested dictionary for results
            nested_dict = {
                "eval_pred": eval_preds,
                "eval_fid": eval_fids,
                "eval_expl_acc": eval_exp_acc,
            }

            results[dataset] = nested_dict

        # Save the results to a JSON file
        file_path = f"results/evaluations/{explainer}.json"
        with open(file_path, "w") as json_file:
            json.dump(results, json_file, indent=2)


def calculate_metrics_logical(predictions_data):
    num_problems = len(predictions_data)

    if num_problems == 1:
        for _, examples in predictions_data.items():
            concept_individuals = set(examples["concept_individuals"])
            pos = set(examples["positive_examples"])
            neg = set(examples["negative_examples"])

            all_examples = pos.union(neg)

            gts = [1 if item in pos else 0 for item in all_examples]
            preds = [1 if item in concept_individuals else 0 for item in all_examples]
            precision, recall, f1_score, jaccard_similarity = calculate_macro_metrics(
                gts, preds
            )

            return precision, recall, f1_score, jaccard_similarity

    macro_precision = 0
    macro_recall = 0
    macro_f1_score = 0
    macro_jaccard_similarity = 0

    EPSILON = 1e-10  # or any small positive number

    for _, examples in predictions_data.items():
        concept_individuals = set(examples["concept_individuals"])
        pos = set(examples["positive_examples"])

        true_positives = len(pos.intersection(concept_individuals))
        false_positives = len(concept_individuals.difference(pos))
        false_negatives = len(pos.difference(concept_individuals))

        # Ensure non-zero denominators by checking for emptiness
        precision_denominator = true_positives + false_positives
        recall_denominator = true_positives + false_negatives

        # Calculate precision
        precision = (
            true_positives / (precision_denominator + EPSILON)
            if precision_denominator > 0
            else 0.0
        )

        # Calculate recall
        recall = (
            true_positives / (recall_denominator + EPSILON)
            if recall_denominator > 0
            else 0.0
        )

        # Calculate F1 score
        f1_denominator = precision + recall
        f1_score = (
            2 * (precision * recall) / (f1_denominator + EPSILON)
            if f1_denominator > 0
            else 0.0
        )

        # Calculate Jaccard similarity
        jaccard_denominator = true_positives + false_positives + false_negatives
        jaccard_similarity = (
            true_positives / (jaccard_denominator + EPSILON)
            if jaccard_denominator > 0
            else 0.0
        )

        # Accumulate values for macro-average
        macro_precision += precision
        macro_recall += recall
        macro_f1_score += f1_score
        macro_jaccard_similarity += jaccard_similarity

    # Calculate macro-average
    macro_precision /= num_problems
    macro_recall /= num_problems
    macro_f1_score /= num_problems
    macro_jaccard_similarity /= num_problems

    return macro_precision, macro_recall, macro_f1_score, macro_jaccard_similarity


def evaluate_logical_explainers(explainers=None, KGs=None):
    if explainers is None:
        explainers = ["EVO", "CELOE"]

    results = {}

    for explainer in explainers:
        if KGs is None:
            KGs = ["mutag", "aifb"]

        for kg in KGs:
            # Evaluate prediction accuracy
            file_path = f"results/predictions/{explainer}/{kg}.json"

            with open(file_path, "r") as file:
                predictions_data = json.load(file)
            micro_metrics = calculate_metrics_logical(predictions_data)
            eval_pred_acc = {
                "Model": explainer,
                "Metric": "Prediction Accuracy",
                "evaluations" : "macro",
                "precision": micro_metrics[0],
                "recall": micro_metrics[1],
                "f1_score": micro_metrics[2],
                "jaccard_similarity": micro_metrics[3],
            }

            # Evaluate fidelity
            file_path_fid = f"results/predictions/{explainer}/{kg}_gnn_preds.json"
            file_path_gts = f"configs/{kg}.json"

            with open(file_path_fid, "r") as file:
                predictions_data_fid = json.load(file)

            micro_metrics_gts = calculate_metrics_logical(predictions_data_fid)
            eval_fid = {
                "Model": explainer,
                "Metric": "Fidelity",
                "evaluations" : "macro",
                "precision": micro_metrics_gts[0],
                "recall": micro_metrics_gts[1],
                "f1_score": micro_metrics_gts[2],
                "jaccard_similarity": micro_metrics_gts[3],
            }

            results[kg] = {
                "eval_fid": eval_fid,
                "eval_expl_acc": eval_pred_acc,
            }

        # Save results to file
        file_path = f"results/evaluations/{explainer}.json"
        with open(file_path, "w") as json_file:
            json.dump(results, json_file, indent=2)
