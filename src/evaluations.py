import json
from collections import defaultdict

EPSILON = 1e-10  # Small constant to avoid division by zero


def calculate_micro_metrics(y_true, y_pred, no_result_value=-1, EPSILON=1e-10):
    class_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

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
            class_metrics[no_result_value]["fp"] += 1

    for predicted_class in set(y_pred) - set(y_true):
        if predicted_class != no_result_value:
            class_metrics[predicted_class]["fp"] += 1

    micro_true_positives = sum(cm["tp"] for cm in class_metrics.values())
    micro_false_positives = sum(cm["fp"] for cm in class_metrics.values())
    micro_false_negatives = sum(cm["fn"] for cm in class_metrics.values())

    # Ensure non-zero denominators by checking for emptiness
    precision_denominator = micro_true_positives + micro_false_positives
    recall_denominator = micro_true_positives + micro_false_negatives

    # Calculate precision
    precision = (
        micro_true_positives / (precision_denominator + EPSILON)
        if precision_denominator > 0
        else 0.0
    )

    # Calculate recall
    recall = (
        micro_true_positives / (recall_denominator + EPSILON)
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
    jaccard_denominator = (
        micro_true_positives + micro_false_positives + micro_false_negatives
    )
    jaccard_similarity = (
        micro_true_positives / (jaccard_denominator + EPSILON)
        if jaccard_denominator > 0
        else 0.0
    )

    return precision, recall, f1_score, jaccard_similarity


def calculate_metrics(y_true, y_pred):
    true_positives = sum(
        (true == 1 and pred == 1) for true, pred in zip(y_true, y_pred)
    )
    false_positives = sum(
        (true == 0 and pred == 1) for true, pred in zip(y_true, y_pred)
    )
    false_negatives = sum(
        (true == 1 and pred == 0) for true, pred in zip(y_true, y_pred)
    )

    epsilon = 1e-10  # Small value to avoid division by zero

    # Calculate precision
    precision_denominator = true_positives + false_positives
    precision = (
        true_positives / (precision_denominator + epsilon)
        if precision_denominator > 0
        else 0.0
    )

    # Calculate recall
    recall_denominator = true_positives + false_negatives
    recall = (
        true_positives / (recall_denominator + epsilon)
        if recall_denominator > 0
        else 0.0
    )

    # Calculate F1 score
    f1_denominator = precision + recall
    f1_score = (
        2 * (precision * recall) / (f1_denominator + epsilon)
        if f1_denominator > 0
        else 0.0
    )

    # Calculate Jaccard similarity
    jaccard_denominator = true_positives + false_positives + false_negatives
    jaccard_similarity = (
        true_positives / (jaccard_denominator + epsilon)
        if jaccard_denominator > 0
        else 0.0
    )

    return precision, recall, f1_score, jaccard_similarity


def evaluate_gnn_explainers(
    datasets=["aifb", "mutag"], explainers=["PGExplainer", "SubGraphX"]
):
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

            exp_preds_list = [value["exp_preds"] for value in predictions_data.values()]
            gnn_pred_list = [value["gnn_pred"] for value in predictions_data.values()]
            gts_list = [value["gts"] for value in predictions_data.values()]

            # Calculate macro scores for precision, recall, F1 score, and Jaccard similarity
            if dataset == "aifb":
                (
                    precision,
                    recall,
                    f1_score,
                    jaccard_similarity,
                ) = calculate_micro_metrics(gts_list, gnn_pred_list)

            else:
                (
                    precision,
                    recall,
                    f1_score,
                    jaccard_similarity,
                ) = calculate_micro_metrics(gts_list, gnn_pred_list)

            eval_preds = {
                "Model": "Hetero-RGCN",
                "Metric": "Prediction Accuracy",
                "evaluations" : "micro",
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "jaccard_similarity": jaccard_similarity,
            }
            if dataset == "aifb":
                (
                    precision,
                    recall,
                    f1_score,
                    jaccard_similarity,
                ) = calculate_micro_metrics(gnn_pred_list, exp_preds_list)

            else:
                (
                    precision,
                    recall,
                    f1_score,
                    jaccard_similarity,
                ) = calculate_micro_metrics(gnn_pred_list, exp_preds_list)

            eval_fids = {
                "Model": explainer,
                "Metric": "Explanation Fidelity",
                "evaluations" : "micro",
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "jaccard_similarity": jaccard_similarity,
            }
            if dataset == "aifb":
                (
                    precision,
                    recall,
                    f1_score,
                    jaccard_similarity,
                ) = calculate_micro_metrics(gts_list, exp_preds_list)

            else:
                (
                    precision,
                    recall,
                    f1_score,
                    jaccard_similarity,
                ) = calculate_micro_metrics(gts_list, exp_preds_list)

            eval_exp_acc = {
                "Model": explainer,
                "Metric": "Explanation Accuracy",
                "evaluations" : "micro",
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "jaccard_similarity": jaccard_similarity,
            }

            nested_dict = {
                "eval_pred": eval_preds,
                "eval_fid": eval_fids,
                "eval_expl_acc": eval_exp_acc,
            }
            results[dataset] = nested_dict

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
            precision, recall, f1_score, jaccard_similarity = calculate_micro_metrics(
                gts, preds
            )

            return precision, recall, f1_score, jaccard_similarity

    micro_true_positives = 0
    micro_false_positives = 0
    micro_false_negatives = 0

    EPSILON = 1e-10  # or any small positive number

    for _, examples in predictions_data.items():
        concept_individuals = set(examples["concept_individuals"])
        pos = set(examples["positive_examples"])

        true_positives = len(pos.intersection(concept_individuals))
        false_positives = len(concept_individuals.difference(pos))
        false_negatives = len(pos.difference(concept_individuals))

        micro_true_positives += true_positives
        micro_false_negatives += false_negatives
        micro_false_positives += false_positives

    # Ensure non-zero denominators by checking for emptiness
    precision_denominator = micro_true_positives + micro_false_positives
    recall_denominator = micro_true_positives + micro_false_negatives

    # Calculate precision
    precision = (
        micro_true_positives / (precision_denominator + EPSILON)
        if precision_denominator > 0
        else 0.0
    )

    # Calculate recall
    recall = (
        micro_true_positives / (recall_denominator + EPSILON)
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
    jaccard_denominator = (
        micro_true_positives + micro_false_positives + micro_false_negatives
    )
    jaccard_similarity = (
        micro_true_positives / (jaccard_denominator + EPSILON)
        if jaccard_denominator > 0
        else 0.0
    )

    return precision, recall, f1_score, jaccard_similarity


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
                "evaluations" : "micro",
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
                "evaluations" : "micro",
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
