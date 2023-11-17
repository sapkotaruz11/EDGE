import json
import os

from ontolearn.metrics import F1, Accuracy, Precision, Recall


def get_positive_negative_examples(kgs):
    positive_examples_dict = {}  # Dictionary to store positive examples for each kg
    negative_examples_dict = {}
    for kg in kgs:
        json_file_path = f"configs/{kg}.json"  # Replace with your JSON file path

        # Open and read the JSON file line by line
        with open(json_file_path, "r") as json_file:
            settings = json.load(json_file)

        # Extract positive examples and add them to the dictionary
        positive_examples_dict[kg] = set(settings["positive_examples"])
        negative_examples_dict[kg] = set(settings["negative_examples"])

    return positive_examples_dict, negative_examples_dict


def evaluate_logical_explainers(KGs=None, classifiers=None):
    KGs = ["mutag", "lymphography"]
    classifiers = ["CELOE", "EVO"]
    f1 = F1().score2
    accuracy = Accuracy().score2
    precision = Precision().score2
    recall = Recall().score2
    actuals_pos, actuals_neg = get_positive_negative_examples(KGs)
    preds_base_path = "results/predictions/"
    evaluations_base_path = "results/evaluations"
    for classifier in classifiers:
        classifier_results = {}
        classifier_base_path = os.path.join(preds_base_path, classifier)
        for kg in KGs:
            classifier_predictions_path = f"{classifier_base_path}/{kg}.json"
            # Open and read the JSON file
            with open(classifier_predictions_path, "r") as json_file:
                classifier_data = json.load(json_file)
            # Assuming the JSON structure includes a key "key_concept_individuals"
            concept_individuals = classifier_data.get("concept_individuals", [])
            pos = actuals_pos[kg]
            neg = actuals_neg[kg]
            tp = len(pos.intersection(concept_individuals))
            fn = len(pos.difference(concept_individuals))
            fp = len(neg.intersection(concept_individuals))
            tn = len(neg.difference(concept_individuals))
            acc = 100 * accuracy(tp, fn, fp, tn)[-1]
            prec = 100 * precision(tp, fn, fp, tn)[-1]
            rec = 100 * recall(tp, fn, fp, tn)[-1]
            f_1 = 100 * f1(tp, fn, fp, tn)[-1]
            # Store the results in the classifier_results dictionary
            classifier_results[kg] = {
                "ACC": acc,
                "precision": prec,
                "recall": rec,
                "F1": f_1,
            }
        evaluations_path = f"{evaluations_base_path}/{classifier}.json"
        with open(evaluations_path, "w") as json_output:
            json.dump(classifier_results, json_output, indent=4)
