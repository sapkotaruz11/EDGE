import json

from tabulate import tabulate


def print_results(results_base=None):
    results_base = "results/evaluations"
    classifiers = ["CELOE", "EVO"]
    table_data = []
    for classifier in classifiers:
        results_path = f"{results_base}/{classifier}.json"
        with open(results_path, "r") as json_file:
            classifier_results = json.load(json_file)

        # Prepare the data for tabulation

        for kg, metrics in classifier_results.items():
            table_data.append(
                [
                    classifier,
                    kg,
                    f"{metrics['ACC']:.2f}%",
                    f"{metrics['precision']:.2f}%",
                    f"{metrics['recall']:.2f}%",
                    f"{metrics['F1']:.2f}%",
                ]
            )
    headers = ["Classifier", "KG", "ACC", "Precision", "Recall", "F1 Score"]

    # Print the results in a tabular format
    table = tabulate(table_data, headers, tablefmt="grid")
    print(table)
