import json
import os

from prettytable import PrettyTable


def read_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def print_data_as_table(data):
    table = PrettyTable()
    table.field_names = [
        "Dataset",
        "Model",
        "Metric",
        "Evaluations",
        "Precision",
        "Recall",
        "F1 Score",
        "Jaccard Similarity",
    ]

    for dataset, evaluations in data.items():
        for metric_name, metrics in evaluations.items():
            table.add_row(
                [
                    str.upper(dataset.split("_")[1]),
                    metrics["Model"],
                    metrics["Metric"],
                    metrics["Evaluations"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1_score"],
                    metrics["jaccard_similarity"],
                ]
            )
            table.add_row(["-------"] * len(table.field_names))

    print(table)


def process_files_in_directory(directory):
    all_data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            data = read_json_file(file_path)
            # Use filename as a prefix to avoid key conflicts
            prefixed_data = {f"{filename}_{key}": value for key, value in data.items()}
            all_data.update(prefixed_data)
    return all_data


def print_results():
    directory_path = (
        "results/evaluations/"  # Replace with the actual path to your directory
    )
    all_data = process_files_in_directory(directory_path)
    print_data_as_table(all_data)
