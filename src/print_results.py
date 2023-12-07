import json
import os
import pandas as pd

from prettytable import PrettyTable


def read_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def print_data_as_table(data, save_to_df=True):
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
    rows = []
    for dataset, evaluations in data.items():
        for metric_name, metrics in evaluations.items():
            row = [
                str.upper(dataset.split("_")[1]),
                metrics["Model"],
                metrics["Metric"],
                metrics["evaluations"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
                metrics["jaccard_similarity"],
            ]
            rows.append(row)

    # Print the table
    table.add_rows(rows)
    print(table)
    # Create a DataFrame from the data
    if save_to_df:
        df = pd.DataFrame(rows, columns=table.field_names)

        # Write the DataFrame to a CSV file
        df.to_csv("eval_data.csv", index=False)


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


def print_results(to_csv=True):
    directory_path = (
        "results/evaluations/"  # Replace with the actual path to your directory
    )
    all_data = process_files_in_directory(directory_path)
    print_data_as_table(all_data, to_csv)
