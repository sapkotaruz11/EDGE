import csv
import json
import os
from urllib.parse import urlparse

import pandas as pd


def create_lp_aifb(file_path=None):
    if file_path is None:
        file_path = "data/aifb-hetero_82d021d8/testSet.tsv"

    # Read the TSV file into a DataFrame
    df = pd.read_csv(file_path, sep="\t")

    person_label_dict = dict(zip(df["person"], df["label_affiliation"]))

    reverse_dict = {values: [] for values in person_label_dict.values()}
    for key, values in person_label_dict.items():
        reverse_dict[values].append(key)

    classes_examples = {
        key.split("/")[-1]: {"positive_examples": values, "negative_examples": []}
        for key, values in reverse_dict.items()
    }

    for key, values in reverse_dict.items():
        for other_key, other_values in reverse_dict.items():
            if key != other_key:
                classes_examples[key.split("/")[-1]]["negative_examples"].extend(
                    other_values
                )
    for key, values in classes_examples.items():
        positive_set = set(values["positive_examples"])
        negative_set = set(values["negative_examples"])
        assert len(positive_set.intersection(negative_set)) == 0
    data = {"data_path": "data/KGs/aifb.owl", "problems": classes_examples}

    json_file_path = "configs/aifb.json"

    # Writing the dictionary to the JSON file with an indentation of 4 spaces

    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Learning Problem created for AIFB dataset and stored at {json_file_path}")


def create_lp_mutag(file_path=None):
    # Replace with the path to your CSV file
    if file_path is None:
        file_path = "data/mutag-hetero_faec5b61/testSet.tsv"
    df = pd.read_csv(file_path, delimiter="\t")

    # Separate the DataFrame into positive and negative examples

    positive_examples = df["bond"][df["label_mutagenic"] == 1].to_list()

    negative_examples = df["bond"][df["label_mutagenic"] == 0].to_list()
    assert len(set(positive_examples).intersection(set(negative_examples))) == 0

    example_dict = {
        "carcino": {
            "positive_examples": positive_examples,
            "negative_examples": negative_examples,
        }
    }

    data = {"data_path": "data/KGs/mutag.owl", "problems": example_dict}

    json_file_path = "configs/mutag.json"

    # Writing the dictionary to the JSON file with an indentation of 4 spaces

    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Learning Problem created for Mutag dataset and stored at {json_file_path}")
