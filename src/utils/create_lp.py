import csv
from urllib.parse import urlparse
import os
import json
import pandas as pd
import json
import os


def create_lp_aifb(file_path=None):
    if file_path is None:
        file_path = "data/aifb-hetero_82d021d8/completeDataset.tsv"
    classes_examples = {
        "id1instance": {"positive_examples": [], "negative_examples": []},
        "id2instance": {"positive_examples": [], "negative_examples": []},
        "id3instance": {"positive_examples": [], "negative_examples": []},
        "id4instance": {"positive_examples": [], "negative_examples": []},
    }

    with open(file_path, "r", newline="", encoding="utf-8") as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter="\t")

        for row in reader:
            person = row["person"]

            label = row["label_affiliation"]

            # Assuming that the URL is present in the 'label' column

            url = urlparse(label)

            # Extracting the list item from the URL path

            path_items = url.path.split("/")

            # Checking if '/idXinstance' is in the path items for each X (1, 2, 3, 4)

            for class_name in classes_examples.keys():
                if class_name == path_items[-1]:
                    # Positive example for the current class

                    classes_examples[class_name]["positive_examples"].append(person)

                else:
                    # Negative example for other classes

                    for other_class_name in classes_examples.keys():
                        if other_class_name != class_name:
                            classes_examples[other_class_name][
                                "negative_examples"
                            ].append(person)

    data = {"data_path": "data/KGs/aifb.owl", "problems": classes_examples}

    json_file_path = "configs/aifb.json"

    # Writing the dictionary to the JSON file with an indentation of 4 spaces

    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Learning Problem created for AIFB dataset and stored at {json_file_path}")


def create_lp_mutag(file_path=None):
    # Replace with the path to your CSV file
    if file_path is None:
        file_path = "data/mutag-hetero_faec5b61/completeDataset.tsv"
    df = pd.read_csv(file_path, delimiter="\t")

    # Separate the DataFrame into positive and negative examples

    positive_examples = df["bond"][df["label_mutagenic"] == 1].to_list()

    negative_examples = df["bond"][df["label_mutagenic"] == 0].to_list()

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
