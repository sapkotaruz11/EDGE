import pandas as pd
import json
import csv
from urllib.parse import urlparse
import os
import json
import pandas as pd
import json
import os


def create_lp_aifb_train_test(train_file_path=None, test_file_path=None):
    if train_file_path is None:
        train_file_path = "data/aifb-hetero_82d021d8/trainingSet.tsv"
    if test_file_path is None:
        test_file_path = "data/aifb-hetero_82d021d8/testSet.tsv"

    # Read the training TSV file into a DataFrame
    train_df = pd.read_csv(train_file_path, sep="\t")

    # Read the test TSV file into a DataFrame
    test_df = pd.read_csv(test_file_path, sep="\t")

    # Create positive and negative examples for training set
    train_person_label_dict = dict(
        zip(train_df["person"], train_df["label_affiliation"])
    )

    train_reverse_dict = {values: [] for values in train_person_label_dict.values()}
    for key, values in train_person_label_dict.items():
        train_reverse_dict[values].append(key)

    train_classes_examples = {
        key.split("/")[-1]: {
            "positive_examples_train": values,
            "negative_examples_train": [],
            "positive_examples_test": [],
            "negative_examples_test": [],
        }
        for key, values in train_reverse_dict.items()
    }

    for key, values in train_reverse_dict.items():
        for other_key, other_values in train_reverse_dict.items():
            if key != other_key:
                train_classes_examples[key.split("/")[-1]][
                    "negative_examples_train"
                ].extend(other_values)

    # Create positive and negative examples for test set
    test_person_label_dict = dict(zip(test_df["person"], test_df["label_affiliation"]))

    test_reverse_dict = {values: [] for values in test_person_label_dict.values()}
    for key, values in test_person_label_dict.items():
        test_reverse_dict[values].append(key)

    for key, values in test_reverse_dict.items():
        train_classes_examples[key.split("/")[-1]]["positive_examples_test"] = values

        for other_key, other_values in test_reverse_dict.items():
            if key != other_key:
                train_classes_examples[key.split("/")[-1]][
                    "negative_examples_test"
                ].extend(other_values)

    for key, values in train_classes_examples.items():
        positive_set_train = set(values["positive_examples_train"])
        negative_set_train = set(values["negative_examples_train"])
        positive_set_test = set(values["positive_examples_test"])
        negative_set_test = set(values["negative_examples_test"])

        assert len(positive_set_train.intersection(negative_set_train)) == 0
        assert len(positive_set_test.intersection(negative_set_test)) == 0

    data = {
        "data_path": "data/KGs/aifb.owl",
        "problems": train_classes_examples,
    }

    json_file_path = "configs/aifb_train_test.json"

    # Writing the dictionary to the JSON file with an indentation of 4 spaces
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)

    print(
        f"Learning Problem created for AIFB dataset with train test split and stored at {json_file_path}"
    )


# Call the function with file paths if different from the default ones
# create_lp_aifb(train_file_path="path/to/trainSet.tsv", test_file_path="path/to/testSet.tsv")
import pandas as pd
import json


def create_lp_mutag_train_test(train_file_path=None, test_file_path=None):
    # Replace with the path to your CSV files
    if train_file_path is None:
        train_file_path = "data/mutag-hetero_faec5b61/trainingSet.tsv"
    if test_file_path is None:
        test_file_path = "data/mutag-hetero_faec5b61/testSet.tsv"

    # Read the training TSV file into a DataFrame
    train_df = pd.read_csv(train_file_path, delimiter="\t")

    # Read the test TSV file into a DataFrame
    test_df = pd.read_csv(test_file_path, delimiter="\t")

    # Separate the training DataFrame into positive and negative examples
    train_positive_examples = train_df["bond"][
        train_df["label_mutagenic"] == 1
    ].to_list()
    train_negative_examples = train_df["bond"][
        train_df["label_mutagenic"] == 0
    ].to_list()

    assert (
        len(set(train_positive_examples).intersection(set(train_negative_examples)))
        == 0
    )

    # Separate the test DataFrame into positive and negative examples
    test_positive_examples = test_df["bond"][test_df["label_mutagenic"] == 1].to_list()
    test_negative_examples = test_df["bond"][test_df["label_mutagenic"] == 0].to_list()

    assert (
        len(set(test_positive_examples).intersection(set(test_negative_examples))) == 0
    )

    example_dict = {
        "carcino": {
            "positive_examples_train": train_positive_examples,
            "negative_examples_train": train_negative_examples,
            "positive_examples_test": test_positive_examples,
            "negative_examples_test": test_negative_examples,
        }
    }

    data = {"data_path": "data/KGs/mutag.owl", "problems": example_dict}

    json_file_path = "configs/mutag_train_test.json"

    # Writing the dictionary to the JSON file with an indentation of 4 spaces
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)

    print(
        f"Learning Problem created for Mutag dataset with train test split and stored at {json_file_path}"
    )
