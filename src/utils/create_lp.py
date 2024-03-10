import csv
import json
import os
from urllib.parse import urlparse

import pandas as pd


def create_lp_aifb(train_file_path=None, test_file_path=None):
    """Function to create learning problem for the AIFB dataset.
    It the train/test file pathas are not provided, the default paths are selected.

    Finally, saves the learning problems in the specified file locations.
    """
    if train_file_path is None:
        train_file_path = "data/aifb-hetero_82d021d8/trainingSet.tsv"
    if test_file_path is None:
        test_file_path = "data/aifb-hetero_82d021d8/testSet.tsv"

    # Read the training TSV file into a DataFrame
    train_df = pd.read_csv(train_file_path, sep="\t")

    # Read the test TSV file into a DataFrame
    test_df = pd.read_csv(test_file_path, sep="\t")

    id1_uri = "http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id1instance"

    # Create lists of positive and negative examples based on the condition for training examples
    train_positive_examples = train_df.loc[
        train_df["label_affiliation"] == id1_uri, "person"
    ].tolist()
    train_negative_examples = train_df.loc[
        train_df["label_affiliation"] != id1_uri, "person"
    ].tolist()

    assert (
        len(set(train_positive_examples).intersection(set(train_negative_examples)))
        == 0
    )

    # Separate the test DataFrame into positive and negative examples
    test_positive_examples = test_df.loc[
        test_df["label_affiliation"] == id1_uri, "person"
    ].tolist()
    test_negative_examples = test_df.loc[
        test_df["label_affiliation"] != id1_uri, "person"
    ].tolist()

    assert (
        len(set(test_positive_examples).intersection(set(test_negative_examples))) == 0
    )

    example_dict = {
        "id1instance": {
            "positive_examples_train": train_positive_examples,
            "negative_examples_train": train_negative_examples,
            "positive_examples_test": test_positive_examples,
            "negative_examples_test": test_negative_examples,
        }
    }

    json_file_path = "configs/aifb.json"

    # Writing the dictionary to the JSON file with an indentation of 4 spaces
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(example_dict, json_file, indent=4)

    print(
        f"Learning Problem created for AIFB dataset with train test split and stored at {json_file_path}"
    )


def create_lp_mutag(train_file_path=None, test_file_path=None):
    """Function to create learning problem for the Mutag dataset.
    It the train/test file pathas are not provided, the default paths are selected.

    Finally, saves the learning problems in the specified file locations.
    """
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

    json_file_path = "configs/mutag.json"

    # Writing the dictionary to the JSON file with an indentation of 4 spaces
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(example_dict, json_file, indent=4)

    print(
        f"Learning Problem created for Mutag dataset with train test split and stored at {json_file_path}"
    )
