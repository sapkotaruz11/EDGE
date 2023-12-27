import json
import os
import shutil
import time

import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from torch import nn

from src.dglnn_local.subgraphx import HeteroSubgraphX
from src.gnn_explainers.configs import get_configs
from src.gnn_explainers.dataset import RDFDatasets
from src.gnn_explainers.hetro_features import HeteroFeature
from src.gnn_explainers.model import RGCN
from src.gnn_explainers.trainer import train_gnn
from src.gnn_explainers.utils import get_nodes_dict


def explain_SGX(dataset="mutag"):
    """
    Trains and explains a graph neural network (GNN) model using SubGraphX (SGX) on the specified RDF dataset.

    This function loads a pre-trained GNN model or trains one if not already available.
    It then applies the SubGraphX explainer to generate explanations for the model's predictions on the test dataset. The results are saved in a JSON file for further analysis.

    Parameters:
        dataset (str, optional): The name of the RDF dataset to be used. Defaults to 'mutag'.

    Returns:
        None. The function writes the explanation results to a JSON file in the 'results/predictions/SubGraphX' directory.

    Raises:
        FileNotFoundError: If the trained GNN model file is not found.

    Example:
        >>> explain_SGX(dataset="mutag")
        # This will load or train the GNN model for the MUTAG dataset and save the SubGraphX explanations in a JSON file.

    Notes:
        - The function uses CUDA for acceleration if available.
        - It includes the model's predictions, ground truth labels, and entity information for each test node.
        - The SubGraphX explainer is used to interpret the GNN model's decisions for each node in the test set.
    """
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = str.lower(dataset)

    configs = get_configs(dataset)
    hidden_dim = configs["hidden_dim"]
    num_bases = configs["n_bases"]
    lr = configs["lr"]
    weight_decay = configs["weight_decay"]
    epochs = configs["max_epoch"]
    validation = configs["validation"]
    hidden_layers = configs["num_layers"] - 1
    act = None

    # build dataset
    my_dataset = RDFDatasets(dataset, root="data/", validation=validation)
    g = my_dataset.g.to(device)
    out_dim = my_dataset.num_classes
    e_types = g.etypes
    category = my_dataset.category

    idx_map = my_dataset.idx_map
    # build model with dataset specific configs
    input_feature = HeteroFeature({}, get_nodes_dict(g), hidden_dim, act=act).to(device)
    model = RGCN(
        hidden_dim,
        hidden_dim,
        out_dim,
        e_types,
        num_bases,
        category,
        num_hidden_layers=hidden_layers,
    ).to(device)

    # Define the optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = F.cross_entropy
    model.add_module("input_feature", input_feature)
    optimizer.add_param_group({"params": input_feature.parameters()})

    PATH = f"trained_models/{dataset}_trained.pt"
    if not os.path.isfile(PATH):
        train_gnn(dataset=dataset, PATH=PATH)

    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    feat = model.input_feature()

    test_idx = my_dataset.test_idx.to(device)
    labels = my_dataset.labels.to(device)
    gt = labels[test_idx].tolist()
    pred_logit = model(g, feat)[category]
    gnn_preds = pred_logit[test_idx].argmax(dim=1).tolist()

    print("Starting SubGraphX")
    t0 = time.time()
    explainer = HeteroSubgraphX(model, num_hops=1, num_rollouts=3, shapley_steps=10)
    exp_preds = {}
    entity = {}
    for idx in test_idx.tolist():
        print(idx)
        explanation, logits = explainer.explain_node(g, feat, idx, category)
        exp_preds[idx] = logits
        entity[idx] = idx_map[idx]["IRI"]

    gnn_pred = dict(zip(exp_preds.keys(), gnn_preds))
    gts = dict(zip(exp_preds.keys(), gt))

    dict_names = ["exp_preds", "gnn_pred", "gts", "entity"]
    list_of_dicts = [exp_preds, gnn_pred, gts, entity]

    nested_dict = {
        key: {name: d[key] for name, d in zip(dict_names, list_of_dicts)}
        for key in exp_preds
    }
    file_path = f"results/predictions/SubGraphX/{dataset}.json"

    # Write the data to the JSON file with formatting (indentation)
    with open(file_path, "w") as json_file:
        json.dump(nested_dict, json_file, indent=2)

    print(f"Data has been written to {file_path}")
    print("Ending SubGraphX")
    t1 = time.time()
    dur = t1 - t0
    print(f"Total time taken for SubGraphX  on {dataset}: {dur:.2f}")
