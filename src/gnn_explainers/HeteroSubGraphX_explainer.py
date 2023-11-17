from src.gnn_explainers.dataset import RDFDatasets
from src.gnn_explainers.configs import get_configs
from src.gnn_explainers.model import RGCN
import torch as th
import torch.nn.functional as F
import torch
from src.gnn_explainers.hetro_features import HeteroFeature
from src.gnn_explainers.utils import get_nodes_dict
import time
import numpy as np
from torch import nn
import shutil
from dglnn_local.subgraphx import HeteroSubgraphX
import os


def explain(dataset="mutag"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = dataset
    root = f"data/{dataset}"
    my_dataset = RDFDatasets(dataset, root)
    configs = get_configs(dataset)
    g = my_dataset.g.to(device)
    category = my_dataset.category

    hidden_dim = configs["hidden_dim"]
    out_dim = my_dataset.num_classes
    e_types = g.etypes
    num_bases = configs["n_bases"]
    lr = configs["lr"]
    weight_decay = configs["weight_decay"]
    act = None
    input_feature = HeteroFeature({}, get_nodes_dict(g), hidden_dim, act=act).to(device)
    model = RGCN(hidden_dim, hidden_dim, out_dim, e_types, num_bases, category).to(
        device
    )

    # Define the optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = F.cross_entropy
    model.add_module("input_feature", input_feature)
    optimizer.add_param_group({"params": input_feature.parameters()})

    PATH = f"trained_models/{dataset}_trained.pt"
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    feat = model.input_feature()

    test_idx = my_dataset.test_idx.to(device)
    valid_idx = my_dataset.valid_idx.to(device)
    labels = my_dataset.labels.to(device)
    val_idx = torch.cat([test_idx, valid_idx], dim=0)
    gt = labels[val_idx].tolist()
    pred_logit = model(g, feat)[category]
    gnn_preds = pred_logit[val_idx].argmax(dim=1).tolist()
    gnn_pred = dict(zip(exp_preds.keys(), gnn_preds))
    gts = dict(zip(exp_preds.keys(), gt))

    explainer = HeteroSubgraphX(model, num_hops=1, num_rollouts=3, shapley_steps=10)
    exp_preds = {}
    for idx in val_idx.tolist():
        explanation, logits = explainer.explain_node(g, feat, idx, category)
        exp_preds[idx] = logits

    import json

    data = {
        "explanation_predictions": exp_preds,
        "gnn_preds": gnn_pred,
        "ground_truth": gts,
    }
    file_path = f"results/predictions/SubgraphX/{dataset}.json"

    # Write the data to the JSON file with formatting (indentation)
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=2)

    print(f"Data has been written to {file_path}")
