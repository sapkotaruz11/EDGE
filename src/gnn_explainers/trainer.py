import json
import os
import time

import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from torch import nn

from src.gnn_explainers.configs import get_configs
from src.gnn_explainers.dataset import RDFDatasets
from src.gnn_explainers.hetro_features import HeteroFeature
from src.gnn_explainers.model import RGCN
from src.gnn_explainers.utils import get_nodes_dict


def get_lp_aifb(gnn_pred_dt, idx_map):
    class_to_pred = {}
    multi_lp_dict = {}

    # Creating new_dict
    for key, value in gnn_pred_dt.items():
        class_to_pred.setdefault(value, []).append(key)

    # Creating news_dict
    multi_lp_dict = {
        f"{key+1}instances": {
            "positive_examples": [idx_map[val]["IRI"] for val in values],
            "negative_examples": [
                idx_map[val]["IRI"]
                for k, v in class_to_pred.items()
                if k != key
                for val in v
            ],
        }
        for key, values in class_to_pred.items()
    }

    return multi_lp_dict


def get_lp_mutag(gnn_pred_dt, idx_map):
    positive_examples = [
        idx_map[item]["IRI"] for item in gnn_pred_dt if gnn_pred_dt[item] == 1
    ]
    negative_examples = [
        idx_map[item]["IRI"] for item in gnn_pred_dt if gnn_pred_dt[item] != 0
    ]

    lp_dict = {
        "positive_examples": positive_examples,
        "negative_examples": negative_examples,
    }
    return lp_dict


def train_gnn(dataset="mutag", device=None, PATH=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = dataset
    my_dataset = RDFDatasets(dataset)
    configs = get_configs(dataset)
    g = my_dataset.g.to(device)
    category = my_dataset.category

    hidden_dim = configs["hidden_dim"]
    out_dim = my_dataset.num_classes
    e_types = g.etypes
    num_bases = configs["n_bases"]
    lr = configs["lr"]
    weight_decay = configs["weight_decay"]
    epochs = configs["max_epoch"]
    act = None

    train_idx = my_dataset.train_idx.to(device)
    valid_idx = my_dataset.valid_idx.to(device)
    test_idx = my_dataset.test_idx.to(device)
    labels = my_dataset.labels.to(device)

    idx_map = my_dataset.idx_map
    val_idx = torch.cat([test_idx, valid_idx], dim=0)
    pred_idx = torch.cat([train_idx, val_idx], dim=0)
    # input_feature = HeteroFeature({}, get_nodes_dict(g), hidden_dim,
    #                                             act=act).to(device)
    input_feature = HeteroFeature({}, get_nodes_dict(g), hidden_dim, act=act).to(device)
    model = RGCN(hidden_dim, hidden_dim, out_dim, e_types, num_bases, category).to(
        device
    )

    # Define the optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = F.cross_entropy
    model.add_module("input_feature", input_feature)
    optimizer.add_param_group({"params": input_feature.parameters()})

    print("Start training...")
    dur = []
    h_dict = model.input_feature()
    model.train()
    for epoch in range(epochs):
        t0 = time.time()
        logits = model(g, h_dict)[category]
        loss = loss_fn(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        t1 = time.time()

        dur.append(t1 - t0)
        train_acc = th.sum(
            logits[train_idx].argmax(dim=1) == labels[train_idx]
        ).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[valid_idx], labels[valid_idx])
        val_acc = th.sum(
            logits[valid_idx].argmax(dim=1) == labels[valid_idx]
        ).item() / len(valid_idx)
        print(
            "Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".format(
                epoch,
                train_acc,
                loss.item(),
                val_acc,
                val_loss.item(),
                np.average(dur),
            )
        )
    print("End Training")
    print(
        "Creating Learning Problems for Logical Explainers based on GNN Predictions to Calculate Fidelity"
    )
    pred_logit = model(g, h_dict)[category]
    gnn_preds_4lp = pred_logit[pred_idx].argmax(dim=1).tolist()
    gnn_pred_dt = {tensor.item(): pred for tensor, pred in zip(pred_idx, gnn_preds_4lp)}
    if dataset == "mutag":
        lp_data = get_lp_mutag(gnn_pred_dt, idx_map)
    if dataset == "aifb":
        lp_data = get_lp_aifb(gnn_pred_dt, idx_map)

    # File path where you want to store the JSON data
    file_path = f"configs/{dataset}_gnn_preds.json"

    # Writing the dictionary to a JSON file with indentation
    with open(file_path, "w") as json_file:
        json.dump(lp_data, json_file, indent=4)
    print("Saving Trained Model ....!!!!")

    save_PATH = f"trained_models/{dataset}_trained.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        save_PATH,
    )
    print(f"Model RGCN trained with {dataset} Dataset and Stored at {save_PATH}.")


train_gnn()
