import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from torch import nn

torch.cuda.empty_cache()

from src.gnn_explainers.configs import get_configs
from src.gnn_explainers.dataset import RDFDatasets
from src.gnn_explainers.hetro_features import HeteroFeature
from src.gnn_explainers.model import RGCN
from src.gnn_explainers.utils import get_nodes_dict


class EarlyStopping:
    def __init__(self, patience_decrease=5, patience_increase=10, delta=0.001):
        self.patience_decrease = patience_decrease
        self.patience_increase = patience_increase
        self.delta = delta
        self.prev_score = None
        self.counter_decrease = 0
        self.counter_increase = 0
        self.early_stop = False

    def __call__(self, loss):
        if self.prev_score is not None:
            if loss < self.prev_score - self.delta:
                # Loss decreased
                self.counter_decrease = 0
            elif loss > self.prev_score + self.delta:
                # Loss increased
                self.counter_increase += 1
                if self.counter_increase >= self.patience_increase:
                    self.early_stop = True
            else:
                # Loss didn't decrease or increase significantly
                self.counter_decrease = 0
                self.counter_increase = 0

        self.prev_score = loss


def get_lp_aifb(gnn_pred_dt, idx_map):
    class_to_pred = {}
    multi_lp_dict = {}

    # Creating new_dict
    for key, value in gnn_pred_dt.items():
        class_to_pred.setdefault(value, []).append(key)

    # Creating news_dict
    multi_lp_dict = {
        f"id{key+1}instance": {
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
        idx_map[item]["IRI"] for item in gnn_pred_dt if gnn_pred_dt[item] == 0
    ]

    lp_dict = {
        "carcino": {
            "positive_examples": positive_examples,
            "negative_examples": negative_examples,
        }
    }
    return lp_dict


def train_gnn(dataset="mutag", device=None, PATH=None):
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

    my_dataset = RDFDatasets(dataset, root="data/", validation=validation)
    g = my_dataset.g.to(device)
    out_dim = my_dataset.num_classes
    e_types = g.etypes
    category = my_dataset.category
    train_idx = my_dataset.train_idx.to(device)
    test_idx = my_dataset.test_idx.to(device)
    labels = my_dataset.labels.to(device)

    if validation:
        valid_idx = my_dataset.valid_idx.to(device)
        test_idx = torch.cat([test_idx, valid_idx], dim=0)

    idx_map = my_dataset.idx_map
    pred_idx = torch.cat([train_idx, test_idx], dim=0)
    input_feature = HeteroFeature({}, get_nodes_dict(g), hidden_dim, act=act).to(device)
    model = RGCN(
        hidden_dim,
        hidden_dim,
        out_dim,
        e_types,
        num_bases,
        category,
    ).to(device)

    # Define the optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = F.cross_entropy
    model.add_module("input_feature", input_feature)
    optimizer.add_param_group({"params": input_feature.parameters()})
    early_stopping = EarlyStopping(
        patience_decrease=5, patience_increase=10, delta=0.001
    )
    print("Start training...")
    dur = []
    train_accs = []
    val_accs = []
    vald_loss = []
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
        train_accs.append(train_acc)

        val_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
        val_acc = th.sum(
            logits[test_idx].argmax(dim=1) == labels[test_idx]
        ).item() / len(test_idx)
        val_accs.append(val_acc)
        vald_loss.append(val_loss.item())
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
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
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

    plt.plot(train_accs, label="Training Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    # plt.savefig(f"results/visualizations/training_validation_plot_{dataset}.png")

    # # File path where you want to store the JSON data
    # file_path = f"configs/{dataset}_gnn_preds.json"

    # # Writing the dictionary to a JSON file with indentation
    # with open(file_path, "w") as json_file:
    #     json.dump(lp_data, json_file, indent=4)
    # print("Saving Trained Model ....!!!!")

    # save_PATH = f"trained_models/{dataset}_trained.pt"
    # torch.save(
    #     {
    #         "epoch": epoch,
    #         "model_state_dict": model.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "loss": loss,
    #     },
    #     save_PATH,
    # )
    # print(f"Model RGCN trained with {dataset} Dataset and Stored at {save_PATH}.")
    # plt.clf()
    # plt.plot(vald_loss, label="Validation Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.savefig(f"results/visualizations/training_validation_loss_{dataset}.png")
