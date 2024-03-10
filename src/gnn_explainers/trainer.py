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
    """
    A utility class for early stopping during the training of machine learning models.

    This class is designed to monitor a model's loss during training and trigger an early stop if the loss does not decrease significantly over a specified number of epochs.
    It helps prevent overfitting by stopping the training process when the model's performance on the validation set starts degrading.

    Attributes:
        patience_decrease (int): The number of epochs with decreased loss to wait before resetting the counter. Defaults to 5.
        patience_increase (int): The number of epochs with increased loss to wait before triggering early stopping. Defaults to 10.
        delta (float): The minimum change in the monitored loss to qualify as an improvement or degradation. Defaults to 0.001.
        prev_score (float or None): The loss from the previous epoch. Initially None and updated during training.
        counter_decrease (int): Counts the number of epochs where loss has decreased. Resets if loss increases.
        counter_increase (int): Counts the number of epochs where loss has increased. Triggers early stopping if it reaches the patience threshold.
        early_stop (bool): Flag indicating whether early stopping has been triggered.

    Methods:
        __call__(loss): Updates the counters and early stopping flag based on the new loss value.

    Example:
        >>> early_stopping = EarlyStopping(patience_decrease=5, patience_increase=10, delta=0.001)
        >>> for epoch in range(epochs):
        >>>     train()  # Your training process here
        >>>     val_loss = validate()  # Your validation process here
        >>>     early_stopping(val_loss)
        >>>     if early_stopping.early_stop:
        >>>         print("Early stopping triggered")
        >>>         break

    Note:
        - This class should be instantiated and used as part of a training loop.
        - It is used in conjunction with a validation loss metric.
    """

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
                self.counter_decrease += 0
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


def get_lp_mutag_fid(gnn_pred_dt_train, gnn_pred_dt_test, idx_map):
    # function tro create learning problems for the Mutag dataset for fidelity evaluations based on GNN model predictions.
    train_positive_examples = [
        idx_map[item]["IRI"]
        for item in gnn_pred_dt_train
        if gnn_pred_dt_train[item] == 1
    ]
    train_negative_examples = [
        idx_map[item]["IRI"]
        for item in gnn_pred_dt_train
        if gnn_pred_dt_train[item] == 0
    ]

    # Positive and negative examples for test set
    test_positive_examples = [
        idx_map[item]["IRI"] for item in gnn_pred_dt_test if gnn_pred_dt_test[item] == 1
    ]
    test_negative_examples = [
        idx_map[item]["IRI"] for item in gnn_pred_dt_test if gnn_pred_dt_test[item] == 0
    ]
    assert (
        len(set(train_positive_examples).intersection(set(train_negative_examples)))
        == 0
    )

    lp_dict_test_train = {
        "carcino": {
            "positive_examples_train": train_positive_examples,
            "negative_examples_train": train_negative_examples,
            "positive_examples_test": test_positive_examples,
            "negative_examples_test": test_negative_examples,
        }
    }
    return lp_dict_test_train


def get_lp_aifb_fid(gnn_pred_dt_train, gnn_pred_dt_test, idx_map):
    # function tro create learning problems for the Mutag dataset for fidelity evaluations based on GNN model predictions.
    train_positive_examples = [
        idx_map[item]["IRI"]
        for item in gnn_pred_dt_train
        if gnn_pred_dt_train[item] == 0
    ]
    train_negative_examples = [
        idx_map[item]["IRI"]
        for item in gnn_pred_dt_train
        if gnn_pred_dt_train[item] == 1
    ]

    # Positive and negative examples for test set
    test_positive_examples = [
        idx_map[item]["IRI"] for item in gnn_pred_dt_test if gnn_pred_dt_test[item] == 0
    ]
    test_negative_examples = [
        idx_map[item]["IRI"] for item in gnn_pred_dt_test if gnn_pred_dt_test[item] == 1
    ]
    assert (
        len(set(train_positive_examples).intersection(set(train_negative_examples)))
        == 0
    )

    lp_dict_test_train = {
        "id1instance": {
            "positive_examples_train": train_positive_examples,
            "negative_examples_train": train_negative_examples,
            "positive_examples_test": test_positive_examples,
            "negative_examples_test": test_negative_examples,
        }
    }
    return lp_dict_test_train


def train_gnn(dataset="mutag", device=None, PATH=None):
    """
    Trains a Graph Neural Network (GNN) model using the specified RDF dataset.

    This function sets up and trains a Relational Graph Convolutional Network (RGCN) model on the provided dataset. It handles the entire training process, including loss calculation, optimizer updates, and early stopping. The function also generates and saves accuracy and loss plots, as well as the trained model.

    Parameters:
        dataset (str, optional): The name of the dataset to train the model on. Defaults to 'mutag'.
        device (torch.device, optional): The device to train the model on (CPU or GPU). Defaults to None, which means the function will automatically choose the device.
        PATH (str, optional): Path to save the trained model. If not specified, the model is saved in the 'trained_models' directory.

    Returns:
        None. The trained model is saved to the specified path, and various plots and JSON data related to the training process are also saved.

    Raises:
        FileNotFoundError: If the configuration file for the specified dataset is not found.

    Example:
        >>> train_gnn(dataset="mutag", device=torch.device("cuda"), PATH="path/to/save/model.pt")
        # This will train an RGCN model on the MUTAG dataset, save the model, accuracy and loss plots, and JSON data.

    Notes:
        - The function includes validation accuracy and loss calculations.
        - Early stopping is implemented to prevent overfitting.
        - The function saves the trained model, accuracy and loss plots, and a JSON file containing logical explainer data.
        - The function supports training on both CPU and GPU, depending on the availability and specification.
    """
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
    else:
        valid_idx = my_dataset.test_idx.to(device)

    idx_map = my_dataset.idx_map
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

        val_loss = F.cross_entropy(logits[valid_idx], labels[valid_idx])
        val_acc = th.sum(
            logits[valid_idx].argmax(dim=1) == labels[valid_idx]
        ).item() / len(valid_idx)
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
    val_acc_final = th.sum(
        logits[test_idx].argmax(dim=1) == labels[test_idx]
    ).item() / len(test_idx)
    gnn_preds_test = pred_logit[test_idx].argmax(dim=1).tolist()
    if validation:
        train_idx = torch.cat([train_idx, valid_idx], dim=0)
    gnn_preds_train = pred_logit[train_idx].argmax(dim=1).tolist()
    print(
        f"Final validation accuracy of the model R-GCN on unseen dataset: {val_acc_final}"
    )
    train_idx_list = train_idx.tolist()
    test_idx_list = test_idx.tolist()
    gnn_pred_dt_train = {
        tensor: pred for tensor, pred in zip(train_idx_list, gnn_preds_train)
    }
    gnn_pred_dt_test = {
        tensor: pred for tensor, pred in zip(test_idx_list, gnn_preds_test)
    }
    if dataset == "mutag":
        lp_data_train_test = get_lp_mutag_fid(
            gnn_pred_dt_train, gnn_pred_dt_test, idx_map
        )
    if dataset == "aifb":
        lp_data_train_test = get_lp_aifb_fid(
            gnn_pred_dt_train, gnn_pred_dt_test, idx_map
        )

    plt.plot(train_accs, label="Training Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"results/visualizations/training_validation_plot_{dataset}.png")

    # # File path where you want to store the JSON data
    file_path = f"configs/{dataset}_gnn_preds.json"

    # # Writing the dictionary to a JSON file with indentation
    with open(file_path, "w") as json_file:
        json.dump(lp_data_train_test, json_file, indent=4)

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
    plt.clf()
    plt.plot(vald_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"results/visualizations/training_validation_loss_{dataset}.png")
