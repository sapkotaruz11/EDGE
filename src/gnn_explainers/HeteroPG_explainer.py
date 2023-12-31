import json
import os
import shutil
import time

import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from dgl.data.utils import load_info, makedirs, save_info
from torch import nn

from src.dglnn_local.pgexplainer import HeteroPGExplainer
from src.gnn_explainers.configs import get_configs
from src.gnn_explainers.dataset import RDFDatasets
from src.gnn_explainers.hetro_features import HeteroFeature
from src.gnn_explainers.model import RGCN
from src.gnn_explainers.trainer import train_gnn
from src.gnn_explainers.utils import get_nodes_dict


def explain_PG(dataset="mutag", explainer_train_epoch=30, print_explainer_loss=False):
    """
    Trains and explains a graph neural network (GNN) model using the PGExplainer on the specified RDF dataset.

    This function loads or trains a GNN model and then applies the PGExplainer to understand the model's predictions.
    It handles the entire process of model training, explainer training, and generating explanations. The results are saved in a JSON file.

    Parameters:
        dataset (str, optional): The name of the RDF dataset to be used. Defaults to 'mutag'.
        explainer_train_epoch (int, optional): The number of training epochs for the PGExplainer. Defaults to 30.
        print_explainer_loss (bool, optional): Flag to print the loss of the explainer during training. Defaults to False.

    Returns:
        None. The function saves the explanation results in a JSON file within the 'results/predictions/PGExplainer' directory.

    Raises:
        FileNotFoundError: If the trained GNN model or the PGExplainer checkpoint file is not found.

    Example:
        >>> explain_PG(dataset="mutag", explainer_train_epoch=30, print_explainer_loss=True)
        # This will train or load the GNN model for the MUTAG dataset, train the PGExplainer,
        # and save the explanations in a JSON file.

    Notes:
        - The function checks for pre-trained models and explainers and loads them if available.
        - It supports GPU acceleration if CUDA is available.
        - The results include predicted labels, ground truth, and entity information from the dataset.
    """
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = str.lower(dataset)

    # load training configs for the dataset
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

    # build model with dataset specific configs
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

    PATH = f"trained_models/{dataset}_trained.pt"
    if not os.path.isfile(PATH):
        print("Trained GNN Model not  found. Training GNN Model")
        train_gnn(dataset=dataset, PATH=PATH)

    print("Trained GNN Model found. Loading from Checkpoints")
    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    feat = model.input_feature()

    # load training and testing data into the device
    test_idx = my_dataset.test_idx.to(device)
    labels = my_dataset.labels.to(device)
    gt = labels[test_idx].tolist()
    pred_logit = model(g, feat)[category]
    gnn_preds = pred_logit[test_idx].argmax(dim=1).tolist()

    # check if the re-trained explainer exists
    explainer_path = f"trained_explainers/{dataset}_PGExplainer.pkl"
    if os.path.isfile(explainer_path):
        t0 = time.time()
        print("Starting PGExplainer")
        print(f"Trained PG Explainer on {dataset} Exists. Loading Trained Checkpoints")
        explainer = load_info(explainer_path)

    else:
        t0 = time.time()
        print("Starting PGExplainer")
        print(
            f"Trained PG Explainer on {dataset} Not found. Training Hetero PG Explainer"
        )
        explainer = HeteroPGExplainer(
            model, hidden_dim, num_hops=1, explain_graph=False
        )
        feat_1 = {item: feat[item].data for item in feat}
        # Train the explainer
        # Define explainer temperature parameter
        # define and train the model
        # Train the explainer
        # Define explainer temperature parameter
        init_tmp, final_tmp = 5.0, 1.0
        optimizer_exp = th.optim.Adam(explainer.parameters(), lr=0.01)
        for epoch in range(explainer_train_epoch):
            tmp = float(init_tmp * np.power(final_tmp / init_tmp, epoch / 20))
            loss = explainer.train_step_node(
                {ntype: g.nodes(ntype) for ntype in g.ntypes if ntype == category},
                g,
                feat_1,
                tmp,
            )
            optimizer_exp.zero_grad()
            loss.backward()
            optimizer_exp.step()
            if print_explainer_loss:
                print(f"Explainer trained for {epoch} epochs with loss {loss :3f}")
        save_info(explainer_path, explainer)
        print(f"Trained PG Explainer on Hetero {dataset} saved")

    entity = {}
    probs, edge_mask, bg, inverse_indices = explainer.explain_node(
        {category: test_idx}, g, feat, training=True
    )
    exp_pred = probs[category][inverse_indices[category]].argmax(dim=1).tolist()
    exp_preds = dict(zip(test_idx.tolist(), exp_pred))
    gnn_pred = dict(zip(exp_preds.keys(), gnn_preds))
    gts = dict(zip(exp_preds.keys(), gt))
    for idx in test_idx.tolist():
        entity[idx] = idx_map[idx]["IRI"]

    dict_names = ["exp_preds", "gnn_pred", "gts", "entity"]
    list_of_dicts = [exp_preds, gnn_pred, gts, entity]

    nested_dict = {
        key: {name: d[key] for name, d in zip(dict_names, list_of_dicts)}
        for key in exp_preds
    }
    file_path = f"results/predictions/PGExplainer/{dataset}.json"

    # Write the data to the JSON file with formatting (indentation)
    with open(file_path, "w") as json_file:
        json.dump(nested_dict, json_file, indent=2)

    print(f"Data has been written to {file_path}")
    print("Ending PG_Explainer")
    t1 = time.time()
    dur = t1 - t0
    print(f"Total time taken for Hetero-PG Explainer on {dataset} : {dur:.2f}")
