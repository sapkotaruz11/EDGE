import json
import os
import shutil
import time

import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from dgl.data.utils import load_info
from torch import nn

from src.dglnn_local.pgexplainer import HeteroPGExplainer
from src.dglnn_local.subgraphx import HeteroSubgraphX
from src.gnn_explainers.configs import get_configs
from src.gnn_explainers.dataset import RDFDatasets
from src.gnn_explainers.hetro_features import HeteroFeature
from src.gnn_explainers.model import RGCN
from src.gnn_explainers.trainer import train_gnn
from src.gnn_explainers.utils import get_nodes_dict
from src.utils.visualize_hetero_graphs import visualize_hd


def viz_pg(dataset="aifb", node_idx=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = "aifb"

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

        train_idx = torch.cat([train_idx, valid_idx], dim=0)
    idx_map = my_dataset.idx_map
    # pred_idx = torch.cat([train_idx, test_idx], dim=0)
    pred_idx = test_idx
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
    checkpoint = torch.load(PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    feat = model.input_feature()
    test_idx = my_dataset.test_idx.to(device)

    node_id = test_idx[node_idx].tolist()
    explainer_path = f"trained_explainers/{dataset}_PGExplainer.pkl"
    explainer = load_info(explainer_path)
    probs, edge_mask, bg, inverse_indices = explainer.explain_node(
        {category: [test_idx[6].tolist()]}, g, feat, training=True
    )

    file_name = f"{dataset}_subg"
    visualize_hd(bg, node_id=node_id, file_name=file_name)


def viz_subgx(dataset="aifb", node_idx=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = "aifb"

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

        train_idx = torch.cat([train_idx, valid_idx], dim=0)
    idx_map = my_dataset.idx_map
    # pred_idx = torch.cat([train_idx, test_idx], dim=0)
    pred_idx = test_idx
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
    checkpoint = torch.load(PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    feat = model.input_feature()
    test_idx = my_dataset.test_idx.to(device)
    node_id = test_idx[node_idx].tolist()
    explainer = HeteroSubgraphX(model, num_hops=1, num_rollouts=3, shapley_steps=5)
    explanation, logits = explainer.explain_node(g, feat, node_id, category)
    file_name = f"{dataset}_subg"
    visualize_hd(
        explanation,
        node_id=node_id,
        file_name="exp_pg",
        edge_label_flag=True,
        caption=idx_map[node_idx]["IRI"],
    )


viz_subgx()
# viz_pg()
