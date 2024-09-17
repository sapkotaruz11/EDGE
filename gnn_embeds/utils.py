import torch
import os
from torch_geometric.data import InMemoryDataset, Data
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import torch.nn.functional as F
from tqdm import tqdm
from torch_scatter import scatter_add
from collections import defaultdict
from negative_sampling import negative_sampling_pk

def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    """
    Edge normalization trick
    - one_hot: (num_edge, num_relation)
    - deg: (num_node, num_relation)
    - index: (num_edge)
    - deg[edge_index[0]]: (num_edge, num_relation)
    - edge_norm: (num_edge)
    """
    one_hot = F.one_hot(edge_type, num_classes=2 * num_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_index[0], dim=0, dim_size=num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]

    return edge_norm


def negative_sampling(pos_samples, num_entity, negative_rate):
    """
    Generate negative samples by corrupting subjects or objects in positive samples.

    Args:
        pos_samples (np.ndarray): Positive samples of shape (N, 3) where each row is (src, rel, dst).
        num_entity (int): Total number of unique entities.
        negative_rate (int): The ratio of negative to positive samples.

    Returns:
        neg_samples (np.ndarray): Combined positive and negative samples.
        labels (np.ndarray): Labels for the samples (1 for positive, 0 for negative).
    """
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate

    # Create negative samples by copying positive ones
    neg_samples = np.tile(pos_samples, (negative_rate, 1))

    # Labels: first part (positive samples) is 1, the rest (negative samples) is 0
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[:size_of_batch] = 1

    # Generate random entity indices to corrupt the samples
    random_entities = np.random.choice(num_entity, size=num_to_generate)
    # Randomly choose to corrupt either the subject or object in the triples
    corrupt_subj = np.random.uniform(size=num_to_generate) > 0.5
    corrupt_obj = ~corrupt_subj

    # Corrupt subject or object with random entities
    neg_samples[corrupt_subj, 0] = random_entities[corrupt_subj]
    neg_samples[corrupt_obj, 2] = random_entities[corrupt_obj]

    return np.concatenate((pos_samples, neg_samples)), labels


def generate_samples_and_labels(data, negative_rate=2):
    """
    Generate positive and negative samples and corresponding labels from a PyG data object.

    Args:
        data (torch_geometric.data.Data): The input graph data containing `edge_index` and `edge_types`.
        negative_rate (int): The ratio of negative to positive samples.

    Returns:
        samples (torch.Tensor): Tensor of combined positive and negative samples.
        labels (torch.Tensor): Tensor of labels for the samples (1 for positive, 0 for negative).
    """
    # Extract edges (src, dst) and edge types (relations)
    src, dst = (
        data.edge_index[0],
        data.edge_index[1],
    )  # Convert edge_index to numpy array and transpose
    rel = data.edge_type.numpy()  # Convert edge types to numpy array

    # Combine src, rel, dst into a single array of positive samples
    pos_samples = np.stack((src, rel, dst), axis=1)

    # Get the number of unique entities in the graph
    num_entity = data.num_nodes

    # Generate negative samples and labels
    samples, labels = negative_sampling_pk(pos_samples, num_entity, negative_rate)

    # Convert samples and labels to PyTorch tensors
    return torch.tensor(samples, dtype=torch.long), torch.tensor(
        labels, dtype=torch.float32
    )


def generate_train_data(triples, negative_rate=2):
    """
    Generate positive and negative samples and corresponding labels from a PyG data object.

    Args:
        data (torch_geometric.data.Data): The input graph data containing `edge_index` and `edge_types`.
        negative_rate (int): The ratio of negative to positive samples.

    Returns:
        samples (torch.Tensor): Tensor of combined positive and negative samples.
        labels (torch.Tensor): Tensor of labels for the samples (1 for positive, 0 for negative).
    """
    # Extract edges (src, dst) and edge types (relations)
    src, rel, dst = triples.transpose()
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    num_rels = len(np.unique(rel))
    relabeled_edges = np.stack((src, rel, dst)).transpose()
    

    split_size = int(len(rel) * 0.5)
    graph_split_ids = np.random.choice(
        np.arange(len(rel)), size=split_size, replace=False
    )

    src = torch.tensor(src[graph_split_ids], dtype=torch.long).contiguous()
    dst = torch.tensor(dst[graph_split_ids], dtype=torch.long).contiguous()
    rel = torch.tensor(rel[graph_split_ids], dtype=torch.long).contiguous()

    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel
    data = Data(edge_index=edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(
        edge_type, edge_index, len(data.entity), len(rel)
    )
    data.samples, data.labels = negative_sampling_pk(data)
    return data
