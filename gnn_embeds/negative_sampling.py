import torch
from torch_geometric.data import Data

def random_replacement_(batch, index, selection, size, max_index):
    """
    Replace a column of a batch of indices with random indices.
    
    This function replaces either the head or tail of a triple with a random entity.
    
    :param batch: Tensor of shape (num_samples, 3), where each row is (head, rel, tail)
    :param index: The index to replace (0 = head, 2 = tail)
    :param selection: A slice specifying the range of samples to replace
    :param size: The number of samples to corrupt in this range
    :param max_index: The maximum index value for replacement
    """
    # Generate random replacements
    replacement = torch.randint(high=max_index - 1, size=(size,), device=batch.device)
    
    # Ensure we do not replace with the original value
    replacement += (replacement >= batch[selection, index]).long()
    
    # Apply the random replacement
    batch[selection, index] = replacement


def negative_sampling_pk(data, negative_rate=10):
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
    src, dst = data.edge_index[0], data.edge_index[1]
    rel = data.edge_type

    # Combine src, rel, dst into a single tensor of positive samples
    pos_samples = torch.stack((src, rel, dst), dim=1)  # shape: (num_edges, 3)

    # Get the number of unique entities and relations in the graph
    num_entity = data.num_nodes
    num_relation = rel.max().item() + 1  # Assuming relation indices are 0-based

    # Create negative samples
    # Repeat the positive samples for the negative sampling
    num_pos_samples = pos_samples.shape[0]
    neg_samples = pos_samples.repeat_interleave(negative_rate, dim=0)

    # Total number of negative samples
    total_neg_samples = neg_samples.shape[0]

    # Corrupt either head or tail randomly
    split_idx = total_neg_samples // 2  # Half of the samples will corrupt the head, the other half the tail

    # Corrupt head entities in the first half
    random_replacement_(neg_samples, index=0, selection=slice(0, split_idx), size=split_idx, max_index=num_entity)

    # Corrupt tail entities in the second half
    random_replacement_(neg_samples, index=2, selection=slice(split_idx, total_neg_samples), size=total_neg_samples - split_idx, max_index=num_entity)

    # Combine positive and negative samples
    all_samples = torch.cat([pos_samples, neg_samples], dim=0)

    # Create labels: 1 for positive samples, 0 for negative samples
    pos_labels = torch.ones(num_pos_samples, dtype=torch.float32)
    neg_labels = torch.zeros(total_neg_samples, dtype=torch.float32)
    all_labels = torch.cat([pos_labels, neg_labels], dim=0)

    return all_samples, all_labels


def negative_sampling_de(data: Data, num_neg_samples: int = 1):
    """
    Generate negative samples for graph data in PyTorch Geometric format.
    
    Args:
        data (Data): PyTorch Geometric Data object containing graph information.
        num_neg_samples (int): Number of negative samples to generate for each positive edge.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - edge_index: Tensor of shape (2, num_edges + num_neg_samples * num_edges) with positive and negative edges.
            - labels: Tensor of shape (num_edges + num_neg_samples * num_edges,) with labels for positive (1) and negative (0) edges.
    """
    # Extract positive edges
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    
    # Extract number of nodes
    num_nodes = data.num_nodes
    
    # Generate negative samples
    neg_edge_index = []
    neg_labels = []
    
    for _ in range(num_neg_samples):
        # Randomly sample node pairs to create negative edges
        sampled_edges = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
        
        # Ensure that no positive edges are included in negative edges
        sampled_edges_set = set(map(tuple, sampled_edges.t().tolist()))
        existing_edges_set = set(map(tuple, edge_index.t().tolist()))
        
        # Remove edges that are already positive
        neg_edges = [edge for edge in sampled_edges_set if edge not in existing_edges_set]
        
        if len(neg_edges) < num_edges:
            # If not enough negative samples were generated, repeat until we get enough
            while len(neg_edges) < num_edges:
                sampled_edges = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
                sampled_edges_set = set(map(tuple, sampled_edges.t().tolist()))
                neg_edges = [edge for edge in sampled_edges_set if edge not in existing_edges_set]
        
        neg_edge_index.append(torch.tensor(neg_edges).t().long())
        neg_labels.extend([0] * len(neg_edges))
    
    neg_edge_index = torch.cat(neg_edge_index, dim=1)
    pos_labels = [1] * num_edges
    labels = torch.tensor(pos_labels + neg_labels)
    
    # Combine positive and negative edges
    combined_edge_index = torch.cat([edge_index, neg_edge_index], dim=1)
    
    return combined_edge_index, labels

