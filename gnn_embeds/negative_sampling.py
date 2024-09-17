import torch

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
