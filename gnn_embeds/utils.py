import numpy as np
import torch


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
    samples, labels = negative_sampling(pos_samples, num_entity, negative_rate)

    # Convert samples and labels to PyTorch tensors
    return torch.tensor(samples, dtype=torch.long), torch.tensor(
        labels, dtype=torch.float32
    )
