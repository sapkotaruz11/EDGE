import torch
import os
from torch_geometric.data import InMemoryDataset, Data
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class KGDataset:
    def __init__(
        self,
        root: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        url: Optional[str] = None,
    ):
        self.node_embeds_path = "KGs/Keci_entity_embeddings.csv"
        self.rel_embeds_path = "KGs/Keci_relation_embeddings.csv"
        self.url = url
        self.raw_paths = [
            os.path.join(root, "raw", file_name) for file_name in self.raw_file_names()
        ]

        self.processed_paths = [
            os.path.join(root, "processed", file_name)
            for file_name in self.processed_file_names()
        ]

        # Load node and relation embeddings
        self.entity_to_feature = self.load_features(self.node_embeds_path)
        self.rel_to_feature = self.load_features(self.rel_embeds_path)

        self.node_idx_mapping = {
            key: i for i, key in enumerate(self.entity_to_feature.keys())
        }

        self.edge_idx_mapping = {
            key: i for i, key in enumerate(self.rel_to_feature.keys())
        }

        self.idx_node_mapping = {v: k for k, v in self.node_idx_mapping.items()}
        self.idx_edge_mapping = {v: k for k, v in self.edge_idx_mapping.items()}

        self.edge_split_mapping = {}
        self.all_triples = []
        # super().__init__(root, transform, pre_transform, pre_filter)

        self.process()
        self.get_splits()

    def raw_file_names(self):
        return ["train.txt", "test.txt", "valid.txt"]

    def processed_file_names(self):
        return ["data.pt", "split.pt"]

    def load_features(self, path: str) -> Dict[str, np.ndarray]:
        df = pd.read_csv(path)
        features = df.iloc[:, 1:].to_numpy(dtype=np.float32)
        entities = df.iloc[:, 0].to_numpy()
        return {entities[i]: features[i] for i in range(len(entities))}

    def process(self):
        for path in self.raw_paths:
            with open(path) as f:
                lines = [x.split("\t") for x in f.read().splitlines() if x]
            self.all_triples += lines
        edges = np.empty((0, 2), int)
        num_edges = len(self.all_triples)
        edge_index = torch.empty((2, num_edges), dtype=torch.long)
        edge_type = torch.empty(num_edges, dtype=torch.long)
        nodes = set()
        for i, (src, rel, dst) in enumerate(self.all_triples):
            src_id, dest_id = self.node_idx_mapping[src], self.node_idx_mapping[dst]
            edge_index[0, i] = src_id
            edge_index[1, i] = dest_id
            nodes.add(src_id)
            nodes.add(dest_id)
            edge_type[i] = self.edge_idx_mapping[rel]
            new_edge = np.array([[src_id, dest_id]])
            edges = np.vstack([edges, new_edge])

        self.node_list = sorted(list(nodes))

        features_list = [
            self.entity_to_feature[self.idx_node_mapping[node]]
            for node in self.node_list
        ]
        x = torch.tensor(
            np.array(features_list),
            dtype=torch.float32,
        )
        edge_feature_list = [
            self.rel_to_feature[self.idx_edge_mapping[rel.tolist()]]
            for rel in edge_type
        ]
        xr = torch.tensor(
            np.array(edge_feature_list),
            dtype=torch.float32,
        )
        self.data = Data(x=x, edge_index=edge_index, edge_type=edge_type, edge_attr=xr)
        # torch.save((data), self.processed_paths[0])
        # torch.save(self.edge_split_mapping, self.processed_paths[1])

    def get_node_count(self):
        return len(self.node_list)

    def get_splits(self):
        for path in self.raw_paths:
            edge = np.empty((0, 2), int)
            edge_ids = []
            split_name = path.split("/")[-1].split(".")[0]
            with open(path) as f:
                lines = [x.split("\t") for x in f.read().splitlines() if x]
            for src, rel, dst in lines:
                new_edge = np.array(
                    [[self.node_idx_mapping[src], self.node_idx_mapping[dst]]]
                )
                edge = np.vstack([edge, new_edge])
                edge_ids.append(self.edge_idx_mapping[rel])
            if split_name != "train":
                neg_edges = self.create_negative_mapping(edge)
                self.edge_split_mapping[split_name] = {
                    "edges": edge,
                    "edge_id": edge_ids,
                    "neg_edges": neg_edges,
                }
            else:

                self.edge_split_mapping[split_name] = {
                    "edges": edge,
                    "edge_id": edge_ids,
                }

    def create_negative_mapping(self, data):
        # Convert the input data to a set of tuples for easy lookup
        original_pairs = set(map(tuple, data))

        # Extract unique IDs from the data
        unique_ids = np.unique(data)

        # Generate all possible pairs of unique IDs
        all_possible_pairs = set(
            (i, j) for i in unique_ids for j in unique_ids if i != j
        )

        # Remove the original pairs to get the negative mapping
        negative_pairs = all_possible_pairs - original_pairs

        # Convert the result to a numpy array for consistency with the input format
        return np.array(list(negative_pairs))


data = KGDataset(root="KGs/UMLS/")
