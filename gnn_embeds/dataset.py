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


class RGDataset:
    def __init__(
        self,
        root: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        url: Optional[str] = None,
    ):
        # Define the common base path for embedding files
        self.embeds_base_path = os.path.join(root, "embeds")
        self.node_embeds_path = os.path.join(
            self.embeds_base_path, "Keci_entity_embeddings.csv"
        )
        self.rel_embeds_path = os.path.join(
            self.embeds_base_path, "Keci_relation_embeddings.csv"
        )
        self.url = url
        self.raw_paths = [
            os.path.join(root, "raw", file_name) for file_name in self.raw_file_names()
        ]

        self.processed_paths = [
            os.path.join(root, "processed", file_name)
            for file_name in self.processed_file_names()
        ]

        self.entity2id = self.load_features(self.node_embeds_path)
        self.relation2id = self.load_features(self.rel_embeds_path)

        # self.entity2id = {key: i for i, key in enumerate(self.entity_to_feature.keys())}

        # self.relation2id = {key: i for i, key in enumerate(self.rel_to_feature.keys())}

        self.num_entites = len(self.entity2id)
        self.num_relations = len(self.relation2id)
        self.triple_splits = {}

        self.all_triples_str = []
        self.process()
        self.train_data = self.get_train_data(self.triple_splits["train"])
        self.test_data = self.get_data(self.triple_splits["test"])
        self.all_triples = torch.LongTensor(
            np.concatenate(list(self.triple_splits.values()))
        )

    def load_features(self, path: str) -> Dict[str, np.ndarray]:
        df = pd.read_csv(path)
        entitiedf = pd.read_csv(path)
        entities = df.iloc[:, 0].to_numpy()
        entity_to_id = defaultdict(lambda: len(entity_to_id))
        # Populate the dictionary
        for entity in entities:
            _ = entity_to_id[entity]
        return dict(entity_to_id)

    def process(self):

        for path in self.raw_paths:
            triples = []
            split_name = path.split("/")[-1].split(".")[0]
            with open(path) as f:
                lines = [x.split("\t") for x in f.read().splitlines() if x]
            self.all_triples_str += lines
            for src, rel, dst in lines:
                triples.append(
                    [self.entity2id[src], self.relation2id[rel], self.entity2id[dst]]
                )
            self.triple_splits[split_name] = np.array(triples)

    def edge_normalization(self, edge_type, edge_index, num_entity, num_relation):
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

    def negative_sampling(self, pos_samples, num_entity, negative_rate):
        size_of_batch = len(pos_samples)
        num_to_generate = size_of_batch * negative_rate
        neg_samples = np.tile(pos_samples, (negative_rate, 1))
        labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
        labels[:size_of_batch] = 1
        values = np.random.choice(num_entity, size=num_to_generate)
        choices = np.random.uniform(size=num_to_generate)
        subj = choices > 0.5
        obj = choices <= 0.5
        neg_samples[subj, 0] = values[subj]
        neg_samples[obj, 2] = values[obj]

        return np.concatenate((pos_samples, neg_samples)), labels

    def raw_file_names(self):
        return ["train.txt", "test.txt", "valid.txt"]

    def processed_file_names(self):
        return ["data.pt", "split.pt"]

    def get_train_data(self, triples):
        src, rel, dst = triples.transpose()
        uniq_entity, edges = np.unique((src, dst), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))
        relabeled_edges = np.stack((src, rel, dst)).transpose()

        num_rels = len(np.unique(rel))
        relabeled_edges = np.stack((src, rel, dst)).transpose()
        samples, labels = self.negative_sampling(relabeled_edges, len(uniq_entity), 5)

        split_size = int(len(rel) * 0.5)
        graph_split_ids = np.random.choice(
            np.arange(len(rel)), size=split_size, replace=False
        )

        src = torch.tensor(src[graph_split_ids], dtype=torch.long).contiguous()
        dst = torch.tensor(dst[graph_split_ids], dtype=torch.long).contiguous()
        rel = torch.tensor(rel[graph_split_ids], dtype=torch.long).contiguous()

        edge_index = torch.stack((src, dst))
        edge_type = rel

        data = Data(edge_index=edge_index)
        data.entity = torch.from_numpy(uniq_entity)
        data.edge_type = edge_type
        data.edge_norm = self.edge_normalization(
            edge_type, edge_index, self.num_entites, self.num_relations
        )
        data.samples = torch.from_numpy(samples)
        data.labels = torch.from_numpy(labels)
        return data

    def get_data(self, triples):
        src, rel, dst = triples.transpose()
        src = torch.from_numpy(src)
        rel = torch.from_numpy(rel)
        dst = torch.from_numpy(dst)
        edge_index = torch.stack((src, dst))
        edge_type = rel

        data = Data(edge_index=edge_index)
        data.entity = torch.from_numpy(np.arange(self.num_entites))
        data.edge_type = edge_type
        data.edge_norm = self.edge_normalization(
            edge_type, edge_index, self.num_entites, self.num_relations
        )
        return data


dataset = RGDataset(root="KGs/UMLS/")
