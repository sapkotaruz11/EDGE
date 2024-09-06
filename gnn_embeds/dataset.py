import torch
import os
from torch_geometric.data import InMemoryDataset, Data, Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import torch.nn.functional as F
from tqdm import tqdm
from torch_scatter import scatter_add
from collections import defaultdict


class RGDataset(Dataset):
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
        # self.raw_paths = [
        #     os.path.join(root, "raw", file_name) for file_name in self.raw_file_names()
        # ]

        # self.processed_paths = [
        #     os.path.join(root, "processed", file_name)
        #     for file_name in self.processed_file_names()
        # ]

        self.entity2id = self.load_features(self.node_embeds_path)
        self.relation2id = self.load_features(self.rel_embeds_path)

        # self.entity2id = {key: i for i, key in enumerate(self.entity_to_feature.keys())}

        # self.relation2id = {key: i for i, key in enumerate(self.rel_to_feature.keys())}

        self.num_entites = len(self.entity2id)
        self.num_relations = len(self.relation2id)
        self.triple_splits = {}

        self.all_triples_str = []

        super().__init__(root, transform, pre_transform, pre_filter)
        self.process()

        self.all_triples = torch.LongTensor(
            np.concatenate(list(self.triple_splits.values()))
        )
        self.train_data = torch.load(os.path.join(self.processed_dir, "train_data.pt"))
        self.test_data = torch.load(os.path.join(self.processed_dir, "test_data.pt"))

    @property
    def processed_file_names(self):
        return ["pre_transform.pt", "pre_filter.pt"]

    @property
    def raw_file_names(self):
        return ["train.txt", "test.txt", "valid.txt"]

    def load_features(self, path: str) -> Dict[str, np.ndarray]:
        df = pd.read_csv(path)
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
            triples = np.array(triples)
            data = self.get_data(triples)
            torch.save(
                data,
                os.path.join(self.processed_dir, f"{split_name}_data.pt"),
            )

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

    def get_data(self, triples):
        src, rel, dst = triples.transpose()
        src = torch.from_numpy(src)
        rel = torch.from_numpy(rel)
        dst = torch.from_numpy(dst)
        edge_index = torch.stack((src, dst))
        edge_type = rel

        data = Data(edge_index=edge_index)

        data.entity = torch.from_numpy(np.arange(self.num_entites))
        data.num_nodes = len(data.entity)
        data.edge_type = edge_type
        data.samples = None
        data.labels = None
        data.edge_norm = self.edge_normalization(
            edge_type, edge_index, self.num_entites, self.num_relations
        )
        return data


dataset = RGDataset(root="KGs/UMLS/")
