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

        self.num_entites = len(self.entity2id)
        self.num_relations = len(self.relation2id)
        self.triple_splits = {}

        self.all_triples_str = []
        self.process()
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


