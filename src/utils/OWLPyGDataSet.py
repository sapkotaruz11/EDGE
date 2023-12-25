"""This script is not utilized in the EDGE framework. Its rahter a proof-of-concept for another approach tried while developing the framework."""

import logging
import os
import os.path as osp
from collections import Counter
from typing import Callable, List, Optional

import torch
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_tar)
from torch_geometric.utils import index_sort


class OWL(InMemoryDataset):
    """FUnction to convert any OWL dataset that is converted to n3/nt objects along with training and testing sets provided beforehand"""

    def __init__(
        self,
        root: str,
        name: str,
        hetero: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower()
        self.hetero = hetero
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def num_relations(self) -> int:
        return self._data.edge_type.max().item() + 1

    @property
    def num_classes(self) -> int:
        return self._data.train_y.max().item() + 1

    @property
    def raw_file_names(self) -> List[str]:
        return [
            f"{self.name}_stripped.nt",
            "completeDataset.tsv",
            "trainingSet.tsv",
            "testSet.tsv",
        ]

    @property
    def processed_file_names(self) -> str:
        return "hetero_data.pt" if self.hetero else "data.pt"

    def process(self):
        import pandas as pd
        import rdflib as rdf

        graph_file, task_file, train_file, test_file = self.raw_paths

        with hide_stdout():
            g = rdf.Graph()
            with open(graph_file, "rb") as f:
                g.parse(file=f, format="nt")

        freq = Counter(g.predicates())

        relations = sorted(set(g.predicates()), key=lambda p: -freq.get(p, 0))
        subjects = set(g.subjects())
        objects = set(g.objects())
        nodes = list(subjects.union(objects))

        N = len(nodes)
        R = 2 * len(relations)

        relations_dict = {rel: i for i, rel in enumerate(relations)}
        nodes_dict = {node: i for i, node in enumerate(nodes)}

        edges = []
        for s, p, o in g.triples((None, None, None)):
            src, dst, rel = nodes_dict[s], nodes_dict[o], relations_dict[p]
            edges.append([src, dst, 2 * rel])
            edges.append([dst, src, 2 * rel + 1])

        edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
        _, perm = index_sort(N * R * edges[0] + R * edges[1] + edges[2])
        edges = edges[:, perm]

        edge_index, edge_type = edges[:2], edges[2]

        label_header = "label"
        nodes_header = "individuals"

        labels_df = pd.read_csv(task_file, sep="\t")
        labels_set = set(labels_df[label_header].values.tolist())
        labels_dict = {lab: i for i, lab in enumerate(list(labels_set))}
        nodes_dict = {str(key): val for key, val in nodes_dict.items()}

        train_labels_df = pd.read_csv(train_file, sep="\t")
        train_indices, train_labels = [], []
        for nod, lab in zip(
            train_labels_df[nodes_header].values, train_labels_df[label_header].values
        ):
            train_indices.append(nodes_dict[nod])
            train_labels.append(labels_dict[lab])

        train_idx = torch.tensor(train_indices, dtype=torch.long)
        train_y = torch.tensor(train_labels, dtype=torch.long)

        test_labels_df = pd.read_csv(test_file, sep="\t")
        test_indices, test_labels = [], []
        for nod, lab in zip(
            test_labels_df[nodes_header].values, test_labels_df[label_header].values
        ):
            test_indices.append(nodes_dict[nod])
            test_labels.append(labels_dict[lab])

        test_idx = torch.tensor(test_indices, dtype=torch.long)
        test_y = torch.tensor(test_labels, dtype=torch.long)

        data = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            train_idx=train_idx,
            train_y=train_y,
            test_idx=test_idx,
            test_y=test_y,
            num_nodes=N,
        )

        if self.hetero:
            data = data.to_heterogeneous(node_type_names=["v"])

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.name.upper()}{self.__class__.__name__}()"


class hide_stdout:
    def __enter__(self):
        self.level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)

    def __exit__(self, *args):
        logging.getLogger().setLevel(self.level)


# data = OWL(root="data", name="family")
# print(data)
