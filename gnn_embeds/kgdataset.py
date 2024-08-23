import pandas as pd
import dgl
import torch
import os
import numpy as np
from collections import Counter, defaultdict


class KnowledgeGraphDataset:
    def __init__(self, kg_path, entity_embedding_path):
        self.kg_path = kg_path
        self.entity_embedding_path = entity_embedding_path

        # Load the triples
        self.train_triples, self.test_triples, self.triples = self._load_triples()

        # Initialize containers
        self.node_types = set()
        self.edge_types = set()
        self.node_id_map = {}
        self.edges = defaultdict(lambda: ([], []))
        self.num_node_dict = None
        self.num_edge_dicts = None
        self.g = None

        # Process the triples
        self._process_triples()

        # Build the graph
        self.g = self._build_graph()

        # Load entity embeddings
        self._load_entity_embeddings()

        self.train_idx = []
        self.test_idx = []

        for triple in self.test_triples:
            self.test_idx.append(self.g.canonical_etypes.index(triple))

        for triple in self.train_triples:
            self.train_idx.append(self.g.canonical_etypes.index(triple))

    def _load_triples(self):
        """Loads triples from the train and test files, with validation for .txt, .csv, and .tsv files."""

        # Define paths for train and test files
        train_path = os.path.join(self.kg_path, "train.txt")
        test_path = os.path.join(self.kg_path, "test.txt")

        # Helper function to load and validate a file
        def load_file(file_path):
            file_extension = os.path.splitext(file_path)[1]

            # Handling for .txt files
            if file_extension == ".txt":
                # Read the file and assert it has 3 columns
                df = pd.read_csv(file_path, delimiter="\t", header=None)

            # Handling for .csv or .tsv files (in case of future extensions)
            elif file_extension in [".csv", ".tsv"]:
                delimiter = "," if file_extension == ".csv" else "\t"
                df = pd.read_csv(file_path, delimiter=delimiter)

            else:
                raise ValueError(
                    f"Unsupported file format for {file_path}. Only .txt, .csv, and .tsv files are allowed."
                )

            # Check that the file has exactly 3 columns
            assert (
                df.shape[1] == 3
            ), f"The file {file_path} must have exactly 3 columns (subject, relation, object)."

            return df

        # Load the train and test files
        train_df = load_file(train_path)
        test_df = load_file(test_path)

        # Convert DataFrames to lists of triples
        train_triples = list(
            zip(train_df.iloc[:, 0], train_df.iloc[:, 1], train_df.iloc[:, 2])
        )
        test_triples = list(
            zip(test_df.iloc[:, 0], test_df.iloc[:, 1], test_df.iloc[:, 2])
        )

        # Combine train and test triples
        combined_triples = train_triples + test_triples

        return train_triples, test_triples, combined_triples

    def _process_triples(self):
        """Processes the triples to build node and edge types, mappings, and edge lists."""
        for subj, rel, obj in self.triples:
            self.node_types.add(subj)
            self.node_types.add(obj)
            self.edge_types.add((subj, rel, obj))

        # Create mappings for node types
        next_id = 0
        for node in self.node_types:
            self.node_id_map[node] = next_id
            next_id += 1

        # Build edge lists
        for subj, rel, obj in self.triples:
            src_id = self.node_id_map[subj]
            dst_id = self.node_id_map[obj]
            edge_type = (subj, rel, obj)
            self.edges[edge_type][0].append(src_id)
            self.edges[edge_type][1].append(dst_id)

        # Convert edge lists to tensors
        for etype in self.edges:
            self.edges[etype] = (
                torch.tensor(self.edges[etype][0]),
                torch.tensor(self.edges[etype][1]),
            )

        # Count occurrences of nodes and edges
        counter = Counter(pd.DataFrame(self.triples)[0]) + Counter(
            pd.DataFrame(self.triples)[2]
        )
        self.num_node_dict = dict(counter)
        self.num_edge_dicts = dict(Counter(pd.DataFrame(self.triples)[1]))

    def _build_graph(self):
        """Builds the DGL heterograph."""
        return dgl.heterograph(self.edges, idtype=torch.int32)

    def _load_entity_embeddings(self):
        """Loads entity embeddings and assigns them to the graph nodes."""
        node_feat = pd.read_csv(self.entity_embedding_path)
        feat = node_feat.iloc[:, 1:].to_numpy(dtype=np.float32)
        entity = node_feat.iloc[:, 0].to_numpy()
        feat_tensor = torch.tensor(feat)
        entity_to_feature = {entity[i]: feat_tensor[i] for i in range(len(entity))}

        for node_type in entity_to_feature.keys():
            num_nodes = self.g.num_nodes(node_type)
            self.g.nodes[node_type].data["feat"] = torch.tile(
                entity_to_feature[node_type], (num_nodes, 1)
            )

    def get_graph(self):
        """Returns the built DGL graph."""
        return self.g

    def get_node_counts(self):
        """Returns the count of nodes."""
        return self.num_node_dict

    def get_edge_counts(self):
        """Returns the count of edges."""
        return self.num_edge_dicts


# Example usage:
# dataset = KnowledgeGraphDataset('KGs/UMLS/train.txt', 'KGs/Keci_entity_embeddings.csv')
# graph = dataset.get_graph()
# print(dataset.get_node_counts())
# print(dataset.get_edge_counts())
