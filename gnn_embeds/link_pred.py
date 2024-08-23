import itertools

import dgl
import dgl.function as fn
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from kgdataset import KnowledgeGraphDataset

from dgl.nn import SAGEConv


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata["score"][:, 0]


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        return {"score": self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]


# ----------- 2. create model -------------- #
# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats, "mean")

    def forward(self, g, in_feat):
        # in_feat = in_feat.float()
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    )
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    return roc_auc_score(labels, scores)


# load dataset and convert to homogenous
g_verbose = False
dataset = KnowledgeGraphDataset("KGs/UMLS/", "KGs/Keci_entity_embeddings.csv")
graph = dataset.get_graph()
if g_verbose:
    print(dataset.get_node_counts())
    print(dataset.get_edge_counts())

g = dgl.to_homogeneous(graph, ndata=graph.ndata)

# Split edge set for training and testing
u, v = g.edges()
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
neg_u, neg_v = np.where(adj_neg != 0)

train_pos_u, train_pos_v = u[dataset.train_idx], v[dataset.train_idx]
test_pos_u, test_pos_v = u[dataset.test_idx], v[dataset.test_idx]

num_train_neg = len(train_pos_u)
num_test_neg = len(test_pos_u)

# Shuffle the negative edges
neg_edge_indices = np.random.permutation(len(neg_u))

# Get training and testing negative edges
train_neg_u = neg_u[neg_edge_indices[:num_train_neg]]
train_neg_v = neg_v[neg_edge_indices[:num_train_neg]]

test_neg_u = neg_u[neg_edge_indices[num_train_neg : num_train_neg + num_test_neg]]
test_neg_v = neg_v[neg_edge_indices[num_train_neg : num_train_neg + num_test_neg]]

# Convert to sets of tuples for easy comparison
train_pos_edges = set(zip(train_pos_u, train_pos_v))
test_pos_edges = set(zip(test_pos_u, test_pos_v))
train_neg_edges = set(zip(train_neg_u, train_neg_v))
test_neg_edges = set(zip(test_neg_u, test_neg_v))

# Assertions to ensure no overlaps
assert train_pos_edges.isdisjoint(
    train_neg_edges
), "There are common edges between train positive and train negative edges."
assert test_pos_edges.isdisjoint(
    test_neg_edges
), "There are common edges between test positive and test negative edges."
assert train_pos_edges.isdisjoint(
    test_neg_edges
), "There are common edges between train positive and test negative edges."
assert test_pos_edges.isdisjoint(
    train_neg_edges
), "There are common edges between test positive and train negative edges."


# Create graphs for training and testing
train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

# Optionally, you might want to remove the test edges from the original graph (if required)
train_g = dgl.remove_edges(g, dataset.test_idx)


model = GraphSAGE(train_g.ndata["feat"].shape[1], 16)
# You can replace DotPredictor with MLPPredictor.
# pred = MLPPredictor(16)
pred = DotPredictor()


# ----------- set up loss and optimizer -------------- #
# in this case, loss will in training loop
optimizer = torch.optim.Adam(
    itertools.chain(model.parameters(), pred.parameters()), lr=0.01
)

# ----------- training -------------------------------- #
all_logits = []
for e in range(100):
    # forward
    h = model(train_g, train_g.ndata["feat"])
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 5 == 0:
        print("In epoch {}, loss: {}".format(e, loss))

# -----------  results ------------------------ #

with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    print("AUC", compute_auc(pos_score, neg_score))
