import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv import SAGEConv
import pandas as pd
import math
import numpy as np
import os


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


class SAGE(torch.nn.Module):
    def __init__(
        self,
        hidden_dims,
        num_entities,
        num_relations,
        num_bases,
        dropout,
        embeds_path=None,
        heads = 3,
        pre_trained_embeds=True,
        embed_model= "Keci",
    ):
        super(SAGE, self).__init__()
        if pre_trained_embeds:
            entity_embeds_file_name = embed_model + "_entity_embeddings.csv"
            relational_embeds_file_name = embed_model + "_relation_embeddings.csv"
            ent_path = os.path.join(embeds_path, entity_embeds_file_name)
            rel_path = os.path.join(embeds_path, relational_embeds_file_name)

            ent_df = pd.read_csv(ent_path)
            rel_df = pd.read_csv(rel_path)

            entity_embedding_weight = torch.tensor(
                ent_df.iloc[:, 1:].to_numpy(dtype=np.float32)
            )
            self.entity_embedding = nn.Embedding.from_pretrained(
                entity_embedding_weight, freeze=False
            )

            self.relation_embedding = nn.Parameter(
                torch.tensor(rel_df.iloc[:, 1:].to_numpy(dtype=np.float32))
            )
        else:
            self.entity_embedding = nn.Embedding(num_entities, hidden_dims)
            self.relation_embedding = nn.Parameter(
                torch.Tensor(num_relations, hidden_dims)
            )

            nn.init.xavier_uniform_(
                self.relation_embedding, gain=nn.init.calculate_gain("relu")
            )

        # self.gatconv1 = RGATConv(
        #     hidden_dims, hidden_dims, num_relations, num_bases=num_bases, edge_dim=hidden_dims,heads=heads, attention_mechanism="within-relation"
        # )
        # self.gatconv2 = RGATConv(
        #     hidden_dims * heads, hidden_dims, num_relations, num_bases=num_bases, heads=heads, concat=False, attention_mechanism="within-relation"
        # )
        self.gatconv1 = SAGEConv(
            hidden_dims, hidden_dims, aggr="mean")
        self.gatconv2 = SAGEConv(
            hidden_dims, hidden_dims,aggr="mean"
        )

        self.dropout_ratio = dropout

    def forward(self, entity, edge_index, edge_type, edge_norm):
        x = self.entity_embedding(entity)
        x_r = self.relation_embedding[edge_type]
        x = F.relu(self.gatconv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.gatconv2(x, edge_index)

        return x

    def distmult(self, embedding, triplets):
        s = embedding[triplets[:, 0]]
        r = self.relation_embedding[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)

        return score

    def score_loss(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)

        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))

