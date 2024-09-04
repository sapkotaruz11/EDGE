from evaluate import calc_mrr
import numpy as np
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_scatter import scatter_add
from torch_geometric.data import Data
from dataset import RGDataset
from models import RGCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = RGDataset(root="KGs/FB15k-237")


train_data = dataset.train_data
train_data.to(device)
valid_triples = torch.LongTensor(dataset.triple_splits["valid"])
valid_triples.to(device)
test_triples = torch.LongTensor(dataset.triple_splits["test"])
test_triples.to(device)
all_triples = dataset.all_triples
all_triples.to(device)

grad_norm = 1.0
model = RGCN(
    256,
    dataset.num_entites,
    dataset.num_relations,
    4,
    0.2,
    embeds_path=dataset.embeds_base_path,
    pre_trained_embeds=True,
)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

from tqdm import tqdm, trange

epochs = 500
eval_count = 0
pbar = tqdm(range(epochs), ncols=100)
for epoch in pbar:
    eval_count += 1
    model.train()
    optimizer.zero_grad()
    entity_embedding = model(
        train_data.entity,
        train_data.edge_index,
        train_data.edge_type,
        train_data.edge_norm,
    )
    loss = model.score_loss(
        entity_embedding, train_data.samples, train_data.labels
    ) + 1e-2 * model.reg_loss(entity_embedding)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
    optimizer.step()
    pbar.set_postfix({"Loss": f"{loss:.3f}"})
    if (eval_count % 25) == 0:

        model.eval()
        mrr = calc_mrr(
            entity_embedding,
            model.relation_embedding,
            valid_triples,
            all_triples,
            hits=[1, 3, 10],
        )
model.eval()
mrr = calc_mrr(
    entity_embedding,
    model.relation_embedding,
    test_triples,
    all_triples,
    hits=[1, 3, 10],
)
