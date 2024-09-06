import torch
from torch.utils.data import DataLoader, Dataset
from dataset import RGDataset
from evaluate import calc_mrr
from models import RGCN
from torch_geometric.loader import ClusterData, ClusterLoader
from utils import generate_samples_and_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


dataset = RGDataset(root="KGs/FB15k-237")

train_data = dataset.train_data
# valid_triples = torch.LongTensor(dataset.triple_splits["valid"])
test_triples = torch.LongTensor(dataset.triple_splits["test"])
all_triples = dataset.all_triples


train_data.to(device)
grad_norm = 1.0
batch_size = 1024  # Define your batch size

cluster_data = ClusterData(dataset.train_data, num_parts=10)

model = RGCN(
    256,
    dataset.num_entites,
    dataset.num_relations,
    4,
    0.2,
    embeds_path=dataset.embeds_base_path,
    pre_trained_embeds=True,
)
if device.type == "cuda":
    model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

from tqdm import tqdm, trange

epochs = 500
eval_count = 0
pbar = tqdm(range(epochs), ncols=100)

for epoch in pbar:
    eval_count += 1
    model.train()
    total_loss = 0
    for batch_data in cluster_data:
        batch_data.samples, batch_data.labels = generate_samples_and_labels(batch_data)
        optimizer.zero_grad()
        entity_embedding = model(
            batch_data.entity,
            batch_data.edge_index,
            batch_data.edge_type,
            batch_data.edge_norm,
        )
        loss = model.score_loss(
            entity_embedding, batch_data.samples, batch_data.labels
        ) + 1e-2 * model.reg_loss(entity_embedding)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()
        total_loss += loss
    pbar.set_postfix({"Loss": f"{loss:.3f}"})

    # if (eval_count % 25) == 0:
    #     if device.type == "cuda":
    #         model.cpu()
    #     model.eval()
    #     mrr = calc_mrr(
    #         entity_embedding,
    #         model.relation_embedding,
    #         valid_triples,
    #         all_triples,
    #         hits=[1, 3, 10],
    #     )

    # if device.type == "cuda":
    #     model.cuda()

if device.type == "cuda":
    model.cpu()


model.eval()
mrr = calc_mrr(
    entity_embedding,
    model.relation_embedding,
    test_triples,
    all_triples,
    hits=[1, 3, 10],
)
