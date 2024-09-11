from evaluate import calc_mrr
import numpy as np
import os
import math
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_scatter import scatter_add
from torch_geometric.data import Data
from utils import generate_train_data
from dataset import RGDataset
from models import RGCN

import json
from datetime import datetime


def save_results(results_dict):

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    file_path = os.path.join(results_dir, "results.json")

    with open(file_path, "w") as file:
        json.dump(results_dict, file, indent=4)

    print(f"Results saved to {file_path}")


def main(args):
    small_dataset = True
    if args.dataset_name == "FB15k-237":
        small_dataset = False
    dataset_root = os.path.join("KGs", args.dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RGDataset(root=dataset_root)
    train_data = dataset.train_data
    train_data.to(device)
    train_triples = dataset.triple_splits["train"]
    valid_triples = torch.LongTensor(dataset.triple_splits["valid"])
    test_triples = torch.LongTensor(dataset.triple_splits["test"])
    all_triples = dataset.all_triples

    train_triples = dataset.triple_splits["train"]
    grad_norm = 1.0
    model = RGCN(
        256,
        dataset.num_entites,
        dataset.num_relations,
        4,
        0.2,
        embeds_path=dataset.embeds_base_path,
        pre_trained_embeds=args.pre_trained,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    from tqdm import tqdm, trange

    epochs = args.num_epochs
    eval_count = 0
    pbar = tqdm(range(epochs), ncols=100)

    if small_dataset:
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
    else:
        triple_chunks = np.array_split(train_triples, 15)
        for epoch in pbar:
            epoch_loss = 0
            eval_count += 1
            for chunk in triple_chunks:
                train_data = generate_train_data(chunk)
                train_data.to(device)
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
            epoch_loss += loss
            pbar.set_postfix({"Loss": f"{epoch_loss:.3f}"})
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
    metrics = calc_mrr(
        entity_embedding,
        model.relation_embedding,
        test_triples,
        all_triples,
        hits=[1, 3, 10],
    )
    metrics["dataset"] = args.dataset_name
    metrics["epochs"] = args.num_epochs
    metrics["Pre_trained_Embedds"] = str(args.pre_trained)
    save_results(metrics)


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Training script")

    # Adding optional arguments with default values
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="UMLS",
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=30, help="Number of epochs for training"
    )
    parser.add_argument(
        "--pre_trained",
        type=bool,
        default=True,
        help="True to use pre-trained embeddings",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)
