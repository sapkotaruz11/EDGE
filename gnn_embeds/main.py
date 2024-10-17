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
from SAGEModel import SAGE

import json
from datetime import datetime


def save_results(results_dict, epoch_dict):

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    file_path = os.path.join(results_dir, "results.json")
    file_path_loss = os.path.join(results_dir, "loss.json")

    with open(file_path, "w") as file:
        json.dump(results_dict, file, indent=4)

    with open(file_path_loss, "w") as file:
        json.dump(epoch_dict, file, indent=4)

    print(f"Results saved to {file_path}")


def main(args):
    small_dataset = True
    if args.dataset == "FB15k-237" or "NELL-995-h50":
        small_dataset = False
    dataset_root = os.path.join("KGs", args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RGDataset(root=dataset_root, embed_model= args.embed_model)
    print(f"Training {args.dataset} on device {device} ")

    train_triples = dataset.triple_splits["train"]
    valid_triples = torch.LongTensor(dataset.triple_splits["valid"])
    test_triples = torch.LongTensor(dataset.triple_splits["test"])
    all_triples = dataset.all_triples
    grad_norm = 1.0

    # Results dictionary with common information only stored once
    results = {
        "dataset": args.dataset,
        "epochs": args.num_epochs,
        "embeds_dim" : dataset.embeds_dim
        
    }
    losses = {}
    # Loop over both with and without pre-trained embeddings
    for pre_trained in [False,True ]:
        loss_key = "True" if pre_trained else "False"
        embed_type = "pre_trained_embeds" if pre_trained else "random_embeds"
        print(f"Running experiment with pre_trained={pre_trained}")
        if pre_trained :
            results['embedding_model'] = args.embed_model
        
        if args.nn_model == "RGCN":
            model = RGCN(
                dataset.embeds_dim,
                dataset.num_entites,
                dataset.num_relations,
                4,
                0.2,
                embeds_path=dataset.embeds_base_path,
                pre_trained_embeds=pre_trained,
                embed_model = args.embed_model
            )
        else:
            model = SAGE(
                dataset.embeds_dim,
                dataset.num_entites,
                dataset.num_relations,
                4,
                0.2,
                embeds_path=dataset.embeds_base_path,
                pre_trained_embeds=pre_trained,
                embed_model = args.embed_model
            )

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        epochs = args.num_epochs
        eval_count = 0
        pbar = tqdm(range(epochs), ncols=100)
        epoch_losss = {}
        if small_dataset:
            
            for epoch in pbar:
                
                training_data = generate_train_data(train_triples)
                # training_data = dataset.train_data
                training_data.to(device)
                eval_count += 1
                model.train()
                optimizer.zero_grad()
                entity_embedding = model(
                        training_data.entity,
                        training_data.edge_index,
                        training_data.edge_type,
                        training_data.edge_norm,
                    )
                loss = model.score_loss(
                    entity_embedding, training_data.samples, training_data.labels
                ) + 1e-2 * model.reg_loss(entity_embedding)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
                optimizer.step()
                pbar.set_postfix({"Loss": f"{loss:.3f}"})
                if (eval_count % 50) == 0:
                    model.eval()
                    mrr = calc_mrr(
                        entity_embedding,
                        model.relation_embedding,
                        valid_triples,
                        all_triples,
                        hits=[1, 3, 10],
                    )
                    print(f"MRR at epoch {epoch}: {mrr}")
                epoch_losss[epoch] = round(loss.item(), 3)
            losses[loss_key] = epoch_losss
        else:
            triple_chunks = np.array_split(train_triples, 10)
            for epoch in pbar:
                full_entity_embedding = torch.zeros(
                    (dataset.num_entites, dataset.embeds_dim), device=device
                )
                epoch_loss = 0
                eval_count += 1
                for chunk in triple_chunks:
                    training_data = generate_train_data(chunk)
                    training_data.to(device)
                    model.train()
                    optimizer.zero_grad()
                    entity_embedding = model(
                        training_data.entity,
                        training_data.edge_index,
                        training_data.edge_type,
                        training_data.edge_norm,
                    )
                    loss = model.score_loss(
                        entity_embedding, training_data.samples, training_data.labels
                    ) + 1e-2 * model.reg_loss(entity_embedding)
                    loss.backward()
                    
                    optimizer.step()
                    full_entity_embedding[training_data.entity] = entity_embedding
                    epoch_loss += loss
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
                epoch_loss = epoch_loss / 10
                
                pbar.set_postfix({"Loss": f"{epoch_loss:.3f}"})
                if (eval_count % 50) == 0:
                    model.eval()
                    mrr = calc_mrr(
                        full_entity_embedding,
                        model.relation_embedding,
                        valid_triples,
                        all_triples,
                        hits=[1, 3, 10],
                    )
                    print(f"MRR at epoch {epoch}: {mrr}")
                    model.to(device)
                epoch_losss[epoch] = round(loss.item(), 3)
            losses[loss_key] = epoch_losss
        if small_dataset:
            full_entity_embedding = entity_embedding
        model.eval()
        
        
        # Calculate final MRR on the test set
        metrics = calc_mrr(
            full_entity_embedding,
            model.relation_embedding,
            test_triples,
            all_triples,
            hits=[1, 3, 10],
        )

        # Store the metrics under the appropriate key
        results[embed_type] = metrics


    
    
    
    save_results(results, losses)


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Training script")

    # Adding optional arguments with default values
    parser.add_argument(
        "--dataset",
        type=str,
        default="UMLS",
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=30, help="Number of epochs for training"
    )
    parser.add_argument(
        "--embed_model",
        type=str,
        default="Keci",
        help="Name of the embedding to use",
    )
    parser.add_argument(
        "--nn_model",
        type=str,
        default="RGCN",
        help="Name of the ML model  to use",
    )
    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)
