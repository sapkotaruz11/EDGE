from src.gnn_explainers import trainer
import json
import os
import shutil
import time

import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from dgl.data.utils import load_info, makedirs, save_info
from torch import nn

from src.dglnn_local.pgexplainer import HeteroPGExplainer
from src.gnn_explainers.configs import get_configs
from src.gnn_explainers.dataset import RDFDatasets
from src.gnn_explainers.hetro_features import HeteroFeature
from src.gnn_explainers.model import RGCN
from src.gnn_explainers.trainer import train_gnn
from src.gnn_explainers.utils import get_nodes_dict
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from src.dglnn_local.subgraphx import HeteroSubgraphX
from src.logical_explainers.EvoLearner import train_evo
from src.logical_explainers.CELOE import train_celoe
import json


def gen_evaluations(prediction_perfromance, explanation_perfromance):
    perfromances_evauations = {
        "pred_accuracy": prediction_perfromance["accuracy"],
        "pred_precision": prediction_perfromance["precision"],
        "pred_recall": prediction_perfromance["recall"],
        "pred_f1_score": prediction_perfromance["f1-score"],
        "exp_accuracy": explanation_perfromance["accuracy"],
        "exp_precision": explanation_perfromance["precision"],
        "exp_recall": explanation_perfromance["precision"],
        "exp_f1_score": explanation_perfromance["precision"],
    }

    return perfromances_evauations


def calcuate_metrics(ground_truths, gnn_preds, explainer_preds):
    pred_accuracy = accuracy_score(ground_truths, explainer_preds)

    # Calculate precision, recall, f1-score, and support for binary class
    pred_precision, pred_recall, pred_f1_score, _ = precision_recall_fscore_support(
        ground_truths, explainer_preds, average="binary"
    )
    exp_accuracy = accuracy_score(gnn_preds, explainer_preds)

    # Calculate precision, recall, f1-score, and support for binary class
    exp_precision, exp_recall, exp_f1_score, _ = precision_recall_fscore_support(
        gnn_preds, explainer_preds, average="binary"
    )
    perfromances_evauations = {
        "pred_accuracy": pred_accuracy,
        "pred_precision": pred_precision,
        "pred_recall": pred_recall,
        "pred_f1_score": pred_f1_score,
        "exp_accuracy": exp_accuracy,
        "exp_precision": exp_precision,
        "exp_recall": exp_recall,
        "exp_f1_score": exp_f1_score,
    }

    return perfromances_evauations


def process_logical_approaches(dataset, learning_problem_fid):
    file_path = f"configs/{dataset}.json"
    with open(file_path, "r") as file:
        # Load the JSON data from the file
        learning_problem_pred = json.load(file)
    target_dict_evo_fid, metrics_evo_fid, duration_evo_fid = train_evo(
        learning_problem_fid, kg=dataset
    )
    target_dict_celoe_fid, metrics_celoe_fid, duration_celoe_fid = train_celoe(
        learning_problem_fid, kg=dataset
    )

    target_dict_evo_pred, metrics_evo_pred, duration_evo_pred = train_evo(
        learning_problem_pred, kg=dataset
    )
    target_dict_celoe_pred, metrics_celoe_pred, duration_celoe_pred = train_celoe(
        learning_problem_pred, kg=dataset
    )
    target_dict_evo = {
        "predicition": target_dict_evo_pred,
        "explanation": target_dict_evo_fid,
        "duration_pred": duration_evo_pred,
        "duration_exp": duration_evo_fid,
    }
    target_dict_celoe = {
        "predicition": target_dict_celoe_pred,
        "explanation": target_dict_celoe_fid,
        "duration_pred": duration_celoe_pred,
        "duration_exp": duration_celoe_fid,
    }
    evo_metrics = gen_evaluations(metrics_evo_pred, metrics_evo_fid)
    celoe_metrics = gen_evaluations(metrics_celoe_pred, metrics_celoe_fid)

    return target_dict_evo, target_dict_celoe, evo_metrics, celoe_metrics


def run_explainers(dataset, print_explainer_loss=True, no_of_runs=5):
    pg_performance = {}
    sgx_performance = {}
    evo_performance = {}
    celoe_performance = {}
    pg_preds = {}
    sgx_preds = {}
    evo_preds = {}
    celoe_preds = {}

    file_path = f"configs/{dataset}.json"
    with open(file_path, "r") as file:
        # Load the JSON data from the file
        learning_problem_pred = json.load(file)
    for i in range(no_of_runs):
        model, my_dataset, learning_problem_fid, configs = train_gnn(dataset=dataset)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        g = my_dataset.g.to(device)
        out_dim = my_dataset.num_classes
        e_types = g.etypes
        category = my_dataset.category
        train_idx = my_dataset.train_idx.to(device)
        test_idx = my_dataset.test_idx.to(device)
        labels = my_dataset.labels.to(device)
        hidden_dim = configs["hidden_dim"]
        idx_map = my_dataset.idx_map
        gt = labels[test_idx].tolist()

        feat = model.input_feature()
        feat_1 = {item: feat[item].data for item in feat}
        pred_logit = model(g, feat)[category]
        gnn_preds = pred_logit[test_idx].argmax(dim=1).tolist()
        # Train the explainer
        # Define explainer temperature parameter
        # define and train the model
        # Train the explainer
        # Define explainer temperature parameter
        print("Starting PGExplainer")
        t0 = time.time()
        explainer_pg = HeteroPGExplainer(
            model, hidden_dim, num_hops=1, explain_graph=False
        )
        init_tmp, final_tmp = 5.0, 1.0
        optimizer_exp = th.optim.Adam(explainer_pg.parameters(), lr=0.01)
        for epoch in range(20):
            tmp = float(init_tmp * np.power(final_tmp / init_tmp, epoch / 20))
            loss = explainer_pg.train_step_node(
                {ntype: g.nodes(ntype) for ntype in g.ntypes if ntype == category},
                g,
                feat_1,
                tmp,
            )
            optimizer_exp.zero_grad()
            loss.backward()
            optimizer_exp.step()
            if print_explainer_loss:
                print(f"Explainer trained for {epoch} epochs with loss {loss :3f}")
        entity_pg = {}
        probs, edge_mask, bg, inverse_indices = explainer_pg.explain_node(
            {category: test_idx}, g, feat, training=True
        )
        exp_pred_pg = probs[category][inverse_indices[category]].argmax(dim=1).tolist()
        t1 = time.time()
        exp_preds_pg = dict(zip(test_idx.tolist(), exp_pred_pg))
        gnn_pred = dict(zip(exp_preds_pg.keys(), gnn_preds))
        gts = dict(zip(exp_preds_pg.keys(), gt))
        for idx in test_idx.tolist():
            entity_pg[idx] = idx_map[idx]["IRI"]

        dict_names = ["exp_preds", "gnn_pred", "gts", "entity"]
        list_of_dicts = [exp_preds_pg, gnn_pred, gts, entity_pg]

        nested_dict_pg = {
            key: {name: d[key] for name, d in zip(dict_names, list_of_dicts)}
            for key in exp_preds_pg
        }
        dur_pg = t1 - t0
        nested_dict_pg["duration"] = dur_pg
        print(f"Total time taken for PGExplainer  on {dataset}: {dur_sg:.2f}")
        pg_performances = calcuate_metrics(gt, gnn_preds, exp_preds_pg)
        pg_performance[i] = pg_performances
        pg_preds[i] = nested_dict_pg

        print("-" * 25)
        print("Starting SubGraphX")
        t2 = time.time()
        explainer_sgx = HeteroSubgraphX(
            model, num_hops=1, num_rollouts=3, shapley_steps=10
        )
        exp_preds_sgx = {}
        entity_sgx = {}
        for idx in test_idx.tolist():
            print(idx)
            explanation, logits = explainer_sgx.explain_node(g, feat, idx, category)
            exp_preds_sgx[idx] = logits
            entity_sgx[idx] = idx_map[idx]["IRI"]
        t3 = time.time()
        dur_sg = t3 - t2
        print(f"Total time taken for SubGraphX  on {dataset}: {dur_sg:.2f}")
        dict_names = ["exp_preds", "gnn_pred", "gts", "entity"]
        list_of_dicts = [exp_preds_sgx, gnn_pred, gts, entity_sgx]
        nested_dict_sgx = {
            key: {name: d[key] for name, d in zip(dict_names, list_of_dicts)}
            for key in exp_preds_sgx
        }
        nested_dict_sgx["duration"] = dur_sg
        exp_pred_sgx = list(exp_preds_sgx.values())
        sgx_performances = calcuate_metrics(gt, gnn_pred, exp_pred_sgx)
        sgx_performance[i] = sgx_performances
        sgx_preds[i] = nested_dict_sgx
        print("-" * 25)

        target_dict_evo, target_dict_celoe, evo_metrics, celoe_metrics = (
            process_logical_approaches(
                dataset=dataset, learning_problem_fid=learning_problem_fid
            )
        )
        evo_preds[i] = target_dict_evo
        celoe_preds[i] = target_dict_celoe

        evo_performance[i] = evo_metrics
        celoe_performance[i] = celoe_metrics

    file_path_predictions_pg = f"results/predictions/PGExplainer/{dataset}.json"
    file_path_evaluations_pg = f"results/evaluations/PGExplainer/{dataset}.json"
    with open(file_path_predictions_pg, "w") as json_file:
        json.dump(pg_preds, json_file, indent=2)
    with open(file_path_evaluations_pg, "w") as json_file:
        json.dump(pg_performance, json_file, indent=2)

    file_path_predictions_sgx = f"results/predictions/SubGraphX/{dataset}.json"
    file_path_evaluations_sgx = f"results/evaluations/SubGraphX/{dataset}.json"

    with open(file_path_predictions_sgx, "w") as json_file:
        json.dump(sgx_preds, json_file, indent=2)

    with open(file_path_evaluations_sgx, "w") as json_file:
        json.dump(sgx_performance, json_file, indent=2)

    file_path_predictions_evo = f"results/predictions/EVO/{dataset}.json"
    file_path_evaluations_evo = f"results/evaluations/EVO/{dataset}.json"

    with open(file_path_predictions_evo, "w") as json_file:
        json.dump(evo_preds, json_file, indent=2)

    with open(file_path_evaluations_evo, "w") as json_file:
        json.dump(evo_performance, json_file, indent=2)

    file_path_predictions_celoe = f"results/predictions/CELOE/{dataset}.json"
    file_path_evaluations_celoe = f"results/evaluations/CELOE/{dataset}.json"

    with open(file_path_predictions_celoe, "w") as json_file:
        json.dump(celoe_preds, json_file, indent=2)

    with open(file_path_evaluations_celoe, "w") as json_file:
        json.dump(celoe_performance, json_file, indent=2)
