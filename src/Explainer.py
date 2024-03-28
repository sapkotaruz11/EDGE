import torch
import json
import os
import time

# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from torch import nn
import pandas as pd

torch.cuda.empty_cache()

from src.gnn_model.configs import get_configs
from src.gnn_model.dataset import RDFDatasets
from src.gnn_model.hetro_features import HeteroFeature
from src.gnn_model.model import RGCN
from src.gnn_model.utils import get_nodes_dict
from src.gnn_model.utils import get_lp_aifb_fid
from src.gnn_model.utils import get_lp_mutag_fid
from src.logical_explainers.EvoLearner import train_evo
from src.logical_explainers.CELOE import train_celoe
from src.gnn_model.utils import gen_evaluations
from src.gnn_model.utils import calculate_metrics


class EarlyStopping:
    """
    A utility class for early stopping during the training of machine learning models.

    This class is designed to monitor a model's loss during training and trigger an early stop
    if the loss does not decrease significantly over a specified number of epochs.
    It helps prevent overfitting by stopping the training process when the model's performance
    on the validation set starts degrading.

    Attributes:
        patience_decrease (int): The number of epochs with decreased loss to wait before resetting the counter. Defaults to 5.
        patience_increase (int): The number of epochs with increased loss to wait before triggering early stopping. Defaults to 10.
        delta (float): The minimum change in the monitored loss to qualify as an improvement or degradation. Defaults to 0.001.
        prev_score (float or None): The loss from the previous epoch. Initially None and updated during training.
        counter_decrease (int): Counts the number of epochs where loss has decreased. Resets if loss increases.
        counter_increase (int): Counts the number of epochs where loss has increased. Triggers early stopping
        if it reaches the patience threshold.
        early_stop (bool): Flag indicating whether early stopping has been triggered.

    Methods:
        __call__(loss): Updates the counters and early stopping flag based on the new loss value.

    Example:
        >>> early_stopping = EarlyStopping(patience_decrease=5, patience_increase=10, delta=0.001)
        >>> for epoch in range(epochs):
        >>>     train()  # Your training process here
        >>>     val_loss = validate()  # Your validation process here
        >>>     early_stopping(val_loss)
        >>>     if early_stopping.early_stop:
        >>>         print("Early stopping triggered")
        >>>         break

    Note:
        - This class should be instantiated and used as part of a training loop.
        - It is used in conjunction with a validation loss metric.
    """

    def __init__(self, patience_decrease=5, patience_increase=10, delta=0.001):
        self.patience_decrease = patience_decrease
        self.patience_increase = patience_increase
        self.delta = delta
        self.prev_score = None
        self.counter_decrease = 0
        self.counter_increase = 0
        self.early_stop = False

    def __call__(self, loss):
        if self.prev_score is not None:
            if loss < self.prev_score - self.delta:
                # Loss decreased
                self.counter_decrease += 0
            elif loss > self.prev_score + self.delta:
                # Loss increased
                self.counter_increase += 1
                if self.counter_increase >= self.patience_increase:
                    self.early_stop = True
            else:
                # Loss didn't decrease or increase significantly
                self.counter_decrease = 0
                self.counter_increase = 0

        self.prev_score = loss


class Explainer:
    def __init__(
        self,
        explainers: list,
        dataset: str,
        model_name: str = "RGCN",
    ):
        self.explainers = explainers
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        self.configs = get_configs(self.dataset)
        self.hidden_dim = self.configs["hidden_dim"]
        self.num_bases = self.configs["n_bases"]
        self.lr = self.configs["lr"]
        self.weight_decay = self.configs["weight_decay"]
        self.epochs = self.configs["max_epoch"]
        self.validation = self.configs["validation"]
        self.hidden_layers = self.configs["num_layers"] - 1
        self.act = None
        self.my_dataset = RDFDatasets(
            self.dataset, root="data/", validation=self.validation
        )

        self.g = self.my_dataset.g.to(self.device)
        self.out_dim = self.my_dataset.num_classes
        self.e_types = self.g.etypes
        self.category = self.my_dataset.category
        self.train_idx = self.my_dataset.train_idx.to(self.device)
        self.test_idx = self.my_dataset.test_idx.to(self.device)
        self.labels = self.my_dataset.labels.to(self.device)

        if self.validation:
            self.valid_idx = self.my_dataset.valid_idx.to(self.device)
        else:
            self.valid_idx = self.my_dataset.test_idx.to(self.device)

        self.idx_map = self.my_dataset.idx_map
        self.pred_df = pd.DataFrame(
            [
                {"IRI": self.idx_map[idx]["IRI"], "idx": idx}
                for idx in self.test_idx.tolist()
            ]
        )

        self.dataset_function_mapping = {
            "mutag": get_lp_mutag_fid,
            "aifb": get_lp_aifb_fid,
            # Add more dataset-function mappings as needed
        }

        self.time_traker = {}
        self.explanations = {}
        self.evaluations = {}

        self.input_feature = HeteroFeature(
            {}, get_nodes_dict(self.g), self.hidden_dim, act=self.act
        ).to(self.device)

        if self.model_name == "RGCN":

            self.model = RGCN(
                self.hidden_dim,
                self.hidden_dim,
                self.out_dim,
                self.e_types,
                self.num_bases,
                self.category,
                num_hidden_layers=self.hidden_layers,
            ).to(self.device)

            self.optimizer = th.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            self.loss_fn = F.cross_entropy
            self.model.add_module("input_feature", self.input_feature)
            self.optimizer.add_param_group({"params": self.input_feature.parameters()})
            self.early_stopping = EarlyStopping(
                patience_decrease=5, patience_increase=10, delta=0.001
            )
            self.feat = self.model.input_feature()

        self.train()
        self.run_explainers()

    def train(self):
        print("Start training...")
        dur = []
        train_accs = []
        val_accs = []
        vald_loss = []
        self.model.train()
        for epoch in range(self.epochs):
            t0 = time.time()
            logits = self.model(self.g, self.feat)[self.category]
            loss = self.loss_fn(logits[self.train_idx], self.labels[self.train_idx])
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            t1 = time.time()

            dur.append(t1 - t0)
            train_acc = th.sum(
                logits[self.train_idx].argmax(dim=1) == self.labels[self.train_idx]
            ).item() / len(self.train_idx)
            train_accs.append(train_acc)

            val_loss = F.cross_entropy(
                logits[self.valid_idx], self.labels[self.valid_idx]
            )
            val_acc = th.sum(
                logits[self.valid_idx].argmax(dim=1) == self.labels[self.valid_idx]
            ).item() / len(self.valid_idx)
            val_accs.append(val_acc)
            vald_loss.append(val_loss.item())
            print(
                "Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".format(
                    epoch,
                    train_acc,
                    loss.item(),
                    val_acc,
                    val_loss.item(),
                    np.average(dur),
                )
            )
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
        print("End Training")

        pred_logit = self.model(self.g, self.feat)[self.category]

        val_acc_final = th.sum(
            pred_logit[self.test_idx].argmax(dim=1) == self.labels[self.test_idx]
        ).item() / len(self.test_idx)
        print(
            f"Final validation accuracy of the model R-GCN on unseen dataset: {val_acc_final}"
        )

        if self.validation:
            self.total_train_idx = torch.cat([self.train_idx, self.valid_idx], dim=0)
        else:
            self.total_train_idx = self.train_idx

        # Train data
        self.gnn_pred_dt_train = {
            t_idx: pred
            for t_idx, pred in zip(
                self.total_train_idx.tolist(),
                pred_logit[self.total_train_idx].argmax(dim=1).tolist(),
            )
        }
        self.gt_dict_train = {
            t_idx: pred
            for t_idx, pred in zip(
                self.total_train_idx.tolist(),
                self.labels[self.total_train_idx].tolist(),
            )
        }

        # Test data
        self.gnn_pred_dt_test = {
            t_idx: pred
            for t_idx, pred in zip(
                self.test_idx.tolist(), pred_logit[self.test_idx].argmax(dim=1).tolist()
            )
        }
        self.gt_dict_test = {
            t_idx: pred
            for t_idx, pred in zip(
                self.test_idx.tolist(), self.labels[self.test_idx].tolist()
            )
        }

        self.pred_df["Ground Truths"] = self.pred_df["idx"].map(self.gt_dict_test)
        self.pred_df["GNN Preds"] = self.pred_df["idx"].map(self.gnn_pred_dt_test)
        self.create_lp()

    def _run_evo(self):
        print(f"Training EvoLearner (prediction) on {self.dataset}")
        target_dict_evo_pred, duration_evo_pred, _ = train_evo(
            learning_problems=self.learning_problem_pred, kg=self.dataset
        )
        print(
            f"Total time taken for EvoLearner (prediction)  on {self.dataset}: {duration_evo_pred:.2f}"
        )

        print(f"Training EvoLearner (explanation) on {self.dataset}")
        target_dict_evo_exp, duration_evo_exp, explanation_dict_evo = train_evo(
            learning_problems=self.learning_problem_fid, kg=self.dataset
        )
        print(
            f"Total time taken for EvoLearner (explanation)  on {self.dataset}: {duration_evo_exp:.2f}"
        )

        self.time_traker["EvoLearner"] = duration_evo_exp
        self.pred_df["Evo(pred)"] = self.pred_df["IRI"].map(target_dict_evo_pred)
        self.pred_df["Evo(exp)"] = self.pred_df["IRI"].map(target_dict_evo_exp)
        self.explanations["EvoLearner"] = explanation_dict_evo
        prediction_prefromance = calculate_metrics(
            self.pred_df["Ground Truths"], self.pred_df["Evo(pred)"]
        )
        explanation_prefromance = calculate_metrics(
            self.pred_df["GNN Preds"], self.pred_df["Evo(exp)"]
        )
        celoe_perfromance = gen_evaluations(
            prediction_perfromance=prediction_prefromance,
            explanation_perfromance=explanation_prefromance,
        )
        self.evaluations["EvoLearner"] = celoe_perfromance

    def _run_celoe(self):
        print(f"Training CELOE (prediction) on {self.dataset}")
        target_dict_celoe_pred, duration_celoe_pred, _ = train_celoe(
            learning_problems=self.learning_problem_pred, kg=self.dataset
        )
        print(
            f"Total time taken for CELOE (prediction)  on {self.dataset}: {duration_celoe_pred:.2f}"
        )

        print(f"Training CELOE (explanation) on {self.dataset}")
        target_dict_celoe_exp, duration_celoe_exp, explanation_dict_celoe = train_celoe(
            learning_problems=self.learning_problem_fid, kg=self.dataset
        )
        print(
            f"Total time taken for CELOE (explanation)  on {self.dataset}: {duration_celoe_exp:.2f}"
        )

        self.time_traker["CELOE"] = duration_celoe_exp
        self.pred_df["CELOE(pred)"] = self.pred_df["IRI"].map(target_dict_celoe_pred)
        self.pred_df["CELOE(exp)"] = self.pred_df["IRI"].map(target_dict_celoe_exp)
        self.explanations["CELOE"] = explanation_dict_celoe

        prediction_prefromance = calculate_metrics(
            self.pred_df["Ground Truths"], self.pred_df["CELOE(pred)"]
        )
        explanation_prefromance = calculate_metrics(
            self.pred_df["GNN Preds"], self.pred_df["CELOE(exp)"]
        )
        celoe_perfromance = gen_evaluations(
            prediction_perfromance=prediction_prefromance,
            explanation_perfromance=explanation_prefromance,
        )
        self.evaluations["CELOE"] = celoe_perfromance

    def _run_pgexplainer(self, print_explainer_loss=True):
        from src.dglnn_local.pgexplainer import HeteroPGExplainer

        print("Starting PGExplainer")
        t0 = time.time()
        explainer_pg = HeteroPGExplainer(
            self.model, self.hidden_dim, num_hops=1, explain_graph=False
        )
        feat_pg = {item: self.feat[item].data for item in self.feat}
        init_tmp, final_tmp = 5.0, 1.0
        optimizer_exp = th.optim.Adam(explainer_pg.parameters(), lr=0.01)
        for epoch in range(20):
            tmp = float(init_tmp * np.power(final_tmp / init_tmp, epoch / 20))
            loss = explainer_pg.train_step_node(
                {
                    ntype: self.g.nodes(ntype)
                    for ntype in self.g.ntypes
                    if ntype == self.category
                },
                self.g,
                feat_pg,
                tmp,
            )
            optimizer_exp.zero_grad()
            loss.backward()
            optimizer_exp.step()
            if print_explainer_loss:
                print(f"Explainer trained for {epoch} epochs with loss {loss :3f}")
        self.entity_pg = {}
        probs, edge_mask, bg, inverse_indices = explainer_pg.explain_node(
            {self.category: self.test_idx}, self.g, self.feat, training=True
        )
        exp_pred_pg = (
            probs[self.category][inverse_indices[self.category]].argmax(dim=1).tolist()
        )
        t1 = time.time()
        pg_preds = dict(zip(self.test_idx.tolist(), exp_pred_pg))
        self.pred_df["PGExplainer"] = self.pred_df["idx"].map(pg_preds)

        prediction_prefromance = calculate_metrics(
            self.pred_df["Ground Truths"], self.pred_df["PGExplainer"]
        )
        explanation_prefromance = calculate_metrics(
            self.pred_df["GNN Preds"], self.pred_df["PGExplainer"]
        )
        pg_perfromance = gen_evaluations(
            prediction_perfromance=prediction_prefromance,
            explanation_perfromance=explanation_prefromance,
        )
        self.evaluations["PGExplainer"] = pg_perfromance

        dur_pg = t1 - t0
        self.time_traker["PGExplainer"] = dur_pg
        print(f"Total time taken for PGExplainer  on {self.dataset}: {dur_pg:.2f}")

    def _run_subgraphx(self):
        from src.dglnn_local.subgraphx import HeteroSubgraphX

        print("Starting SubGraphX")
        t0 = time.time()
        explainer_sgx = HeteroSubgraphX(
            self.model, num_hops=1, num_rollouts=3, shapley_steps=5
        )
        exp_preds_sgx = {}
        for idx in self.test_idx.tolist():
            print(idx)
            explanation, prediction = explainer_sgx.explain_node(
                self.g, self.feat, idx, self.category
            )
            exp_preds_sgx[idx] = prediction

        t1 = time.time()
        self.pred_df["SubGraphX"] = self.pred_df["idx"].map(exp_preds_sgx)
        dur_sgx = t0 - t1
        self.time_traker["SubGraphX"] = dur_sgx

        prediction_prefromance = calculate_metrics(
            self.pred_df["Ground Truths"], self.pred_df["SubGraphX"]
        )
        explanation_prefromance = calculate_metrics(
            self.pred_df["GNN Preds"], self.pred_df["SubGraphX"]
        )
        sgx_perfromance = gen_evaluations(
            prediction_perfromance=prediction_prefromance,
            explanation_perfromance=explanation_prefromance,
        )
        self.evaluations["SubGraphX"] = sgx_perfromance

        print(f"Total time taken for SubGraphX  on {self.dataset}: {dur_sgx:.2f}")

    def run_explainers(self):
        explainer_methods = {
            "EvoLearner": self._run_evo,
            "CELOE": self._run_celoe,
            "PGExplainer": self._run_pgexplainer,
            "SubGraphX": self._run_subgraphx,
            # Add more explainer-method mappings as needed
        }

        for explainer in self.explainers:
            explainer_method = explainer_methods.get(explainer)
            if explainer_method:
                explainer_method()

    def create_lp(self):
        self.lp_function = self.dataset_function_mapping.get(self.dataset)

        if self.lp_function:
            self.learning_problem_pred = self.lp_function(
                self.gt_dict_train, self.gt_dict_test, self.idx_map
            )

            self.learning_problem_fid = self.lp_function(
                self.gnn_pred_dt_train, self.gnn_pred_dt_test, self.idx_map
            )
