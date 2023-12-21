# from src.utils.create_lp import create_lp_aifb, create_lp_mutag

# create_lp_aifb()
# create_lp_mutag()

# from src.gnn_explainers.HeteroPG_explainer import explain_PG
# from src.gnn_explainers.HeteroSubGraphX_explainer import explain_SGX


# explain_PG(dataset="aifb", print_explainer_loss=True)
# explain_PG(dataset="mutag", print_explainer_loss=True)

# explain_SGX(dataset="aifb")
# explain_SGX(dataset="mutag")


# from src.logical_explainers.EvoLearner import train_evo, train_evo_fid
# train_evo()
# train_evo_fid()

# from src.logical_explainers.CELOE import train_celoe, train_celoe_fid
# train_celoe(use_heur= False)
# train_celoe_fid(use_heur=False)

# from src.evaluations_macro import evaluate_gnn_explainers, evaluate_logical_explainers

# evaluate_logical_explainers()
# evaluate_gnn_explainers()
# # from src.print_results import print_results

# # print_results()
# # print("_____________________________")
# # print("_____________________________")
from src.evaluations import evaluate_gnn_explainers, evaluate_logical_explainers

evaluate_logical_explainers()
evaluate_gnn_explainers()

from src.print_results import print_results

print_results()

# from src.gnn_explainers.trainer import train_gnn
# train_gnn(dataset="aifb")
# train_gnn(dataset="mutag")

# from src.utils.preprocess_kgs import pre_process_aifb, pre_process_mutag

# pre_process_mutag()
# pre_process_aifb()
