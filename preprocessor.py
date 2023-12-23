from src.utils.preprocess_kgs import pre_process_aifb, pre_process_mutag
from src.gnn_explainers.trainer import train_gnn
from src.utils.create_lp import create_lp_aifb, create_lp_mutag


train_gnn(dataset="mutag")
train_gnn(dataset="aifb")

pre_process_aifb()
pre_process_mutag()

create_lp_aifb()
create_lp_mutag()
