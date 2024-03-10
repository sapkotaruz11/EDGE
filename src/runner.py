from src.logical_explainers.EvoLearner import train_evo

# from src.logical_explainers.CELOE import train_celoe
from src.gnn_explainers.trainer import train_gnn
import json


def main_runner(models, datasets):
    print(models, datasets)
    # train_gnn(dataset="aifb")
    json_file_path = "configs/aifb.json"
    with open(json_file_path, "r", encoding="utf-8") as json_file:
        learning_problems = json.load(json_file)
    train_evo(learning_problems=learning_problems, kgs=["aifb"])
