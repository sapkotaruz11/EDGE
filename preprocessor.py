import os
from src.utils.create_lp import create_lp_aifb, create_lp_mutag
from src.utils.preprocess_kgs import pre_process_aifb, pre_process_mutag


def create_edge_directories():
    """
    Create the necessary EDGE framework directories if they don't exist.
    """
    # List of subdirectories for "results" and "data"
    sub_dirs = [
        "results",
        "data",
        "data/KGs",
        "results/dataframes",
        "results/evaluations",
        "results/exp_visualizations",
        "results/predictions",
        "results/predictions/CELOE",
        "results/predictions/EVO",
        "results/predictions/PGExplainer",
        "results/predictions/SubGraphX",
        "results/evaluations/CELOE",
        "results/evaluations/EVO",
        "results/evaluations/PGExplainer",
        "results/evaluations/SubGraphX",
    ]

    # Function to create directories if they don't exist
    def create_directories(dirs):
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)
                print(f"Created directory: {dir}")
            else:
                print(f"Directory already exists: {dir}")

    # Create the directories
    create_directories(sub_dirs)


# create_edge_directories()

# pre_process_aifb()
# pre_process_mutag()
create_lp_aifb()
create_lp_mutag()
