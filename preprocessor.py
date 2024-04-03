import os


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
        "results/predictions/EvoLearner",
        "results/predictions/PGExplainer",
        "results/predictions/SubGraphX",
        "results/evaluations/CELOE",
        "results/evaluations/EvoLearner",
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


create_edge_directories()

