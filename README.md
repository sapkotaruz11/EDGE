# EDGE: Evaluation of Diverse Knowledge Graph Explanations

The **EDGE** framework represents a novel approach in evaluating explanations produced by various node classifiers on knowledge graphs. Standing for "Evaluation of Diverse Knowledge Graph Explanations," EDGE integrates an array of advanced Graph Neural Networks (GNNs), sub-graph-based GNN explainers, logical explainers, and a comprehensive set of evaluation metrics. This framework is designed to automate the evaluation process, efficiently handling methods within its scope and delivering results in a clear, structured manner. The primary goals of EDGE are to incorporate cutting-edge node classifiers from existing literature, to provide a quantitative assessment of different explainers using multiple metrics, to streamline the evaluation process, and to conduct evaluations using real-world datasets.


### Logical Approaches
1. **EvoLearner:** [EvoLearner: Learning Description Logics with Evolutionary Algorithms](https://arxiv.org/abs/2111.04879)
2. **CELOE:**  [Class Expression Learning for Ontology Engineering](https://www.sciencedirect.com/science/article/pii/S1570826811000023)

The logical approaches in the EDGE framework, including EvoLearner and CELOE, were adapted from [OntoLearn](https://github.com/dice-group/Ontolearn).


### Sub-graph-based Approaches
1. **PGExplainer:**  [Parameterized Explainer for Graph Neural Network](https://arxiv.org/abs/2011.04573)
2. **SubgraphX:**  [On Explainability of Graph Neural Networks via Subgraph Explorations](https://arxiv.org/abs/2102.05152)

The sub-graph-based approaches in the EDGE framework, including PGExplainer and SubgraphX, were adapted from the [DGL (Deep Graph Library)](https://docs.dgl.ai/en/1.1.x/api/python/nn-pytorch.html).

These explainers collectively enhance the capability of the EDGE framework to provide comprehensive and insightful evaluations.



# Installation Guide for the EDGE Framework

Follow these steps to set up the EDGE environment on your system:

## Step 1: Clone the EDGE Repository

First, clone the EDGE repository from GitHub using the following command:

```shell
git clone https://github.com/sapkotaruz11/EDGE.git
```

Change to the cloned directory:

```shell
cd EDGE
```

## Step 2: Install Conda

If you don't have Conda installed, download and install it from [Anaconda's official website](https://www.anaconda.com/products/individual).

## Step 3: Create the Conda Environment

Open your terminal or command prompt and run the following command to create a Conda environment named EDGE with Python 3.10:

```shell
conda create --name EDGE python=3.10
```

## Step 4: Activate the Environment

Activate the newly created environment using:

```shell
conda activate EDGE
```

## Step 5: Install Dependencies

Ensure you have a `requirements.txt` file in your project directory. To install the required dependencies, run:

```shell
pip install -r requirements.txt
```

This command will automatically install all the libraries and packages listed in your `requirements.txt` file.

After completing these steps, your EDGE environment should be set up with all the necessary dependencies.

# Preprocessing Datasets and Training GNN Model

## Note for Using Existing Resources

If you wish to use the existing learning problems and datasets that are available with the EDGE framework, this preprocessing step can be skipped.

## Important Note

Please do not delete any directories from the original setup of the EDGE framework. If you wish to create new files or start with a fresh setup, it is recommended to remove the old files manually. Deleting directories may result in the loss of some `scripts` and data analysis tools.

However, if you accidentally delete any directories, you can recreate all the necessary directories for "results" and "data" using the "create_edge_directories" function provided in the preprocessor. Please note that by doing this, you may lose some `scripts` and tools for data analysis that were originally included in the directories.


## Step-by-Step Instructions

Before running evaluations or training with the EDGE framework, it's essential to preprocess the datasets and train the Graph Neural Network (GNN) model. This process also includes creating learning problems for the logical approaches. Follow the steps below to execute these tasks:

## Running the Preprocessor Script

To start the preprocessing and training, run the `preprocessor.py` script. This script will handle training the GNN model and setting up learning problems for logical explainers:

```shell
python preprocessor.py
```

## Installing the ROBOT Tool

For converting N3/NT files to the OWL file format within the EDGE framework, the ROBOT (RObotic Batch Ontology) tool is required. 

Download the ROBOT tool from its official website for the latest release and installation instructions:

[ROBOT Official Website](http://robot.obolibrary.org/)

Follow the instructions on the website to download and install ROBOT. Ensure it's properly installed and configured on your system for use with the EDGE framework.



## Navigating to the Knowledge Graphs Directory

After running the `preprocessor.py` script, navigate to the directory containing the Knowledge Graphs (KGs) with the following command:

```shell
cd data/KGs
```

## Converting the N3/NT files to OWL format

```shell
robot convert --input aifbfixed_complete_processed.n3 --output aifb.owl
robot convert --input mutag_stripped_processed.nt --output mutag.owl
```
### Modifying the AIFB Dataset for Compatibility with EvoLearner

When using the AIFB dataset with EvoLearner, you need to remove a specific line from the `aifb.owl` file to avoid the 'PSet Terminals have to have unique names' error.


1. **Locate the File:**
   Open the `aifb.owl` file. This file should be located in the dataset directory of your EDGE framework setup.

2. **Find the Specific Line:**
   Search for the line containing the following data:

   \```
   <owl:Class rdf:about="http://www.w3.org/2002/07/owl#Thing"/>
   \```

   This line is typically around line 932, but it might vary slightly depending on the file version.

3. **Remove the Line:**
   Delete this entire line from the file. Be careful not to remove or alter any other lines.

4. **Save the File:**
   After removing the line, save the changes to `aifb.owl`.

5. **Verify the Changes:**
   Ensure that the modification is correct and the file is saved properly. This step is crucial for the successful use of the AIFB dataset with EvoLearner.

By following these steps, you'll make the AIFB dataset compatible with EvoLearner, thereby avoiding potential errors during processing.


### What Does `preprocessor.py` Do?

- **Train GNN Model:** The script trains the GNN model on the specified datasets, ensuring that the model is ready for evaluation or further training.
  
- **Create Learning Problems:** It also generates learning problems required for the logical approaches like EvoLearner and CELOE, setting the stage for evaluation.

### When to Run This Script

- **Before First Use:** It's crucial to run this script before the first use of the framework to ensure all components are properly initialized and trained.


Running the `preprocessor.py` script ensures that your EDGE framework is fully prepared with trained models and set learning problems, providing a  foundation for your evaluation tasks.


# Example Commands for Using the EDGE Framework

Below are some example commands to illustrate how to use the EDGE framework. These examples assume that you have already set up the environment as per the installation guide.

## Evaluating with Existing Prediction Data
By default, the script evaluates using existing prediction data:

```shell
python main.py
```

## Training Models with Specific Specifications
To train models with specific models and/or datasets, use the `--retrain_models` flag along with `--models` and `--datasets` flags as needed:

- Training specific models:
  ```shell
  python main.py --retrain_models --models PGExplainer EvoLearner
  ```

- Training models on specific datasets:
  ```shell
  python main.py --retrain_models --datasets Mutag
  ```

- Combining specific models and datasets:
  ```shell
  python main.py --retrain_models --models SubGraphX CELOE --datasets AIFB
  ```

## Selecting Evaluation Type
Choose between micro and macro evaluations:

- For micro evaluation (default):

  ```shell
  python main.py
  ```

- For macro evaluation:

  ```shell
  python main.py --evaluation_type macro
  ```

Use these commands as a guide to operate the EDGE framework according to your specific needs.
