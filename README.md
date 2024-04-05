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


### Datasets
1. **Mutag**
2. **AIFB**
3. **BGS**


# Installation Guide for the EDGE Framework

Follow these steps to set up the EDGE environment on your system:

## Step 1: Clone the EDGE Repository

First, clone the EDGE repository from GitHub using the following command:

```shell
git clone https://github.com/sapkotaruz11/EDGE.git
```

## Step 2: Install Python-virtual environment

If you don't have Python virtual environment installed, download and install it using:
```shell
pip install virtualenv
```


## Step 3: Create the Python Environment

Open your terminal or command prompt and run the following command to create a Python environment named `edge` ( or anything you want) with Python 3.10:

```shell
python3 -m venv edge
```
If you wish, you can also use a conda environment.

## Step 4: Activate the Environment

Activate the newly created environment using: (or change the environment name if you used any other names)

```shell
source edge/bin/activate
```
Change to the cloned directory:

```shell
cd EDGE
```

## Step 5: Install Dependencies

Ensure you have a `requirements.txt` file in your project directory. To install the required dependencies, run:

```shell
pip install -r requirements.txt
```

This command will automatically install all the libraries and packages listed in your `requirements.txt` file.

After completing these steps, your EDGE environment should be set up with all the necessary dependencies.

If you are using GPU devices, install the GPU version of DGL from official DGL website.

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
   Ensure that the modification is correct and the file is saved properly. This step is crucial for successfully using the AIFB dataset with EvoLearner.

By following these steps, you'll make the AIFB dataset compatible with EvoLearner, avoiding potential processing errors.


# Example Commands for Using the EDGE Framework

Below are some example commands to illustrate how to use the EDGE framework. These examples assume you have already set up the environment per the installation guide.

## Evaluating with Existing Prediction Data
By default, the script evaluates using existing prediction data:

```shell
python main.py
```

## Training Models with Specific Specifications
To train models with specific models and/or datasets, use the `--train` flag along with `--model`, `--explainers`  and `--datasets` flags as needed:

- Training specific explainers:
  ```shell
  python main.py --train  --explainers PGExplainer EvoLearner 
  ```

- Training models on specific datasets:
  ```shell
  python main.py --train --datasets mutag bgs
  ```

- Combining specific explainers and datasets:
  ```shell
  python main.py --train --explainers SubGraphX CELOE --datasets aifb
  ```
- Training specific GNN Model:
  ```shell
  python main.py --train  --model RGCN 
  ```


If you wisth to train for all the explainers and datasets, you can simply omit the tags from the arguments. The default number of runs is 5 and the default GNN Model is RGCN.

- Training for certains number of runs for all explainers-dataset combo:
  ```shell
  python main.py --train  --num_runs 3 
  ```
There is also support for prining results, in which you can specify the model you want to print results for, defaults to "RGCN"

- Print Results for RGCN model
  ```shell
  python main.py --print_results --model GIN
  ```