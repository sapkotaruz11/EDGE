# `src` Folder Structure

This document provides an overview of the `src` folder, which contains all the major code for the framework. The folder is organized into several subfolders, each dedicated to different aspects of the framework's functionality.

## Directory Overview

- **dglnn_local**
- **logical_explainers**
- **gnn_expllainers**
- **utils**
- `evaluation.py`
- `evaluation_macro.py`

### dglnn_local

This folder contains scripts adapted from the DGL (Deep Graph Library) library for local implementation. These scripts are tailored to fit the models and datasets specific to our framework.

#### Contents:
- `GNNExplainer(not used for evalution)`
- `PGExplainer`
- `SubGraphX`
- `RDFDataset`

### logical_explainers

Includes scripts used to train logical approaches within the framework.

#### Contents:
- `EvoLearner`
- `CELOE`


### gnn_explainers

Contains the model training scripts for subgraph-based GNN (Graph Neural Network) explainers.

#### Contents:
- `.SubGraphX`
-  `PGExplainer`

The directory also consists of scripts to train the GNN model inclusing config data for different dataset while training the model as well as the script to build the RDFdatasets.

### utils

The utils folder contains various utility scripts and its own README file, providing detailed documentation and usage instructions for the utilities.



### Directory-Level Scripts

#### `evaluation.py`

Used for performing evaluations  with micro-analysis within the framework. The script has separate functions to process Logical and sub-graph (GNN) explainers.

#### `evaluation_macro.py`

Used for performing evaluations  with macro-analysis within the framework. The script has separate functions to process Logical and sub-graph (GNN) explainers.

#### `eprint_results.py`
Script to print the results that are saved on the `resuts/evaluations`. The scripts 
