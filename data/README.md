# Edge Framework Directory Structure

The directory consists of raw datasets, extracted and processed datasets, and a 'KGs' folder containing data in OWL format.

## Directory Overview

- **Raw Datasets (ZIP folder)**
- **Extracted and Processed Datasets**
- **Kgs Folder (OWL Format Data)**

### Raw Datasets

This folder contains the raw datasets used in the Edge Framework. These datasets are provided in ZIP format for compact storage and are used to create the dataset objects if required

#### Contents:
- `aifb-hetero.zip`
- `mutag.zip`

### Extracted and Processed Datasets

This section includes datasets that have been extracted from the ZIP files and processed for use in the framework. The processing are done using the dataset script in `src\gnn_explainers\dataset.py` . The processed fies also contain train/test split data files.

#### Contents:
- `aifb-hetero_82d021d8`
- `mutag-hetero_faec5b61`

### Kgs Folder

The 'Kgs' folder contains datasets in OWL (Web Ontology Language) format. These are structured datasets that are used for knowledge representation for the training of Logical approaches in the Edge framework.

#### Contents:
- `mutag.owl`
- `aifb.owl`

