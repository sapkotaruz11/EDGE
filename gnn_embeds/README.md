## Experiment Overview

### Objective
The goal is to train an **RGCN model** for link prediction on a knowledge graph using different types of embeddings and evaluate its performance using **MRR** and Hits metrics.

### Key Features
- **Dataset**: Uses **UMLS** (default) or **FB15k-237** (larger dataset).
- **Model**: **RGCN** with support for multiple relations and neighborhood aggregation.
- **Embeddings**: Experiments with both **pre-trained** and **randomly initialized** embeddings.

### Training
- **Device**: Trains on **GPU** if available.
- **Batching**: Small datasets are trained in full, larger ones in chunks.

### Evaluation
- **MRR** and **Hits@1, 3, 10** are calculated during training (every 25 epochs) and on the test set at the end.
- Results are stored for both **pre-trained** and **random embeddings**.

 


## Evaluations

### Overview

This evaluation script implements a function to calculate the Mean Reciprocal Rank (MRR) and Hit@k metrics for evaluating knowledge graph embedding models. The evaluation is done by perturbing triplets and measuring the model's ability to rank the correct entity among a list of candidates.


1. **Perturbation**:
   - **Subject Perturbation**: The object in the test triplet is replaced with all other possible entities, and the rank of the correct object is computed.
   - **Object Perturbation**: The subject in the test triplet is replaced with all other possible entities, and the rank of the correct subject is computed.

2. **Score Calculation**:
   - The score for each candidate entity is computed using the dot product of the embeddings of the perturbed triplet and the candidate entities.
   - The resulting scores are normalized using the sigmoid function.

3. **Ranking and Metrics**:
   - The ranks of the correct entities are computed, and the ranks are converted to a 1-indexed format.
   - **MRR**: The Mean Reciprocal Rank is computed by taking the mean of the reciprocal ranks across all test triplets.
   - **Hit@k**: The Hit@k metrics are computed by measuring the proportion of times the correct entity appears in the top k ranks.

4. **Output**:
   - The MRR and Hit@k metrics are printed and returned in a dictionary format.


