# tests/test_your_module.py

import numpy as np
from sklearn.metrics import (f1_score, jaccard_score, precision_score,
                             recall_score)

from src.evaluations import calculate_micro_metrics


def test_calculate_micro_metrics_binary():
    # Binary classification example
    y_true_binary = np.array([1, 0, 1, 0, 1, 0])
    y_pred_binary = np.array([1, 1, 0, 0, 1, 1])

    # Calculate micro metrics for binary classification using scikit-learn
    sklearn_precision_binary = precision_score(
        y_true_binary, y_pred_binary, average="micro"
    )
    sklearn_recall_binary = recall_score(y_true_binary, y_pred_binary, average="micro")
    sklearn_f1_binary = f1_score(y_true_binary, y_pred_binary, average="micro")
    sklearn_jaccard_binary = jaccard_score(
        y_true_binary, y_pred_binary, average="micro"
    )

    # Calculate micro metrics for binary classification using your function
    (
        your_precision_binary,
        your_recall_binary,
        your_f1_binary,
        your_jaccard_binary,
    ) = calculate_micro_metrics(y_true_binary, y_pred_binary)

    # Assert that the results for binary classification match
    assert np.isclose(
        your_precision_binary, sklearn_precision_binary
    ), "Binary Precision does not match"
    assert np.isclose(
        your_recall_binary, sklearn_recall_binary
    ), "Binary Recall does not match"
    assert np.isclose(
        your_f1_binary, sklearn_f1_binary
    ), "Binary F1 score does not match"
    assert np.isclose(
        your_jaccard_binary, sklearn_jaccard_binary
    ), "Binary Jaccard score does not match"


def test_calculate_micro_metrics_multiclass():
    # Multi-class classification example
    y_true_multiclass = np.array([0, 1, 2, 0, 1, 2])
    y_pred_multiclass = np.array([0, 1, 2, 0, 2, 1])

    # Calculate micro metrics for multi-class classification using scikit-learn
    sklearn_precision_multiclass = precision_score(
        y_true_multiclass, y_pred_multiclass, average="micro"
    )
    sklearn_recall_multiclass = recall_score(
        y_true_multiclass, y_pred_multiclass, average="micro"
    )
    sklearn_f1_multiclass = f1_score(
        y_true_multiclass, y_pred_multiclass, average="micro"
    )
    sklearn_jaccard_multiclass = jaccard_score(
        y_true_multiclass, y_pred_multiclass, average="micro"
    )

    # Calculate micro metrics for multi-class classification using your function
    (
        your_precision_multiclass,
        your_recall_multiclass,
        your_f1_multiclass,
        your_jaccard_multiclass,
    ) = calculate_micro_metrics(y_true_multiclass, y_pred_multiclass)

    # Assert that the results for multi-class classification match
    assert np.isclose(
        your_precision_multiclass, sklearn_precision_multiclass
    ), "Multiclass Precision does not match"
    assert np.isclose(
        your_recall_multiclass, sklearn_recall_multiclass
    ), "Multiclass Recall does not match"
    assert np.isclose(
        your_f1_multiclass, sklearn_f1_multiclass
    ), "Multiclass F1 score does not match"
    assert np.isclose(
        your_jaccard_multiclass, sklearn_jaccard_multiclass
    ), "Multiclass Jaccard score does not match"
