# tests/test_your_module.py

import math

from sklearn.metrics import jaccard_score, precision_recall_fscore_support

from src.evaluations_macro import calculate_metrics_logical


def assert_almost_equal(actual, expected, rel_tol=1e-9, abs_tol=0.0):
    assert math.isclose(
        actual, expected, rel_tol=rel_tol, abs_tol=abs_tol
    ), f"{actual} != {expected}"


def test_binary_classification():
    # Binary classification test case with strings
    predictions_data_binary = {
        "problem1": {
            "concept_individuals": ["apple", "banana", "cherry"],
            "positive_examples": ["banana", "cherry", "date"],
            "negative_examples": ["apple", "kiwi", "orange"],
        }
    }

    for _, examples in predictions_data_binary.items():
        concept_individuals = set(examples["concept_individuals"])
        pos = set(examples["positive_examples"])
        neg = set(examples["negative_examples"])

        all_examples = pos.union(neg)

        y_true = [1 if item in pos else 0 for item in all_examples]
        y_pred = [1 if item in concept_individuals else 0 for item in all_examples]

    precision, recall, f1_score, jaccard_similarity = calculate_metrics_logical(
        predictions_data_binary
    )
    (
        sklearn_precision,
        sklearn_recall,
        sklearn_f1_score,
        _,
    ) = precision_recall_fscore_support(y_true, y_pred, average="macro")
    sklearn_jaccard_similarity = jaccard_score(y_true, y_pred, average="macro")

    # Assert that the results are consistent
    assert_almost_equal(precision, sklearn_precision)
    assert_almost_equal(recall, sklearn_recall)
    assert_almost_equal(f1_score, sklearn_f1_score)
    assert_almost_equal(jaccard_similarity, sklearn_jaccard_similarity)


# def test_multi_class_classification():
#     # Multi-class classification test case with strings

#     # Multi-class classification test case with strings
#     predictions_data_multi_class = {
#         "problem1": {
#             "concept_individuals": [
#                 "banana",
#                 "cherry",
#                 "date",
#                 "orange",
#             ],  # Adding "kiwi" as a concept individual
#             "positive_examples": ["banana", "cherry", "date", "kiwi"],
#             "negative_examples": ["apple", "orange"],
#         },
#         "problem2": {
#             "concept_individuals": ["orange", "pear"],
#             "positive_examples": ["orange", "pear"],
#             "negative_examples": ["banana", "cherry", "date", "kiwi"],
#         },
#     }

#     precision, recall, f1_score, jaccard_similarity = calculate_metrics_logical(
#         predictions_data_multi_class
#     )

#     # Calculate the same metrics using scikit-learn with micro-averaging
#     y_true = [1, 1, 1, 1, 2, 2]  # Multi-class representation of samples
#     y_pred = [1, 1, 1, 2, 2, 2]  # Multi-class representation of predictions

#     (
#         sklearn_precision,
#         sklearn_recall,
#         sklearn_f1_score,
#         _,
#     ) = precision_recall_fscore_support(y_true, y_pred, average="macro")
#     sklearn_jaccard_similarity = jaccard_score(y_true, y_pred, average="macro")

#     # Assert that the results are consistent
#     assert_almost_equal(precision, sklearn_precision)
#     assert_almost_equal(recall, sklearn_recall)
#     assert_almost_equal(f1_score, sklearn_f1_score)
#     assert_almost_equal(jaccard_similarity, sklearn_jaccard_similarity)

#     precision, recall, f1_score, jaccard_similarity = calculate_metrics_logical(
#         predictions_data_multi_class
#     )


# if __name__ == "__main__":
#     test_binary_classification()
#     test_multi_class_classification()
