import pandas as pd
import pytest

from fraud_detection import evaluate_fraud_dataset


def sample_data():
    return pd.DataFrame({
        "amount": [10, 12, 9, 11, 500, 450, 600, 550, 8, 13, 520, 7, 14, 480, 6, 590, 15, 530, 5, 610],
        "merchant": ["A", "B", "A", "C", "X", "Y", "X", "Y", "B", "C", "X", "A", "B", "Y", "C", "X", "A", "Y", "B", "X"],
        "Class": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
    })


def test_evaluation_uses_a_held_out_test_set_and_returns_three_models():
    result = evaluate_fraud_dataset(sample_data())
    assert result["training_rows"] + result["testing_rows"] == result["rows_analyzed"]
    assert set(result["models"]) == {"Isolation Forest", "Linear SVM", "Logistic Regression", "Random Forest"}
    for metrics in result["models"].values():
        assert 0 <= metrics["accuracy"] <= 1
        assert len(metrics["confusion_matrix"]) == 2
        assert 0 <= metrics["balanced_accuracy"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["pr_auc"] <= 1


def test_missing_target_is_rejected_instead_of_inventing_labels():
    with pytest.raises(ValueError, match="binary target column is required"):
        evaluate_fraud_dataset(sample_data().drop(columns=["Class"]))


def test_non_binary_target_is_rejected():
    frame = sample_data()
    frame["Class"] = 2
    with pytest.raises(ValueError, match="binary labels"):
        evaluate_fraud_dataset(frame)


def test_missing_values_are_supported_in_features():
    frame = sample_data()
    frame.loc[0, "amount"] = None
    result = evaluate_fraud_dataset(frame)
    assert result["rows_analyzed"] == len(frame)
