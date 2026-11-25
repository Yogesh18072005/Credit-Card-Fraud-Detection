"""Leakage-safe model training and evaluation for binary fraud datasets."""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

TARGET_ALIASES = ("Class", "class", "fraudulent", "label", "is_fraud")


def _find_target(frame: pd.DataFrame, target_column: str | None) -> str:
    if target_column:
        if target_column not in frame.columns:
            raise ValueError(f"Target column '{target_column}' was not found.")
        return target_column
    for candidate in TARGET_ALIASES:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        "A binary target column is required. Use one of: "
        + ", ".join(TARGET_ALIASES)
        + ". The app will not invent fraud labels."
    )


def _build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [c for c in features.columns if c not in numeric_columns]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if not transformers:
        raise ValueError("The CSV must contain at least one usable feature column.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _best_f1_threshold(y_true: pd.Series, scores: Any) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(f1_values.argmax())])


def _metrics(y_true: pd.Series, predictions: Any, scores: Any, threshold: float) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "threshold": float(threshold),
        "classification_report": classification_report(
            y_true, predictions, labels=[0, 1],
            target_names=["Legitimate", "Fraud"], zero_division=0
        ),
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_fraud_dataset(
    frame: pd.DataFrame,
    target_column: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Tune supervised thresholds on validation data and evaluate on held-out rows."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("The uploaded CSV is empty.")
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("Test size must be between 0.1 and 0.4.")

    target = _find_target(frame, target_column)
    data = frame.copy()
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna(subset=[target])
    if data.empty:
        raise ValueError("The target column contains no valid labels.")
    labels = set(data[target].unique().tolist())
    if not labels.issubset({0, 1}) or len(labels) != 2:
        raise ValueError("The target column must contain both binary labels: 0 (legitimate) and 1 (fraud).")

    y = data[target].astype(int)
    X = data.drop(columns=[target])
    if X.empty:
        raise ValueError("The CSV must contain at least one feature besides the target.")
    if y.value_counts().min() < 4:
        raise ValueError("At least four examples of each class are required for train, validation, and test splits.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    if y_train.value_counts().min() < 2:
        raise ValueError("Not enough examples of each class remain for model validation.")
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train, y_train, test_size=0.25, random_state=random_state, stratify=y_train
    )
    fraud_rate = float(y_train.mean())
    contamination = min(max(fraud_rate, 0.001), 0.5)

    models: dict[str, Any] = {
        "Isolation Forest": IsolationForest(
            n_estimators=200, contamination=contamination,
            random_state=random_state, n_jobs=-1
        ),
        "Linear SVM": LinearSVC(
            class_weight="balanced", random_state=random_state, max_iter=5000
        ),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, class_weight="balanced_subsample",
            random_state=random_state, n_jobs=-1
        ),
    }

    results: dict[str, Any] = {}
    fitted_pipelines: dict[str, Pipeline] = {}
    for name, estimator in models.items():
        validation_pipeline = Pipeline([
            ("preprocessor", _build_preprocessor(X_fit)),
            ("model", estimator),
        ])
        validation_pipeline.fit(X_fit, y_fit if name != "Isolation Forest" else None)
        if name == "Isolation Forest":
            threshold = 0.0
        elif hasattr(validation_pipeline, "predict_proba"):
            validation_scores = validation_pipeline.predict_proba(X_validation)[:, 1]
            threshold = _best_f1_threshold(y_validation, validation_scores)
        else:
            validation_scores = validation_pipeline.decision_function(X_validation)
            threshold = _best_f1_threshold(y_validation, validation_scores)
        pipeline = Pipeline([
            ("preprocessor", _build_preprocessor(X_train)),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train if name != "Isolation Forest" else None)
        fitted_pipelines[name] = pipeline
        if name == "Isolation Forest":
            scores = -pipeline.decision_function(X_test)
        elif hasattr(pipeline, "predict_proba"):
            scores = pipeline.predict_proba(X_test)[:, 1]
        else:
            scores = pipeline.decision_function(X_test)
        predictions = (scores >= threshold).astype(int)
        model_result = _metrics(y_test, predictions, scores, threshold)
        model_result["error_rate"] = 1.0 - model_result["accuracy"]
        results[name] = model_result

    best_model_name = max(
        ("Linear SVM", "Logistic Regression", "Random Forest"),
        key=lambda model_name: results[model_name]["pr_auc"],
    )
    explain_rows = min(len(X_test), 500)
    importance = permutation_importance(
        fitted_pipelines[best_model_name],
        X_test.iloc[:explain_rows],
        y_test.iloc[:explain_rows],
        scoring="average_precision",
        n_repeats=3,
        random_state=random_state,
        n_jobs=1,
    )
    feature_importance = [
        {"feature": str(feature), "importance": float(score)}
        for feature, score in sorted(
            zip(X.columns, importance.importances_mean),
            key=lambda item: item[1], reverse=True
        )[:10]
    ]

    return {
        "target_column": target,
        "rows_analyzed": int(len(data)),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_validation)),
        "testing_rows": int(len(X_test)),
        "fraudulent_count": int(y.sum()),
        "non_fraudulent_count": int((y == 0).sum()),
        "fraud_rate": float(y.mean()),
        "statistical_analysis": data.describe(include="all").fillna("").to_string(),
        "models": results,
        "feature_importance": feature_importance,
        "explanation_model": best_model_name,
        "pipelines": fitted_pipelines,
    }
