import json
from io import BytesIO

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier

from app import app, save_model_artifacts


def test_predict_requires_login():
    app.config["TESTING"] = True
    client = app.test_client()
    response = client.post("/predict")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_model_artifacts_are_saved_with_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr("app.MODEL_DIR", tmp_path)
    analysis = {
        "target_column": "Class",
        "models": {"Demo Model": {"threshold": 0.42, "pr_auc": 0.91}},
        "pipelines": {"Demo Model": DummyClassifier(strategy="most_frequent")},
    }

    manifest = save_model_artifacts(analysis)

    assert (tmp_path / "demo_model.joblib").is_file()
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.is_file()
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved == manifest
    assert saved["models"]["Demo Model"]["threshold"] == 0.42
    assert "pipelines" not in analysis


def test_saved_model_inference_requires_a_registered_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr("app.MODEL_DIR", tmp_path)
    app.config["TESTING"] = True
    client = app.test_client()
    with client.session_transaction() as session:
        session["username"] = "test-user"

    response = client.post(
        "/api/predict",
        data={"model": "Random Forest", "file": (BytesIO(bytes([97, 109, 111, 117, 110, 116, 10, 49, 53, 10, 51, 53, 10])), "new.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 404
    assert b"No saved model" in response.data


def test_saved_model_api_returns_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr("app.MODEL_DIR", tmp_path)
    app.config["TESTING"] = True
    pipeline = Pipeline([("model", DummyClassifier(strategy="prior"))])
    training = pd.DataFrame({"amount": [10, 20, 30, 40]})
    pipeline.fit(training, [0, 0, 1, 1])
    joblib.dump(pipeline, tmp_path / "random_forest.joblib")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "models": {"Random Forest": {"file": "random_forest.joblib", "threshold": 0.5}}
    }), encoding="utf-8")

    client = app.test_client()
    with client.session_transaction() as session:
        session["username"] = "test-user"
    response = client.post(
        "/api/predict",
        data={"model": "Random Forest", "file": (BytesIO(bytes([97, 109, 111, 117, 110, 116, 10, 49, 53, 10, 51, 53, 10])), "new.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.get_json()["rows"] == 2
    assert len(response.get_json()["predictions"]) == 2
