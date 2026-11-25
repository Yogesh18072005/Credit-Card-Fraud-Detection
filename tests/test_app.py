import json

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
        data={"model": "Random Forest", "file": (__import__("io").BytesIO(b""), "transactions.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 404
    assert b"No saved model" in response.data
