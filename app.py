import json
import os
from datetime import datetime, timezone
from pathlib import Path
from functools import wraps

import bcrypt
import joblib
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from werkzeug.utils import secure_filename

from fraud_detection import evaluate_fraud_dataset

load_dotenv()

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.getenv("FLASK_SECRET_KEY") or os.urandom(32),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("COOKIE_SECURE", "false").lower() == "true",
)

MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000) if MONGO_URI else None
users = mongo_client["user_database"]["users"] if mongo_client is not None else None
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MODEL_FILES = {
    "Isolation Forest": "isolation_forest.joblib",
    "Linear SVM": "linear_svm.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
}
DEMO_LOGIN_ENABLED = os.getenv("DEMO_LOGIN_ENABLED", "false").lower() == "true"


def save_model_artifacts(analysis):
    """Save only pipelines trained by this application; never load uploaded pickle files."""
    pipelines = analysis.pop("pipelines")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_column": analysis["target_column"],
        "models": {},
    }
    for model_name, pipeline in pipelines.items():
        slug = model_name.lower().replace(" ", "_")
        destination = MODEL_DIR / f"{slug}.joblib"
        temporary = MODEL_DIR / f".{slug}.tmp"
        joblib.dump(pipeline, temporary)
        os.replace(temporary, destination)
        manifest["models"][model_name] = {
            "file": destination.name,
            "threshold": analysis["models"][model_name]["threshold"],
            "pr_auc": analysis["models"][model_name]["pr_auc"],
        }
    temporary_manifest = MODEL_DIR / ".manifest.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, MODEL_DIR / "manifest.json")
    return manifest


def get_users_collection():
    if users is None:
        raise RuntimeError("MongoDB is not configured. Set MONGO_URI in your environment.")
    return users


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login_page"))
        return view(*args, **kwargs)
    return wrapped


@app.errorhandler(413)
def file_too_large(_error):
    return "The uploaded file exceeds the 16 MB limit.", 413


@app.route("/")
def index():
    return render_template("index.html", demo_login_enabled=DEMO_LOGIN_ENABLED)


@app.route("/register", methods=["GET"])
def register_page():
    return render_template("login.html")


@app.route("/register", methods=["POST"])
def register():
    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    if not username or not email or len(password) < 8:
        session["error"] = "Enter a username, valid email, and password with at least 8 characters."
        return redirect(url_for("register_page"))
    try:
        collection = get_users_collection()
        if collection.find_one({"$or": [{"username": username}, {"email": email}]}):
            session["error"] = "That username or email is already registered."
            return redirect(url_for("register_page"))
        collection.insert_one({
            "username": username,
            "email": email,
            "password": bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()),
        })
    except (RuntimeError, PyMongoError):
        app.logger.exception("Registration could not access MongoDB")
        return "Account registration is temporarily unavailable. Configure MongoDB and try again.", 503
    return redirect(url_for("login_page"))


@app.route("/login", methods=["GET"])
def login_page():
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    if (DEMO_LOGIN_ENABLED and os.getenv("DEMO_PASSWORD") and username == os.getenv("DEMO_USERNAME", "demo") and password == os.getenv("DEMO_PASSWORD")):
        session.clear()
        session.update(username="Demo User", demo=True)
        return redirect(url_for("dashboard"))
    try:
        user = get_users_collection().find_one({"username": username})
    except (RuntimeError, PyMongoError):
        app.logger.exception("Login could not access MongoDB")
        return "Login is temporarily unavailable. Configure MongoDB and try again.", 503
    if user and bcrypt.checkpw(password.encode("utf-8"), user["password"]):
        session.clear()
        session["username"] = user["username"]
        return redirect(url_for("dashboard"))
    session["error"] = "Invalid username or password."
    return redirect(url_for("login_page"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", username=session["username"])


@app.route("/home")
@login_required
def home():
    return render_template("home.html", username=session["username"])


@app.route("/admin")
@login_required
def admin_page():
    return render_template("admin.html", username=session["username"])


@app.route("/profile")
@login_required
def profile_page():
    user_data = None
    if not session.get("demo"):
        try:
            user_data = get_users_collection().find_one({"username": session["username"]})
        except (RuntimeError, PyMongoError):
            app.logger.exception("Profile could not access MongoDB")
            return "Profile service is temporarily unavailable.", 503
    return render_template("profile.html", username=session["username"], user_data=user_data)


@app.route("/profile/update", methods=["POST"])
@login_required
def update_profile():
    if session.get("demo"):
        return "Demo profiles are read-only.", 403
    fields = ("first_name", "last_name", "organization_name", "location", "phone_number", "birthday")
    updated_data = {field: request.form.get(field, "").strip() for field in fields}
    try:
        get_users_collection().update_one({"username": session["username"]}, {"$set": updated_data})
    except (RuntimeError, PyMongoError):
        app.logger.exception("Profile update could not access MongoDB")
        return "Profile service is temporarily unavailable.", 503
    return redirect(url_for("profile_page"))


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return "Choose a CSV file to upload.", 400
    safe_name = secure_filename(uploaded_file.filename)
    if os.path.splitext(safe_name)[1].lower() != ".csv":
        return "Only .csv files are supported.", 400
    try:
        data = pd.read_csv(uploaded_file, low_memory=False)
        analysis = evaluate_fraud_dataset(data)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return "The uploaded file is not a readable CSV.", 400
    except ValueError as error:
        return str(error), 400
    except Exception:
        app.logger.exception("Fraud analysis failed")
        return "The dataset could not be analyzed. Check its columns and values.", 422
    models = analysis["models"]
    try:
        model_manifest = save_model_artifacts(analysis)
    except OSError:
        app.logger.exception("Could not persist trained model artifacts")
        model_manifest = None
    chart_data = {
        "labels": list(models.keys()),
        "precision": [models[name]["precision"] for name in models],
        "recall": [models[name]["recall"] for name in models],
        "f1": [models[name]["f1"] for name in models],
        "roc_auc": [models[name]["roc_auc"] for name in models],
        "pr_auc": [models[name]["pr_auc"] for name in models],
    }
    return render_template(
        "admin.html",
        username=session["username"],
        statistical_analysis=analysis["statistical_analysis"],
        fraudulent_count=analysis["fraudulent_count"],
        non_fraudulent_count=analysis["non_fraudulent_count"],
        iso_forest_accuracy=models["Isolation Forest"]["accuracy"],
        iso_forest_error=models["Isolation Forest"]["error_rate"],
        iso_forest_classification_report=models["Isolation Forest"]["classification_report"],
        svm_accuracy=models["Linear SVM"]["accuracy"],
        svm_error=models["Linear SVM"]["error_rate"],
        svm_classification_report=models["Linear SVM"]["classification_report"],
        logistic_accuracy=models["Logistic Regression"]["accuracy"],
        logistic_error=models["Logistic Regression"]["error_rate"],
        logistic_classification_report=models["Logistic Regression"]["classification_report"],
        model_metrics=models,
        analysis=analysis,
        chart_data=chart_data,
        feature_importance=analysis["feature_importance"],
        explanation_model=analysis["explanation_model"],
        model_manifest=model_manifest,
    )


@app.route("/api/predict", methods=["POST"])
@login_required
def predict_with_saved_model():
    """Run inference with a model artifact generated by this application."""
    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify(error="Choose a CSV file."), 400
    if os.path.splitext(secure_filename(uploaded_file.filename))[1].lower() != ".csv":
        return jsonify(error="Only .csv files are supported."), 400

    model_name = request.form.get("model", "Random Forest")
    filename = MODEL_FILES.get(model_name)
    if filename is None:
        return jsonify(error="Choose a supported model."), 400
    manifest_path = MODEL_DIR / "manifest.json"
    model_path = MODEL_DIR / filename
    if not manifest_path.is_file() or not model_path.is_file():
        return jsonify(error="No saved model is available. Upload a labelled training CSV first."), 404

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("models", {}).get(model_name, {}).get("file") != filename:
            return jsonify(error="The selected model is not registered in the local manifest."), 404
        pipeline = joblib.load(model_path)
        data = pd.read_csv(uploaded_file, low_memory=False)
        expected_columns = list(pipeline.feature_names_in_)
        missing = sorted(set(expected_columns) - set(data.columns))
        extra = sorted(set(data.columns) - set(expected_columns))
        if missing or extra:
            return jsonify(error="CSV columns must match the training features.", missing=missing, extra=extra), 400
        data = data[expected_columns]
        if model_name == "Isolation Forest":
            scores = -pipeline.decision_function(data)
        elif hasattr(pipeline, "predict_proba"):
            scores = pipeline.predict_proba(data)[:, 1]
        else:
            scores = pipeline.decision_function(data)
        threshold = float(manifest["models"][model_name]["threshold"])
        predictions = (scores >= threshold).astype(int)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return jsonify(error="The uploaded file is not a readable CSV."), 400
    except (KeyError, OSError, ValueError, TypeError):
        app.logger.exception("Saved model inference failed")
        return jsonify(error="The saved model could not process this dataset."), 422

    return jsonify(
        model=model_name,
        threshold=threshold,
        predictions=predictions.tolist(),
        scores=[float(score) for score in scores],
        fraud_predictions=int(predictions.sum()),
        rows=int(len(data)),
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
