# Credit Card Fraud Detection

A Flask web application for exploring credit-card transaction datasets and comparing supervised and anomaly-detection approaches.

## What it does

- Provides account registration and login backed by MongoDB, with bcrypt password hashing.
- Accepts CSV uploads (up to 16 MB) through an authenticated route.
- Validates the target labels instead of silently inventing fraud labels.
- Handles numeric and categorical features, including missing feature values.
- Splits data into stratified training and held-out test sets before fitting models.
- Compares Isolation Forest, Linear SVM, Logistic Regression, and Random Forest.
- Tunes supervised classification thresholds on a separate validation split before final held-out testing.
- Calculates permutation feature importance for the best supervised model by PR-AUC.
- Saves trained pipelines and thresholds locally with joblib and displays interactive charts.
- Reports accuracy, balanced accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC, confusion matrices, and classification reports.
- Includes automated tests and a GitHub Actions workflow.

> **Important:** This is an educational demonstration, not a payment decision system. Metrics depend on the uploaded dataset and class balance. Do not use the output to approve or decline real transactions.

## Technology

- Python, Flask, Jinja templates
- pandas, NumPy, scikit-learn
- MongoDB / PyMongo
- bcrypt
- pytest and GitHub Actions

## Requirements

- Python 3.11 or newer
- MongoDB connection for registration, normal login, and profile storage
- A CSV dataset with a binary target column

## Run locally (Windows PowerShell)

```powershell
git clone https://github.com/Yogesh18072005/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` with your own values, then start the app:

```powershell
python app.py
```

Open http://127.0.0.1:5000.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `FLASK_SECRET_KEY` | Secret used to sign session cookies. Set a long random value. |
| `MONGO_URI` | MongoDB connection string for user accounts. |
| `COOKIE_SECURE` | Set to `true` when served over HTTPS. |
| `DEMO_LOGIN_ENABLED` | Keep `false` by default; enable only for a controlled demonstration. |
| `DEMO_USERNAME` | Optional demo account username. |
| `DEMO_PASSWORD` | Strong password for the optional demo account. |

Never commit your real `.env` file or credentials.

## Dataset format

Upload a CSV containing feature columns and one binary target column. The app recognizes `Class`, `class`, `fraudulent`, `label`, or `is_fraud` by default. You can use:

- `0` = legitimate transaction
- `1` = fraudulent transaction

Both labels must be present. The target must not be inferred from missing data. Keep transaction IDs and personally identifying information out of demo uploads.

Example structure:

| amount | merchant_type | hour | Class |
| ---: | --- | ---: | ---: |
| 24.50 | grocery | 13 | 0 |
| 890.00 | electronics | 2 | 1 |

The example rows above are illustrative only, not a real training dataset.

## Evaluation methodology

The pipeline uses a stratified train/test split with a fixed random seed. Preprocessing (imputation, scaling, and categorical encoding) is fitted within each model pipeline using the training data only. Reported metrics are calculated on the held-out test set.

- **Isolation Forest** learns an anomaly boundary from the training features; its outlier predictions are mapped to the fraud label.
- **Linear SVM** provides a scalable margin-based classifier.
- **Logistic Regression** provides a linear probabilistic classification baseline.
- **Random Forest** adds a non-linear ensemble baseline with class-balanced bootstrap sampling.

Accuracy alone can be misleading for highly imbalanced fraud data. Compare balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, specificity, and the confusion matrix as well. The results page presents a side-by-side model comparison and individual confusion matrices. The current comparison uses a default decision threshold and is intended for learning, not production calibration.

## Docker deployment

The repository includes Docker and Docker Compose configuration for the Flask application and MongoDB.

```powershell
Copy-Item .env.example .env
# Edit .env and set FLASK_SECRET_KEY to a long random value.
docker compose --env-file .env up --build
```

Open http://localhost:5000. Compose persists MongoDB data and trained model artifacts in named volumes. For production, use HTTPS and set `COOKIE_SECURE=true`.

## Saved models and explainability

After analysis, the application saves the latest fitted pipelines under `MODEL_DIR` (default: `models/`) as joblib files. A `manifest.json` records model filenames, selected thresholds, PR-AUC values, and the creation time. Only load model files created by this application: joblib/pickle files can execute code and must never be accepted from untrusted uploads.

Permutation importance is calculated on a capped sample of held-out rows for the best supervised model by PR-AUC. It is a global explanation aid, not a causal explanation. Supervised thresholds are selected on validation data to maximize F1; Isolation Forest keeps its anomaly boundary threshold.

## Tests

```powershell
pytest -q
python -m compileall -q .
```

GitHub Actions runs these checks on pushes and pull requests to `main`.

## Project structure

```text
.
├── app.py
├── fraud_detection.py
├── requirements.txt
├── templates/
├── static/
├── tests/
│   └── test_fraud_detection.py
└── .github/
    └── workflows/
        └── python-ci.yml
```

## Limitations and next steps

- The application saves the latest trained pipelines locally; it does not yet provide a versioned model registry or a separate production inference API.
- MongoDB is required for normal account operations.
- Add audit logging, drift monitoring, authenticated inference APIs, and a versioned model registry before considering production use.
