# Credit Card Fraud Detection

A Flask web application for exploring credit-card transaction datasets and comparing supervised and anomaly-detection approaches.

## What it does

- Provides account registration and login backed by MongoDB, with bcrypt password hashing.
- Accepts CSV uploads (up to 16 MB) through an authenticated route.
- Validates the target labels instead of silently inventing fraud labels.
- Handles numeric and categorical features, including missing feature values.
- Splits data into stratified training and held-out test sets before fitting models.
- Compares Isolation Forest, Linear SVM, Logistic Regression, and Random Forest.
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

- The application currently trains models when a CSV is uploaded; it does not provide a persistent model registry or production inference service.
- MongoDB is required for normal account operations.
- Add dataset-specific threshold tuning, model persistence, audit logging, and deployment monitoring before considering any production use.
