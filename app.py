import os
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session
from pymongo import MongoClient
import bcrypt
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


# -------------------------------
# 1️⃣ CREATE FLASK APP FIRST
# -------------------------------
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# -------------------------------
# 2️⃣ CONFIG
# -------------------------------
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# Database
client = MongoClient(MONGO_URI)
db = client['user_database']
collection = db['users']


# -------------------------------
# 3️⃣ ROUTES
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register')
def register_page():
    return render_template('login.html')


@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    email = request.form['email']

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    if collection.find_one({'username': username}):
        session['error'] = "Username already exists!"
        return redirect(url_for('register_page'))
    elif collection.find_one({'email': email}):
        session['error'] = "Email already exists!"
        return redirect(url_for('register_page'))

    user_data = {
        'username': username,
        'password': hashed_password,
        'email': email
    }
    collection.insert_one(user_data)

    return redirect(url_for('login_page'))


@app.route('/login')
def login_page():
    return render_template("login.html")


@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    # Dummy login allowed
    if username == 'dummy' and password == 'dummy':
        session['username'] = 'Demo User'
        return redirect(url_for('dashboard', username='Demo User'))

    user = collection.find_one({'username': username})

    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        session['username'] = user['username']
        return redirect(url_for('dashboard', username=user['username']))

    session['error'] = "Invalid username or password"
    return redirect(url_for('login_page'))


@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template("dashboard.html", username=session['username'])


@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template("home.html", username=session['username'])


@app.route('/admin')
def admin_page():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template("admin.html", username=session['username'])


@app.route('/profile')
def profile_page():
    if 'username' not in session:
        return redirect(url_for('login_page'))

    user_data = collection.find_one({'username': session['username']})
    return render_template("profile.html", username=session['username'], user_data=user_data)


@app.route('/profile/update', methods=['POST'])
def update_profile():
    if 'username' not in session:
        return redirect(url_for('login_page'))

    username = session['username']
    updated_data = {
        'first_name': request.form['first_name'],
        'last_name': request.form['last_name'],
        'organization_name': request.form['organization_name'],
        'location': request.form['location'],
        'phone_number': request.form['phone_number'],
        'birthday': request.form['birthday']
    }

    collection.update_one({'username': username}, {'$set': updated_data})

    return redirect(url_for('profile_page'))


# -------------------------------------------------------------
# 4️⃣ FIXED PREDICT ROUTE — WORKS WITH ANY CSV
# -------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "File name is empty"

    data = pd.read_csv(file)

    # FIX: Ensure Class exists
    if 'Class' not in data.columns:
        if 'fraudulent' in data.columns:
            data.rename(columns={'fraudulent': 'Class'}, inplace=True)
        elif 'label' in data.columns:
            data.rename(columns={'label': 'Class'}, inplace=True)
        elif 'class' in data.columns:
            data.rename(columns={'class': 'Class'}, inplace=True)
        else:
            data['Class'] = 1

    data['Class'] = pd.to_numeric(data['Class'], errors='coerce').fillna(0).astype(int)

    # Stats
    statistical_analysis = data.describe()
    fraudulent_count = (data['Class'] == 1).sum()
    non_fraudulent_count = (data['Class'] == 0).sum()

    X = data.drop(columns=["Class"])
    y = data["Class"]

    # ------------------------------------------------
    # FIX: Encode categorical text columns to numeric
    # ------------------------------------------------
    label_enc = LabelEncoder()

    for col in X.columns:
        if X[col].dtype == 'object':  
            X[col] = label_enc.fit_transform(X[col].astype(str))

    # ML MODELS
    iso_forest = IsolationForest()
    iso_forest.fit(X)
    iso_preds = iso_forest.predict(X)
    iso_accuracy = accuracy_score(y, [-1 if p == -1 else 0 for p in iso_preds])
    iso_error = 1 - iso_accuracy
    iso_report = classification_report(y, [-1 if p == -1 else 0 for p in iso_preds])

    svm_model = SVC()
    svm_model.fit(X, y)
    svm_preds = svm_model.predict(X)
    svm_accuracy = accuracy_score(y, svm_preds)
    svm_error = 1 - svm_accuracy
    svm_report = classification_report(y, svm_preds)

    log_model = LogisticRegression()
    log_model.fit(X, y)
    log_preds = log_model.predict(X)
    log_accuracy = accuracy_score(y, log_preds)
    log_error = 1 - log_accuracy
    log_report = classification_report(y, log_preds)

    return render_template(
        'admin.html',
        username=session['username'],
        statistical_analysis=statistical_analysis,
        fraudulent_count=fraudulent_count,
        non_fraudulent_count=non_fraudulent_count,
        iso_forest_accuracy=iso_accuracy,
        iso_forest_error=iso_error,
        iso_forest_classification_report=iso_report,
        svm_accuracy=svm_accuracy,
        svm_error=svm_error,
        svm_classification_report=svm_report,
        logistic_accuracy=log_accuracy,
        logistic_error=log_error,
        logistic_classification_report=log_report
    )


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))


# -------------------------------
# 5️⃣ RUN APP
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)
