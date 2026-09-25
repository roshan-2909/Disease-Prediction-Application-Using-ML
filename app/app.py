"""
app.py
Flask web application for the Disease Prediction Application.
Loads the three trained models (heart, diabetes, breast cancer) and serves
prediction forms + results, and logs every prediction to the SQLite database.
"""

import os
import json
import sqlite3
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, g, redirect, url_for, flash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DB_PATH = os.path.join(ROOT_DIR, "database", "app.db")

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-in-production"

# ---------------------------------------------------------------
# Load trained models + scalers once at startup
# ---------------------------------------------------------------
heart_model = joblib.load(os.path.join(MODELS_DIR, "Heart_Disease-pred_model.sav"))
heart_scaler = joblib.load(os.path.join(MODELS_DIR, "Heart-scaler.sav"))

diabetes_model = joblib.load(os.path.join(MODELS_DIR, "Diabetes-pred_model.sav"))
diabetes_scaler = joblib.load(os.path.join(MODELS_DIR, "Diabetes-scaler.sav"))

breast_model = joblib.load(os.path.join(MODELS_DIR, "Breast_Cancer-pred_model.sav"))
breast_scaler = joblib.load(os.path.join(MODELS_DIR, "BreastCancer-scaler.sav"))

HEART_FIELDS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

DIABETES_FIELDS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

SEX_VALUES = {"male": 1.0, "female": 0.0, "other": 0.5}

# 30 features used by the breast cancer model, in training-column order
BREAST_FIELDS = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension",
]


# ---------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def log_prediction(disease_type, input_data, result, probability=None):
    db = get_db()
    db.execute(
        "INSERT INTO predictions (user_id, disease_type, input_data, result, probability) "
        "VALUES (?, ?, ?, ?, ?)",
        (None, disease_type, json.dumps(input_data), result, probability),
    )
    db.commit()


HOSPITAL_RULES = {
    "heart": {"specialty": "Cardiology", "limit": 3},
    "diabetes": {"specialty": "Endocrinology", "limit": 3},
    "breast_cancer": {"specialty": "Oncology", "limit": 3},
}


def recommend_hospitals(disease_type):
    """Return the highest-rated hospitals matching the predicted disease."""
    rule = HOSPITAL_RULES.get(disease_type)
    if rule is None:
        return []

    db = get_db()
    rows = db.execute(
        "SELECT name, specialty, city, rating FROM hospitals "
        "WHERE specialty = ? ORDER BY rating DESC, name ASC LIMIT ?",
        (rule["specialty"], rule["limit"]),
    ).fetchall()
    return [dict(r) for r in rows]


def get_sex_value(form):
    sex = form.get("sex", "").lower()
    if sex not in SEX_VALUES:
        raise ValueError("Invalid sex selection")
    return SEX_VALUES[sex]


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/heart", methods=["GET", "POST"])
def heart():
    if request.method == "POST":
        try:
            values = [get_sex_value(request.form) if f == "sex"
                      else float(request.form[f]) for f in HEART_FIELDS]
        except (ValueError, KeyError):
            flash("Please fill all fields with valid numeric values.")
            return redirect(url_for("heart"))

        x_df = pd.DataFrame([values], columns=HEART_FIELDS)
        x = heart_scaler.transform(x_df)
        pred = heart_model.predict(x)[0]
        result = "Positive" if pred == 1 else "Negative"
        input_data = dict(zip(HEART_FIELDS, values))
        log_prediction("heart", input_data, result)
        hospitals = recommend_hospitals("heart") if result == "Positive" else []
        return render_template("result.html", disease="Heart Disease", result=result,
                                hospitals=hospitals, back_url=url_for("heart"))
    return render_template("heart.html", fields=HEART_FIELDS)


@app.route("/diabetes", methods=["GET", "POST"])
def diabetes():
    if request.method == "POST":
        try:
            values = [float(request.form[f]) for f in DIABETES_FIELDS]
            sex = request.form["sex"]
            get_sex_value(request.form)
        except (ValueError, KeyError):
            flash("Please fill all fields with valid numeric values.")
            return redirect(url_for("diabetes"))

        x_df = pd.DataFrame([values], columns=DIABETES_FIELDS)
        x = diabetes_scaler.transform(x_df)
        pred = diabetes_model.predict(x)[0]
        result = "Positive" if pred == 1 else "Negative"
        input_data = dict(zip(DIABETES_FIELDS, values))
        input_data["sex"] = sex
        log_prediction("diabetes", input_data, result)
        hospitals = recommend_hospitals("diabetes") if result == "Positive" else []
        return render_template("result.html", disease="Diabetes", result=result,
                                hospitals=hospitals, back_url=url_for("diabetes"))
    return render_template("diabetes.html", fields=DIABETES_FIELDS)


@app.route("/breast-cancer", methods=["GET", "POST"])
def breast_cancer():
    if request.method == "POST":
        try:
            values = [float(request.form[f]) for f in BREAST_FIELDS]
            sex = request.form["sex"]
            get_sex_value(request.form)
        except (ValueError, KeyError):
            flash("Please fill all fields with valid numeric values.")
            return redirect(url_for("breast_cancer"))

        x_df = pd.DataFrame([values], columns=BREAST_FIELDS)
        x = breast_scaler.transform(x_df)
        pred = breast_model.predict(x)[0]
        result = "Positive" if pred == 1 else "Negative"
        input_data = dict(zip(BREAST_FIELDS, values))
        input_data["sex"] = sex
        log_prediction("breast_cancer", input_data, result)
        hospitals = recommend_hospitals("breast_cancer") if result == "Positive" else []
        return render_template("result.html", disease="Breast Cancer", result=result,
                                hospitals=hospitals, back_url=url_for("breast_cancer"))
    return render_template("breast_cancer.html", fields=BREAST_FIELDS)


@app.route("/history")
def history():
    db = get_db()
    rows = db.execute(
        "SELECT id, disease_type, result, created_at FROM predictions ORDER BY id DESC LIMIT 50"
    ).fetchall()
    return render_template("history.html", records=rows)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
