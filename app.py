"""
AI-Based Healthcare Risk Prediction & Patient Risk Analysis System
---------------------------------------------------------------------
Flask backend serving:
  - A dashboard UI for entering patient vitals and viewing risk analysis
  - A REST API (/api/predict) for programmatic predictions
  - A patient records API (/api/patients) backed by SQLite for history tracking

Run with:  python app.py
Then open: http://127.0.0.1:5000
"""

import os
import json
import sqlite3
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, g

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model", "artifacts")
DB_PATH = os.path.join(BASE_DIR, "patients.db")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load trained ML artifacts once at startup
# ---------------------------------------------------------------------------
risk_model = joblib.load(os.path.join(ARTIFACTS_DIR, "risk_level_model.pkl"))
diabetes_model = joblib.load(os.path.join(ARTIFACTS_DIR, "diabetes_model.pkl"))
heart_model = joblib.load(os.path.join(ARTIFACTS_DIR, "heart_model.pkl"))
scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
le_gender = joblib.load(os.path.join(ARTIFACTS_DIR, "le_gender.pkl"))
le_activity = joblib.load(os.path.join(ARTIFACTS_DIR, "le_activity.pkl"))
le_risk = joblib.load(os.path.join(ARTIFACTS_DIR, "le_risk.pkl"))

with open(os.path.join(ARTIFACTS_DIR, "feature_columns.json")) as f:
    FEATURE_COLUMNS = json.load(f)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
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


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            gender TEXT,
            bmi REAL,
            systolic_bp INTEGER,
            diastolic_bp INTEGER,
            glucose INTEGER,
            cholesterol INTEGER,
            heart_rate INTEGER,
            smoking INTEGER,
            alcohol_consumption INTEGER,
            physical_activity TEXT,
            family_history INTEGER,
            sleep_hours REAL,
            stress_level INTEGER,
            risk_level TEXT,
            risk_score REAL,
            diabetes_risk REAL,
            heart_disease_risk REAL,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Core prediction logic
# ---------------------------------------------------------------------------
def build_feature_vector(payload):
    gender_enc = le_gender.transform([payload["gender"]])[0]
    activity_enc = le_activity.transform([payload["physical_activity"]])[0]

    row = {
        "age": payload["age"],
        "gender_enc": gender_enc,
        "bmi": payload["bmi"],
        "systolic_bp": payload["systolic_bp"],
        "diastolic_bp": payload["diastolic_bp"],
        "glucose": payload["glucose"],
        "cholesterol": payload["cholesterol"],
        "heart_rate": payload["heart_rate"],
        "smoking": payload["smoking"],
        "alcohol_consumption": payload["alcohol_consumption"],
        "activity_enc": activity_enc,
        "family_history": payload["family_history"],
        "sleep_hours": payload["sleep_hours"],
        "stress_level": payload["stress_level"],
    }
    ordered = [row[c] for c in FEATURE_COLUMNS]
    return pd.DataFrame([ordered], columns=FEATURE_COLUMNS)


def generate_recommendations(payload, risk_level, diabetes_risk, heart_risk):
    recs = []
    if payload["bmi"] >= 30:
        recs.append("BMI indicates obesity — consult a nutritionist for a structured weight management plan.")
    elif payload["bmi"] >= 25:
        recs.append("BMI is in the overweight range — moderate calorie control and regular exercise are advised.")

    if payload["systolic_bp"] >= 140 or payload["diastolic_bp"] >= 90:
        recs.append("Blood pressure readings suggest hypertension — schedule a follow-up with a physician.")
    elif payload["systolic_bp"] >= 130:
        recs.append("Blood pressure is elevated — monitor regularly and reduce sodium intake.")

    if payload["glucose"] >= 126:
        recs.append("Fasting glucose is in the diabetic range — recommend an HbA1c test and endocrinologist review.")
    elif payload["glucose"] >= 100:
        recs.append("Glucose levels are pre-diabetic — dietary changes and regular monitoring recommended.")

    if payload["cholesterol"] >= 240:
        recs.append("Total cholesterol is high — consider a lipid panel and dietary fat reduction.")

    if payload["smoking"] == 1:
        recs.append("Smoking cessation would significantly reduce cardiovascular and cancer risk.")

    if payload["physical_activity"] == "Low":
        recs.append("Increase physical activity to at least 150 minutes of moderate exercise per week.")

    if payload["sleep_hours"] < 6:
        recs.append("Sleep duration is low — aim for 7-9 hours per night to support metabolic health.")

    if payload["stress_level"] >= 7:
        recs.append("High stress levels reported — consider stress-management techniques or counseling.")

    if diabetes_risk >= 60:
        recs.append(f"Diabetes risk score is high ({diabetes_risk:.1f}%) — prioritize a diabetes screening.")

    if heart_risk >= 60:
        recs.append(f"Cardiovascular risk score is high ({heart_risk:.1f}%) — prioritize a cardiology consultation.")

    if not recs:
        recs.append("No major risk factors detected. Maintain current healthy lifestyle habits.")

    return recs


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)

        required = [
            "age", "gender", "bmi", "systolic_bp", "diastolic_bp", "glucose",
            "cholesterol", "heart_rate", "smoking", "alcohol_consumption",
            "physical_activity", "family_history", "sleep_hours", "stress_level",
        ]
        missing = [k for k in required if k not in payload]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        X = build_feature_vector(payload)
        X_scaled = scaler.transform(X)

        risk_pred_enc = risk_model.predict(X_scaled)[0]
        risk_level = le_risk.inverse_transform([risk_pred_enc])[0]
        risk_proba = risk_model.predict_proba(X_scaled)[0]
        proba_map = {
            cls: round(float(p) * 100, 1)
            for cls, p in zip(le_risk.classes_, risk_proba)
        }

        diabetes_risk = float(np.clip(diabetes_model.predict(X_scaled)[0], 0, 100))
        heart_risk = float(np.clip(heart_model.predict(X_scaled)[0], 0, 100))

        # Composite score derived from class probabilities for a smooth 0-100 gauge
        risk_score = round(
            proba_map.get("Low", 0) * 0.15
            + proba_map.get("Medium", 0) * 0.5
            + proba_map.get("High", 0) * 0.95,
            1,
        )

        recommendations = generate_recommendations(payload, risk_level, diabetes_risk, heart_risk)

        result = {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_probabilities": proba_map,
            "diabetes_risk": round(diabetes_risk, 1),
            "heart_disease_risk": round(heart_risk, 1),
            "recommendations": recommendations,
        }

        # Persist to DB
        db = get_db()
        db.execute(
            """INSERT INTO patients
            (name, age, gender, bmi, systolic_bp, diastolic_bp, glucose, cholesterol,
             heart_rate, smoking, alcohol_consumption, physical_activity, family_history,
             sleep_hours, stress_level, risk_level, risk_score, diabetes_risk,
             heart_disease_risk, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                payload.get("name", "Unnamed Patient"),
                payload["age"], payload["gender"], payload["bmi"],
                payload["systolic_bp"], payload["diastolic_bp"], payload["glucose"],
                payload["cholesterol"], payload["heart_rate"], payload["smoking"],
                payload["alcohol_consumption"], payload["physical_activity"],
                payload["family_history"], payload["sleep_hours"], payload["stress_level"],
                risk_level, risk_score, diabetes_risk, heart_risk,
                datetime.now().isoformat(),
            ),
        )
        db.commit()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/patients", methods=["GET"])
def list_patients():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM patients ORDER BY id DESC LIMIT 100"
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/patients/<int:patient_id>", methods=["DELETE"])
def delete_patient(patient_id):
    db = get_db()
    db.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
    db.commit()
    return jsonify({"status": "deleted"})


@app.route("/api/stats", methods=["GET"])
def stats():
    db = get_db()
    rows = db.execute("SELECT risk_level FROM patients").fetchall()
    total = len(rows)
    counts = {"Low": 0, "Medium": 0, "High": 0}
    for r in rows:
        counts[r["risk_level"]] = counts.get(r["risk_level"], 0) + 1
    return jsonify({"total": total, "counts": counts})


if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)
