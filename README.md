# Vitalis — AI-Based Healthcare Risk Prediction & Patient Risk Analysis System

An end-to-end machine learning web application that estimates a patient's
overall health risk (Low / Medium / High), plus disease-specific risk scores
for **diabetes** and **heart disease**, from vitals and lifestyle inputs. Built
with scikit-learn, Flask, SQLite, and a custom dashboard frontend.

> ⚠️ **Disclaimer**: This project uses a **synthetically generated** dataset
> for educational/demonstration purposes. It is **not** a medical device and
> must never be used for real clinical decision-making.

---

## Features

- **Synthetic patient data generator** — 5,000 realistic records with
  correlated vitals (age, BMI, blood pressure, glucose, cholesterol, etc.)
- **Three trained ML models**:
  - Random Forest classifier → overall risk level (Low/Medium/High), 85% accuracy
  - Gradient Boosting regressor → Diabetes risk % (R² = 0.995)
  - Gradient Boosting regressor → Heart disease risk % (R² = 0.996)
- **Rules-based recommendation engine** — plain-language, prioritized
  clinical suggestions based on out-of-range vitals
- **Flask REST API** — `/api/predict`, `/api/patients`, `/api/stats`
- **SQLite persistence** — every assessment is logged and browsable
- **Interactive dashboard** — risk gauge, probability breakdown, sub-risk
  bars, patient records table, and a model methodology page
- **EDA script** for dataset exploration and correlation analysis

---

## Project Structure

```
healthcare-risk-system/
├── app.py                     # Flask backend + REST API
├── requirements.txt
├── patients.db                 # SQLite DB (created on first run)
│
├── data/
│   ├── generate_data.py        # Synthetic dataset generator
│   └── patient_data.csv        # Generated dataset (5,000 rows)
│
├── model/
│   ├── train_model.py          # Trains & saves all 3 models
│   ├── metrics_report.txt      # Accuracy / MAE / R² / feature importance
│   └── artifacts/              # Saved .pkl models, scaler, encoders
│
├── notebooks/
│   └── eda.py                  # Exploratory data analysis script
│
├── templates/
│   └── index.html              # Dashboard UI
│
└── static/
    ├── css/style.css
    └── js/script.js
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset
```bash
python data/generate_data.py
```
This creates `data/patient_data.csv` with 5,000 synthetic patient records.

### 3. Train the models
```bash
python model/train_model.py
```
This trains all three models and saves them to `model/artifacts/`, along
with a `model/metrics_report.txt` summary.

### 4. Run the web app
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

### 5. (Optional) Explore the data
```bash
python notebooks/eda.py
```

---

## How It Works

### 1. Data
Synthetic patient records are generated with intentional, clinically-inspired
correlations — e.g. BMI rises loosely with age, blood pressure rises with age
and BMI, glucose rises with BMI, etc. A composite `risk_score` (0–100) is
computed from weighted risk factors, then bucketed into `risk_level`
(Low/Medium/High). Two additional continuous targets — `diabetes_risk` and
`heart_disease_risk` — are computed with independent logistic-style formulas
plus noise, to emulate realistic (not perfectly correlated) sub-conditions.

### 2. Feature engineering
14 features are used: age, gender, BMI, systolic/diastolic BP, glucose,
cholesterol, heart rate, smoking, alcohol use, physical activity level,
family history, sleep hours, and stress level. Categorical fields are
label-encoded and all features are standardized with `StandardScaler`
before being passed to the models.

### 3. Models
| Model | Algorithm | Target | Metric |
|---|---|---|---|
| Risk Level | RandomForestClassifier (300 trees) | Low/Medium/High | 85% accuracy |
| Diabetes Risk | GradientBoostingRegressor | 0–100% | R² = 0.995 |
| Heart Disease Risk | GradientBoostingRegressor | 0–100% | R² = 0.996 |

The Random Forest additionally exposes class probabilities, which are
blended into a smooth 0–100 **composite risk score** for the gauge shown on
the dashboard.

### 4. Recommendations
A separate rules layer inspects each raw vital (not just the model output)
and produces specific, human-readable guidance — e.g. flagging hypertension,
pre-diabetic glucose, high cholesterol, smoking, poor sleep, etc. — so the
output is explainable rather than a single opaque number.

### 5. API

**POST `/api/predict`**
```json
{
  "name": "Anita Sharma",
  "age": 52, "gender": "Female", "bmi": 27.4,
  "systolic_bp": 132, "diastolic_bp": 85,
  "glucose": 118, "cholesterol": 205, "heart_rate": 78,
  "smoking": 0, "alcohol_consumption": 0,
  "physical_activity": "Moderate", "family_history": 1,
  "sleep_hours": 6.5, "stress_level": 6
}
```
Returns risk level, composite score, class probabilities, diabetes/heart
risk percentages, and a list of recommendations. Also logs the record to
SQLite.

**GET `/api/patients`** — last 100 logged assessments.
**GET `/api/stats`** — aggregate counts by risk level.
**DELETE `/api/patients/<id>`** — remove a record.

---

## Retraining on Your Own Data

Replace `data/patient_data.csv` with your own dataset (same column names —
see `FEATURE_COLUMNS` in `model/train_model.py`), then re-run:
```bash
python model/train_model.py
```
The Flask app automatically loads whatever is in `model/artifacts/` on
startup, so no code changes are needed downstream.

---

## Tech Stack

- **ML**: scikit-learn (RandomForest, GradientBoosting), pandas, numpy
- **Backend**: Flask, SQLite
- **Frontend**: vanilla HTML/CSS/JS (no framework dependency)
- **Persistence**: joblib for model artifacts

---

## Limitations & Next Steps

- Trained on synthetic, not real clinical data — do not deploy for actual
  patient care without validated data and clinical oversight.
- No authentication/authorization on the API — add before any multi-user
  or production deployment.
- Could be extended with: SHAP-based per-prediction explainability, more
  disease-specific models (e.g. stroke, kidney disease), time-series
  tracking of a patient's risk over multiple visits, and export to PDF
  reports.
