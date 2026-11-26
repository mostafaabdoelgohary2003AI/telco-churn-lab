"""Telco Churn Lab: predict one customer, see the model leaderboard and the honest ceiling."""
import json, os, sqlite3
import joblib, pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .prep import engineer, NUM, CATEGORICAL

app = FastAPI(title="Telco Churn Lab")
app.mount("/static", StaticFiles(directory="static"), name="static")
_m = None
DB = os.environ.get("SQLITE_PATH", "results/predictions.db")

def bundle():
    global _m
    if _m is None: _m = joblib.load("results/telco_model.joblib")
    return _m

def db():
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY, ts TEXT DEFAULT CURRENT_TIMESTAMP, prob REAL, label TEXT)")
    return c

class Customer(BaseModel):
    gender: str = "Female"; SeniorCitizen: int = 0; Partner: str = "No"; Dependents: str = "No"
    tenure: int = 2; PhoneService: str = "Yes"; MultipleLines: str = "No"; InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"; OnlineBackup: str = "No"; DeviceProtection: str = "No"; TechSupport: str = "No"
    StreamingTV: str = "No"; StreamingMovies: str = "No"; Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"; PaymentMethod: str = "Electronic check"; MonthlyCharges: float = 85.0

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/")
def index(): return HTMLResponse(open("static/index.html", encoding="utf-8").read())

@app.get("/api/metrics")
def metrics():
    return {"cv": json.load(open("results/cv_results.json")), "final": json.load(open("results/final_metrics.json"))}

@app.post("/api/predict")
def predict(c: Customer):
    d = c.model_dump(); d["TotalCharges"] = d["MonthlyCharges"] * max(d["tenure"], 1)
    X = engineer(pd.DataFrame([d]))[NUM + CATEGORICAL]
    b = bundle(); p = float(b["model"].predict_proba(X)[0, 1])
    label = "likely to leave" if p > b["threshold"] else "likely to stay"
    with db() as con: con.execute("INSERT INTO predictions(prob,label) VALUES(?,?)", (p, label))
    why = []
    if c.Contract == "Month-to-month": why.append("month-to-month contract (no commitment)")
    if c.tenure <= 12: why.append("less than a year as a customer")
    if c.InternetService == "Fiber optic" and c.OnlineSecurity == "No": why.append("fiber internet without online security")
    if c.PaymentMethod == "Electronic check": why.append("pays by electronic check")
    return {"probability": round(p, 3), "label": label, "common_risk_signals": why}

@app.get("/api/history")
def history():
    with db() as con: rows = con.execute("SELECT ts, prob, label FROM predictions ORDER BY id DESC LIMIT 20").fetchall()
    return [{"ts": r[0], "prob": r[1], "label": r[2]} for r in rows]
