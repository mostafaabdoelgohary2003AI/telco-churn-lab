import os
os.environ["SQLITE_PATH"] = "results/test_pred.db"
from fastapi.testclient import TestClient
from src.api import app
c = TestClient(app)

def test_predict_orders_risk():
    hi = c.post("/api/predict", json={"tenure": 1, "MonthlyCharges": 95}).json()
    lo = c.post("/api/predict", json={"tenure": 60, "MonthlyCharges": 40, "Contract": "Two year", "InternetService": "DSL", "PaymentMethod": "Bank transfer (automatic)"}).json()
    assert hi["probability"] > lo["probability"] + 0.3

def test_metrics_and_history():
    m = c.get("/api/metrics").json()
    assert m["final"]["test_accuracy_at_threshold"] > m["final"]["majority_class_baseline_accuracy"]
    assert len(c.get("/api/history").json()) >= 1
    assert c.get("/").status_code == 200

def test_no_leakage_columns():
    from src.prep import NUM, CATEGORICAL
    assert "Churn" not in NUM + CATEGORICAL and "customerID" not in CATEGORICAL
