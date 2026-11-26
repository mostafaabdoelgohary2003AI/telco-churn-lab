"""IBM Telco churn: loading + leakage-free feature engineering.
Only pre-churn attributes are used; customerID is dropped."""
import numpy as np
import pandas as pd

CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
               "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
               "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
               "PaperlessBilling", "PaymentMethod"]
NUM = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend", "num_services",
       "tenure_years", "charge_per_service", "senior_no_support", "fiber_no_security",
       "month_to_month_high_charge"]
RAW = "data/Telco-Customer-Churn.csv"

def load_df(path=RAW):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    return df

def engineer(df):
    df = df.copy()
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    services = ["PhoneService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
    df["num_services"] = df[services].apply(lambda c: (c == "Yes").astype(int)).sum(axis=1)
    df["tenure_years"] = df["tenure"] / 12.0
    df["charge_per_service"] = df["MonthlyCharges"] / df["num_services"].clip(lower=1)
    df["senior_no_support"] = ((df["SeniorCitizen"] == 1) & (df["TechSupport"] == "No")).astype(int)
    df["fiber_no_security"] = ((df["InternetService"] == "Fiber optic") &
                               (df["OnlineSecurity"] == "No")).astype(int)
    df["month_to_month_high_charge"] = ((df["Contract"] == "Month-to-month") &
                                        (df["MonthlyCharges"] > df["MonthlyCharges"].median())).astype(int)
    return df

def load_xy(path=RAW):
    df = engineer(load_df(path))
    X = df[NUM + CATEGORICAL]
    y = (df["Churn"] == "Yes").astype(int)
    return X, y
