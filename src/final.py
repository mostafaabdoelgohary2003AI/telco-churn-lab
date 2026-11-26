"""Final evaluation. Model chosen by 5-fold CV on the TRAIN part only (80%); the 20% test part is used once."""
import json, joblib, numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score
from .prep import load_xy
from .models import zoo

def main():
    X, y = load_xy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = json.load(open("results/cv_results.json"))
    best = max(cv, key=lambda k: cv[k]["cv_roc_auc"])
    print("best by CV AUC:", best, cv[best], flush=True)
    m = zoo()[best]
    # decision threshold picked on out-of-fold train predictions (maximise accuracy), never on test
    oof = cross_val_predict(m, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
    ths = np.linspace(0.2, 0.8, 61); th = float(ths[int(np.argmax([accuracy_score(ytr, oof > t) for t in ths]))])
    m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]
    res = {"chosen_model": best, "threshold_from_train_oof": round(th, 2), "train_rows": len(Xtr), "test_rows": len(Xte),
           "test_accuracy_at_threshold": round(accuracy_score(yte, p > th), 4),
           "test_accuracy_at_0.5": round(accuracy_score(yte, p > 0.5), 4),
           "test_roc_auc": round(roc_auc_score(yte, p), 4),
           "test_f1_at_0.5": round(f1_score(yte, p > 0.5), 4),
           "test_precision_at_0.5": round(precision_score(yte, p > 0.5), 4),
           "test_recall_at_0.5": round(recall_score(yte, p > 0.5), 4),
           "confusion_at_0.5": confusion_matrix(yte, p > 0.5).tolist(),
           "majority_class_baseline_accuracy": round(1 - float(yte.mean()), 4)}
    print(res)
    json.dump(res, open("results/final_metrics.json", "w"), indent=2)
    m.fit(X, y); joblib.dump({"model": m, "threshold": th}, "results/telco_model.joblib")

if __name__ == "__main__":
    main()
