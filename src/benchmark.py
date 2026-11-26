"""5-fold stratified CV across the zoo. Model selection happens HERE only."""
import json
from sklearn.model_selection import StratifiedKFold, cross_validate
from .prep import load_xy
from .models import zoo

def main():
    X, y = load_xy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out = {}
    for name, m in zoo().items():
        r = cross_validate(m, X, y, cv=cv, scoring=["accuracy", "roc_auc", "f1"], n_jobs=1)
        out[name] = {"cv_accuracy": round(float(r["test_accuracy"].mean()), 4),
                     "cv_accuracy_std": round(float(r["test_accuracy"].std()), 4),
                     "cv_roc_auc": round(float(r["test_roc_auc"].mean()), 4),
                     "cv_f1": round(float(r["test_f1"].mean()), 4)}
        print(name, out[name], flush=True)
    json.dump(out, open("results/cv_results.json", "w"), indent=2)
    print("saved results/cv_results.json")

if __name__ == "__main__":
    main()
