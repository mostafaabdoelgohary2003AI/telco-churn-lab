"""Model zoo for the honest-ceiling study. All preprocessing inside pipelines."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              ExtraTreesClassifier, StackingClassifier, VotingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from .prep import NUM, CATEGORICAL

def pre():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])

def zoo():
    z = {}
    z["logistic_regression"] = Pipeline([("pre", pre()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])
    z["hist_gradient_boosting"] = Pipeline([("pre", pre()),
        ("clf", HistGradientBoostingClassifier(random_state=42, max_iter=400, learning_rate=0.06, early_stopping=False))])
    z["hgb_tuned"] = Pipeline([("pre", pre()),
        ("clf", HistGradientBoostingClassifier(random_state=42, max_iter=800, learning_rate=0.04,
                                               max_leaf_nodes=15, l2_regularization=2.0,
                                               min_samples_leaf=40, early_stopping=False))])
    z["random_forest_balanced"] = Pipeline([("pre", pre()),
        ("clf", RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced"))])
    z["extra_trees_balanced"] = Pipeline([("pre", pre()),
        ("clf", ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced"))])
    z["mlp"] = Pipeline([("pre", pre()),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.01, max_iter=600,
                              early_stopping=True, random_state=42))])
    stack = StackingClassifier(
        estimators=[("lr", LogisticRegression(max_iter=3000, class_weight="balanced")),
                    ("hgb", HistGradientBoostingClassifier(random_state=42, max_iter=400, learning_rate=0.06, early_stopping=False)),
                    ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"))],
        final_estimator=LogisticRegression(max_iter=2000), cv=3)
    z["stacking_ensemble"] = Pipeline([("pre", pre()), ("clf", stack)])
    vote = VotingClassifier(
        estimators=[("lr", LogisticRegression(max_iter=3000, class_weight="balanced")),
                    ("hgb", HistGradientBoostingClassifier(random_state=42, max_iter=400, learning_rate=0.06, early_stopping=False)),
                    ("et", ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"))],
        voting="soft")
    z["soft_voting"] = Pipeline([("pre", pre()), ("clf", vote)])
    return z
