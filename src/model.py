"""Train and evaluate two models: interpretable logistic regression and a
gradient-boosting model. We keep both on purpose -- the logistic model is the
explanation, the boosted model is the performance benchmark.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config


def load_split():
    df = pd.read_csv(config.DATA)
    X = df[config.FEATURES]
    y = df[config.TARGET]
    groups = df[config.GROUP_COL]
    return train_test_split(X, y, groups, test_size=0.25,
                            random_state=config.RANDOM_STATE, stratify=y)


def main() -> None:
    X_tr, X_te, y_tr, y_te, g_tr, g_te = load_split()

    scaler = StandardScaler().fit(X_tr)
    logit = LogisticRegression(max_iter=1000).fit(scaler.transform(X_tr), y_tr)
    hgb = HistGradientBoostingClassifier(random_state=config.RANDOM_STATE).fit(X_tr, y_tr)

    lines = ["MODEL EVALUATION (held-out test set)", "=" * 44]
    for name, p in [("Logistic regression", logit.predict_proba(scaler.transform(X_te))[:, 1]),
                    ("Gradient boosting", hgb.predict_proba(X_te)[:, 1])]:
        lines.append(f"{name:<22} ROC-AUC={roc_auc_score(y_te, p):.3f}  PR-AUC={average_precision_score(y_te, p):.3f}")

    lines += ["", "Gradient-boosting detail @0.5 threshold",
              classification_report(y_te, (hgb.predict_proba(X_te)[:, 1] >= .5).astype(int), digits=3)]

    report = "\n".join(lines)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "evaluation.txt").write_text(report + "\n")
    joblib.dump({"logit": logit, "scaler": scaler, "hgb": hgb,
                 "X_te": X_te, "y_te": y_te, "g_te": g_te}, config.RESULTS / "models.joblib")
    print(report)


if __name__ == "__main__":
    main()
