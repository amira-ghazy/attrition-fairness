"""Interpretation, not just importance.

Two lenses:
  - Logistic odds ratios: direction and size of each predictor, in units a
    stakeholder can read ("each point of job satisfaction multiplies the odds of
    leaving by ...").
  - Permutation importance on the boosted model: which predictors actually carry
    the model's predictive power on held-out data.
The point is to connect both back to why, in turnover theory, each predictor matters.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import config

# A short, honest note tying each predictor to the literature.
THEORY = {
    "job_satisfaction": "Core of the satisfaction-turnover link; low satisfaction is a classic antecedent.",
    "manager_support": "Perceived support / LMX quality; weak relationships push people out.",
    "overtime": "Workload and work-life conflict; a strong, actionable driver.",
    "commute_minutes": "An embeddedness / sacrifice factor that erodes retention.",
    "promoted_last_3yrs": "Growth and met expectations; stalled advancement raises quit intentions.",
    "tenure_years": "Embeddedness accrues with time; early tenure is the risky window.",
    "salary_band": "Pay relative to alternatives; matters, but rarely the whole story.",
    "age": "Often a proxy for life stage and mobility; interpret with care.",
}


def main() -> None:
    bundle = joblib.load(config.RESULTS / "models.joblib")
    logit, scaler, hgb = bundle["logit"], bundle["scaler"], bundle["hgb"]
    X_te, y_te = bundle["X_te"], bundle["y_te"]

    odds = pd.Series(np.exp(logit.coef_[0]), index=config.FEATURES).sort_values()

    perm = permutation_importance(hgb, X_te, y_te, n_repeats=20,
                                  random_state=config.RANDOM_STATE, scoring="roc_auc")
    imp = pd.Series(perm.importances_mean, index=config.FEATURES).sort_values(ascending=False)

    lines = ["INTERPRETATION", "=" * 60, "",
             "Logistic odds ratios (standardized; >1 raises odds of leaving)"]
    for feat, val in odds.items():
        lines.append(f"  {feat:<20} OR={val:.2f}   {THEORY.get(feat, '')}")

    lines += ["", "Permutation importance on the boosted model (drop in ROC-AUC)"]
    for feat, val in imp.items():
        lines.append(f"  {feat:<20} {val:+.3f}")

    report = "\n".join(lines)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "interpretation.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
