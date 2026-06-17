"""Who does the model flag, and is it even-handed?

A turnover model is often used to target retention spending. If its errors fall
unevenly across groups -- say it misses at-risk leavers in one group, or wrongly
flags another -- that shapes who gets attention. We report per-group flag rates
and error rates (FPR / FNR) so the disparities are visible, not buried in a
single accuracy number.
"""
import joblib
import numpy as np
import pandas as pd

import config


def rates(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    flag = (tp + fp) / max(tp + fp + tn + fn, 1)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    return flag, fpr, fnr


def main() -> None:
    bundle = joblib.load(config.RESULTS / "models.joblib")
    hgb, X_te, y_te, g_te = bundle["hgb"], bundle["X_te"], bundle["y_te"], bundle["g_te"]
    y_pred = (hgb.predict_proba(X_te)[:, 1] >= 0.5).astype(int)

    y_te = np.asarray(y_te)
    g_te = np.asarray(g_te)

    lines = ["FAIRNESS AUDIT  (gradient-boosting model @0.5)", "=" * 50,
             f"{'group':<8}{'flag rate':>12}{'FPR':>10}{'FNR':>10}"]
    for grp in sorted(pd.unique(g_te)):
        m = g_te == grp
        flag, fpr, fnr = rates(y_te[m], y_pred[m])
        lines.append(f"{grp:<8}{flag:>12.2f}{fpr:>10.2f}{fnr:>10.2f}")

    lines += ["",
              "Read the gaps, not just the averages: a meaningful FNR gap means the",
              "model overlooks at-risk employees in one group more than another."]

    report = "\n".join(lines)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "fairness.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
