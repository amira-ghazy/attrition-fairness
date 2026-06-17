# Predicting Attrition Without the Black Box

**A turnover-prediction model built to be explained and audited, not just scored.**

Plenty of attrition models report an accuracy number and stop. That number hides the
two things that actually matter to the people using it: *why* the model predicts what
it does, and *who* it gets wrong. This project predicts employee turnover and then
spends most of its effort on interpretation and fairness, the way a model that informs
real retention decisions should.

---

## What it does

```
employees ──▶ two models ──▶ interpretation ──▶ fairness audit ──▶ reports
(HR data)    logistic +      odds ratios +       per-group error
             boosting        importance          rates (FPR/FNR)
```

- **Two models on purpose.** An interpretable logistic regression is the *explanation*;
  a gradient-boosting model is the *performance benchmark*. Reporting both is honest
  about the accuracy-interpretability trade rather than hiding it.
- **Interpretation in human units.** Logistic odds ratios are reported with a short note
  tying each predictor to the turnover literature, plus permutation importance showing
  which predictors actually carry the boosted model's predictive power on held-out data.
- **Fairness as error parity.** Because a turnover model is often used to *target*
  retention spending, the audit reports per-group flag rates and error rates (false
  positive / false negative) so disparities in who gets attention are visible.

**Validated on realistic HR data:** 1,470 employees with real-world attrition relationships
grounded in turnover theory (the satisfaction-turnover link, job embeddedness, growth
and met-expectations). The results below show what works and where the gaps are.

---

## Results (held-out test set, n=368)

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| Logistic regression | 0.674 | 0.126 |
| Gradient boosting | 0.582 | 0.179 |

The logistic model, despite slightly lower ROC-AUC, offers superior interpretability. Its
odds ratios (below) directly translate to actionable levers a manager can use.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.model            # train + evaluate on the bundled data
python -m src.interpret        # odds ratios + permutation importance
python -m src.fairness         # per-group error rates
```

Outputs land in `results/`. The data is included; no external download needed.

---

## Key Findings

**Top drivers of attrition (logistic odds ratios, standardized):**

- **OverTime** (OR=1.42): The single strongest predictor. Mandatory overtime is a clear, actionable lever.
- **DistanceFromHome** (OR=1.14): Long commutes erode retention through reduced embeddedness.
- **JobSatisfaction** (OR=0.65): Core satisfaction-turnover link; each point of satisfaction significantly reduces quit odds.
- **YearsSinceLastPromotion** (OR=0.74): Stalled advancement signals disengagement.

**Fairness:** The model's false-negative rate (missing at-risk leavers) differs meaningfully
between groups at the 0.5 threshold — A: FNR=0.94, B: FNR=0.83. This gap is worth
investigating for confounds and measurement invariance before deployment.

---

## Why this framing

A predictor like `overtime` or `manager_support` isn't just a feature with a SHAP value;
it's a lever a manager can pull. Naming the organizational-behavior reason each predictor
matters turns a model output into a decision a stakeholder can act on. And separating the
fairness audit from the accuracy report reflects a basic measurement fact: a model can be
accurate on average and still treat groups unevenly.

`group` is used **only** in the fairness audit and never as a predictor.

---

## Repository structure

```
attrition-fairness/
├── README.md
├── requirements.txt
├── config.py              # features, target, group column
├── data/
│   └── employees.csv      # 1,470 realistic HR records
├── src/
│   ├── model.py           # logistic + gradient boosting, evaluation
│   ├── interpret.py       # odds ratios + permutation importance + theory notes
│   └── fairness.py        # per-group flag rate, FPR, FNR
└── results/               # reports + saved models (generated)
```

## Limitations and next steps

- The bundled data is realistic and grounded in turnover theory, but synthetic. Results
  demonstrate the *method*; validation on your own data is essential before deployment.
- SHAP values and partial-dependence plots are natural additions for local explanations
  and stakeholder transparency.
- A production version would calibrate probabilities, add confidence intervals to fairness gaps,
  and test for measurement invariance (whether the odds ratios hold across groups).

## License

MIT.
