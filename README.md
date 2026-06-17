# attrition-fairness
Employee turnover prediction built to be explained and audited — odds ratios tied to turnover theory, plus per-group error-rate parity.
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

The bundled data is synthetic but built from real turnover relationships (the
satisfaction-turnover link, job embeddedness, met-expectations), so the model has
something true to recover.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.generate_data    # synthetic HR data with real relationships
python -m src.model            # train + evaluate, save models
python -m src.interpret        # odds ratios + permutation importance
python -m src.fairness         # per-group error rates
```

Outputs land in `results/`.

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
├── src/
│   ├── generate_data.py    # synthetic HR data grounded in turnover theory
│   ├── model.py            # logistic + gradient boosting, evaluation
│   ├── interpret.py        # odds ratios + permutation importance + theory notes
│   └── fairness.py         # per-group flag rate, FPR, FNR
├── data/                   # generated data (git-ignored)
└── results/                # reports + saved models (git-ignored)
```

## Limitations and next steps

- Synthetic data: the relationships are planted, so treat the numbers as a demonstration
  of the *method*, not findings.
- SHAP values and partial-dependence plots are a natural addition for local explanations.
- A production version would calibrate probabilities and add confidence intervals to the
  fairness gaps before anyone acts on them.

## License

MIT.
