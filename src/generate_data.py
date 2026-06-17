"""Generate synthetic HR data with turnover relationships grounded in I/O theory.

The probability of leaving rises with low job satisfaction, mandatory overtime,
long commutes, weak manager support, and no recent promotion -- the kinds of
predictors the turnover literature (job embeddedness, met-expectations, the
satisfaction-turnover link) would lead you to expect. Real relationships in the
data mean the model and its interpretation have something true to recover.
"""
import numpy as np
import pandas as pd

import config

RNG = np.random.default_rng(config.RANDOM_STATE)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n: int = 1500) -> pd.DataFrame:
    job_satisfaction = RNG.integers(1, 6, n)          # 1-5
    tenure_years = RNG.gamma(2.0, 2.0, n).round(1)
    commute_minutes = RNG.normal(35, 18, n).clip(5, 120).round()
    manager_support = RNG.integers(1, 6, n)           # 1-5
    age = RNG.normal(38, 10, n).clip(21, 65).round()
    salary_band = RNG.integers(1, 6, n)               # 1-5
    overtime = RNG.binomial(1, 0.3, n)
    promoted = RNG.binomial(1, 0.25, n)
    group = RNG.choice(["A", "B"], n)

    # log-odds of leaving (coefficients chosen to reflect theory, not fit to anything)
    logit = (
        -0.9
        - 0.55 * (job_satisfaction - 3)
        - 0.40 * (manager_support - 3)
        + 0.80 * overtime
        + 0.012 * (commute_minutes - 35)
        - 0.70 * promoted
        - 0.06 * (tenure_years - 4)
        - 0.20 * (salary_band - 3)
    )
    left = RNG.binomial(1, _sigmoid(logit))

    return pd.DataFrame({
        "job_satisfaction": job_satisfaction,
        "tenure_years": tenure_years,
        "commute_minutes": commute_minutes,
        "manager_support": manager_support,
        "age": age,
        "salary_band": salary_band,
        "overtime": overtime,
        "promoted_last_3yrs": promoted,
        "group": group,
        "left": left,
    })


if __name__ == "__main__":
    df = generate()
    config.DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATA, index=False)
    print(f"Wrote {len(df)} employees to {config.DATA}  (attrition rate {df['left'].mean():.1%})")
