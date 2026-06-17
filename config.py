"""Configuration for the attrition-prediction project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "employees.csv"
RESULTS = ROOT / "results"

TARGET = "left"          # 1 = employee left, 0 = stayed
GROUP_COL = "group"      # used only for the fairness audit, never as a predictor
RANDOM_STATE = 42

# Predictors, grouped the way an I/O psychologist would reason about them.
NUMERIC = ["job_satisfaction", "tenure_years", "commute_minutes",
           "manager_support", "age", "salary_band"]
BINARY = ["overtime", "promoted_last_3yrs"]
FEATURES = NUMERIC + BINARY
