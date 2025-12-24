"""
Causality vs Correlation – Guided Assignment
Course: Data Science
Topic: Causal Inference using DoWhy

This script demonstrates the full causal inference pipeline:
1. Load data
2. Define a causal question
3. Perform naive (correlational) analysis
4. Define causal assumptions using a DAG
5. Estimate causal effect using DoWhy
6. Perform refutation (robustness checks)

Author: Course Staff
"""

# =========================
# Part 0: Imports & Setup
# =========================

import pandas as pd
import numpy as np
from dowhy import CausalModel
import dowhy.datasets

# =========================
# Part 1: Load the Dataset
# =========================

df = dowhy.datasets.lalonde_dataset()

try:
    data = df["df"]
except Exception:
    data = df

print("First rows of the dataset:")
print(data.head())

# =========================
# Part 2: Define Variables
# =========================

print("\nColumns in dataset:")
print(data.columns)

# =========================
# Part 3: Naive Correlation Analysis
# =========================

treated_mean = data[data["treat"] == 1]["re78"].mean()
control_mean = data[data["treat"] == 0]["re78"].mean()
naive_difference = treated_mean - control_mean

print("\nNaive (correlational) analysis:")
print(f"Mean income (treated): {treated_mean:.2f}")
print(f"Mean income (control): {control_mean:.2f}")
print(f"Naive difference (treated - control): {naive_difference:.2f}")

# =========================
# Part 4: Define Causal Graph (DAG)
# =========================

graph = """
digraph {
age -> treat; educ -> treat; black -> treat; hisp -> treat;
married -> treat; nodegr -> treat; re74 -> treat; re75 -> treat;

age -> re78; educ -> re78; black -> re78; hisp -> re78;
married -> re78; nodegr -> re78; re74 -> re78; re75 -> re78;

treat -> re78;
}
"""

# =========================
# Part 5: Build Causal Model
# =========================

model = CausalModel(
    data=data,
    treatment="treat",
    outcome="re78",
    graph=graph
)

# =========================
# Part 6: Identify Causal Effect
# =========================

identified_estimand = model.identify_effect()

print("\nIdentified causal estimand:")
print(identified_estimand)

# =========================
# Part 7: Estimate Causal Effect
# =========================

estimate_psm = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_matching"
)

print("\nCausal Effect Estimate using Propensity Score Matching:")
print(f"ATE (PSM): {estimate_psm.value:.2f}")

estimate_lr = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)

print("\nCausal Effect Estimate using Linear Regression:")
print(f"ATE (Linear Regression): {estimate_lr.value:.2f}")

# =========================
# Part 8: Refutation Tests
# =========================

refute_placebo = model.refute_estimate(
    identified_estimand,
    estimate_psm,
    method_name="placebo_treatment_refuter"
)

print("\nRefutation using Placebo Treatment:")
print(refute_placebo)

refute_random = model.refute_estimate(
    identified_estimand,
    estimate_psm,
    method_name="random_common_cause"
)

print("\nRefutation using Random Common Cause:")
print(refute_random)

print("\nAnalysis complete.")
