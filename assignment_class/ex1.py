import pandas as pd
from dowhy import CausalModel
from pathlib import Path

# Load the Lalonde dataset
def load_lalonde_dataframe() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[1] / "data" / "lalonde_data.csv"
    return pd.read_csv(data_path)

df = load_lalonde_dataframe()

# Define treatment and outcome
X = "treat"
Y = "re78"

# Define confounders
confounder_cols = [
    "age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"
]

# Naive mean difference
naive_mean_diff = df[df[X] == 1][Y].mean() - df[df[X] == 0][Y].mean()

print(f"n={len(df)} treated={int((df[X] == 1).sum())} control={int((df[X] == 0).sum())}")
print(f"Naive mean difference (E[Y|X=1] - E[Y|X=0]) = {naive_mean_diff:.6f}")

# Define the causal model with DAG
model = CausalModel(
    data=df,
    treatment=X,
    outcome=Y,
    common_causes=confounder_cols,
    graph='''
    digraph {
        age -> treat
        age -> re78

        educ -> treat
        educ -> re78

        black -> treat
        black -> re78

        hispan -> treat
        hispan -> re78

        married -> treat
        married -> re78

        nodegree -> treat
        nodegree -> re78

        re74 -> treat
        re74 -> re78

        re75 -> treat
        re75 -> re78

        treat -> re78
    }
    '''
)

# Visualize the DAG (opens in notebook or saves PDF if specified)
model.view_model()

# Identify the causal effect
identified_estimand = model.identify_effect()
print(identified_estimand)

# Estimate the effect using linear regression
estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.linear_regression")
print("Causal Estimate:", estimate.value)

# Refutation 1: Add a random common cause
ref1 = model.refute_estimate(identified_estimand, estimate,
                             method_name="random_common_cause")
print("Refutation (Random Common Cause):", ref1)

# Refutation 2: Placebo treatment test
ref2 = model.refute_estimate(identified_estimand, estimate,
                             method_name="placebo_treatment_refuter",
                             placebo_type="permute")
print("Refutation (Placebo Treatment):", ref2)

# Additional refutation: Data subset validation
ref3 = model.refute_estimate(identified_estimand, estimate,
                            method_name="data_subset_refuter",
                            subset_fraction=0.8)
print("\nRefutation (Data Subset):", ref3)
