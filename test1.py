from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter

X_A, y_A = make_classification(
    n_samples=4000,  # size; feel free to change
    n_features=15,  # total features
    n_informative=5,  # informative
    n_redundant=5,  # correlated with informative
    n_repeated=0,
    n_classes=2,
    weights=[0.5, 0.5],  # balanced
    flip_y=0.30,  # <-- high label noise per spec
    class_sep=0.8,  # moderate separation
    random_state=42,
)

print("Dataset A class balance:", Counter(y_A))

X_B, y_B = make_classification(
    n_samples=4000,  # pick enough samples so 5% isn't too tiny (≈200 positives)
    n_features=35,  # 5 informative + 30 irrelevant
    n_informative=5,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    weights=[0.95, 0.05],  # <-- 95% vs 5%
    flip_y=0.00,  # focus on imbalance + irrelevant features (noise comes from irrelevance)
    n_clusters_per_class=3,  # "very complex" class structure
    class_sep=0.8,  # moderate separation
    random_state=42,
)

print("Dataset B class balance:", Counter(y_B))

# Dataset A split
X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(
    X_A, y_A, test_size=0.30, stratify=y_A, random_state=42
)

# Dataset B split
X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(
    X_B, y_B, test_size=0.30, stratify=y_B, random_state=42
)

print("A - train:", Counter(y_A_train), "test:", Counter(y_A_test))
print("B - train:", Counter(y_B_train), "test:", Counter(y_B_test))


# Helper to build column names
def to_df(X, y, prefix="f"):
    cols = [f"{prefix}{i+1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    return df


df_A_train = to_df(X_A_train, y_A_train, prefix="A_")
df_A_test = to_df(X_A_test, y_A_test, prefix="A_")
df_B_train = to_df(X_B_train, y_B_train, prefix="B_")
df_B_test = to_df(X_B_test, y_B_test, prefix="B_")

# Save if you like:
df_A_train.to_csv("dataset_A_train.csv", index=False)
df_A_test.to_csv("dataset_A_test.csv", index=False)
df_B_train.to_csv("dataset_B_train.csv", index=False)
df_B_test.to_csv("dataset_B_test.csv", index=False)
