import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

RESULTS_DIR = "results"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_df_csv(df, path):
    df.to_csv(path, index=False)


def to_df(X, y, prefix):
    cols = [f"{prefix}f{i+1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    return df


def evaluate_all(y_true, y_pred, y_proba_pos):
    """Returns the required metrics: Accuracy, Precision, Recall, F1, ROC-AUC."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba_pos) if len(np.unique(y_true)) == 2 else np.nan
    return accuracy, precision, recall, f1, auc


def plot_roc(y_true, y_proba, title, path_png):
    """Create and save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_png, dpi=144)
    plt.close()


def plot_pr(y_true, y_proba, title, path_png):
    """Create and save Precision-Recall curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure()
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_png, dpi=144)
    plt.close()


def eval_model(name, model, X_train, y_train, X_test, y_test):
    """Helper function: train model and return a full row of metrics for train and test."""
    model.fit(X_train, y_train)

    # Calculate train metrics
    ytr_pred = model.predict(X_train)
    ytr_proba = model.predict_proba(X_train)[:, 1]
    tr_acc, tr_prec, tr_rec, tr_f1, tr_auc = evaluate_all(y_train, ytr_pred, ytr_proba)

    # Calculate test metrics
    yte_pred = model.predict(X_test)
    yte_proba = model.predict_proba(X_test)[:, 1]
    te_acc, te_prec, te_rec, te_f1, te_auc = evaluate_all(y_test, yte_pred, yte_proba)

    # Package results
    row = {
        "model": name,
        "train_accuracy": tr_acc,
        "train_precision": tr_prec,
        "train_recall": tr_rec,
        "train_f1": tr_f1,
        "train_auc": tr_auc,
        "test_accuracy": te_acc,
        "test_precision": te_prec,
        "test_recall": te_rec,
        "test_f1": te_f1,
        "test_auc": te_auc,
    }
    return row, yte_proba  # Return test probabilities as well for plotting


def round_cols(df):
    """Round all numeric columns in a DataFrame to 3 decimal places."""
    for c in df.columns:
        if c not in ("model", "dataset", "C", "max_depth"):
            df[c] = df[c].astype(float).round(3)
    return df


# ----- Part A: Create two datasets -----


def build_datasets():
    """Create datasets A and B as per specifications, return train/test splits."""
    # A: Balanced, very noisy, 15 informative = 15 features
    X_A, y_A = make_classification(
        n_samples=4000,
        n_features=15,
        n_informative=15,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        weights=None,
        flip_y=0.30,
        random_state=42,
    )

    # B: Unbalanced (95/5), very complex, 5 informative + 30 irrelevant = 35 features
    X_B, y_B = make_classification(
        n_samples=4000,
        n_features=35,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        weights=[0.95, 0.05],
        flip_y=0.0,
        n_clusters_per_class=3,
        random_state=42,
    )

    # Splits with stratify=y (critical for set B)
    X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(
        X_A, y_A, test_size=0.30, stratify=y_A, random_state=42
    )
    X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(
        X_B, y_B, test_size=0.30, stratify=y_B, random_state=42
    )

    print("A shape:", X_A.shape, "A balance:", Counter(y_A))
    print("B shape:", X_B.shape, "B balance:", Counter(y_B))
    print("B train/test balance:", Counter(y_B_train), Counter(y_B_test))
    print("Datasets created.\n")

    return (
        X_A_train,
        X_A_test,
        y_A_train,
        y_A_test,
        X_B_train,
        X_B_test,
        y_B_train,
        y_B_test,
    )


# ----- Part B: Baseline model - LogisticRegression w/o regularization + ROC/PR -----


def run_baseline(X_train, y_train, X_test, y_test, dataset_name):
    """Run baseline model w/o regularization, including plotting"""
    print(f"Running baseline on {dataset_name}...")
    lr = LogisticRegression(
        penalty=None, solver="lbfgs", max_iter=1000, random_state=42
    )
    row, y_proba_test = eval_model(
        "LogReg (no regularization)", lr, X_train, y_train, X_test, y_test
    )

    # Metrics table (one row for the set)
    df = pd.DataFrame([row])
    df.insert(0, "dataset", dataset_name)
    df = round_cols(df)

    # Plots (on test set only, as required)
    roc_png = os.path.join(
        RESULTS_DIR, f"roc_{dataset_name.replace(' ', '_')}_baseline.png"
    )
    pr_png = os.path.join(
        RESULTS_DIR, f"pr_{dataset_name.replace(' ', '_')}_baseline.png"
    )
    plot_roc(y_test, y_proba_test, f"{dataset_name} - ROC (LogReg, no reg)", roc_png)
    plot_pr(
        y_test,
        y_proba_test,
        f"{dataset_name} - Precision-Recall (LogReg, no reg)",
        pr_png,
    )
    print(f"ROC/PR plots for {dataset_name} saved.")

    return df


# ----- Part C — LogisticRegression with L2 (C scan) + RandomForest with max_depth control -----


def run_logreg_L2_grid(X_train, y_train, X_test, y_test, dataset_name, C_grid=None):
    """Run L2 regularization experiment"""
    print(f"Running L2 (LogReg) scan on {dataset_name}...")
    if C_grid is None:
        C_grid = [
            1e-3,
            1e-2,
            1e-1,
            1,
            10,
            100,
            1e3,
        ]  # 0.001, 0.01, 0.1, 1, 10, 100, 1000
    rows = []
    for C in C_grid:
        clf = LogisticRegression(
            penalty="l2", solver="lbfgs", C=C, max_iter=1000, random_state=42
        )
        row, _ = eval_model(f"LogReg_L2_C={C}", clf, X_train, y_train, X_test, y_test)
        row["dataset"] = dataset_name
        row["C"] = C
        rows.append(row)
    df = pd.DataFrame(rows)
    # Arrange columns for a clean table
    df = df[
        [
            "dataset",
            "model",
            "C",
            "train_accuracy",
            "train_precision",
            "train_recall",
            "train_f1",
            "train_auc",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_auc",
        ]
    ]
    return round_cols(df)


def run_rf_depth_grid(X_train, y_train, X_test, y_test, dataset_name, depth_grid=None):
    """Run structural regularization experiment (RandomForest)"""
    print(f"Running max_depth (RandomForest) scan on {dataset_name}...")
    if depth_grid is None:
        depth_grid = [2, 4, 6, 8, None]  # None = no depth limit
    rows = []
    for d in depth_grid:
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=d, random_state=42, n_jobs=-1
        )
        row, _ = eval_model(
            f"RandomForest_depth={d}", clf, X_train, y_train, X_test, y_test
        )
        row["dataset"] = dataset_name
        row["max_depth"] = "None" if d is None else d
        rows.append(row)
    df = pd.DataFrame(rows)
    # Arrange columns for a clean table
    df = df[
        [
            "dataset",
            "model",
            "max_depth",
            "train_accuracy",
            "train_precision",
            "train_recall",
            "train_f1",
            "train_auc",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_auc",
        ]
    ]
    return round_cols(df)


def add_generalization_gap(df):
    """Calculate generalization gap (train-test difference) to show overfitting"""
    df = df.copy()
    df["generalization_gap_auc"] = (
        (df["train_auc"] - df["test_auc"]).astype(float).round(3)
    )
    return df


# ----- Driver code -----


def main():
    ensure_dir(RESULTS_DIR)

    # ----- Part A -----
    # Create datasets A and B
    (
        X_A_train,
        X_A_test,
        y_A_train,
        y_A_test,
        X_B_train,
        X_B_test,
        y_B_train,
        y_B_test,
    ) = build_datasets()

    # ----- Part B -----
    # Run baseline models (no regularization) and create plots
    baseline_A = run_baseline(X_A_train, y_A_train, X_A_test, y_A_test, "Dataset A")
    baseline_B = run_baseline(X_B_train, y_B_train, X_B_test, y_B_test, "Dataset B")
    baseline = pd.concat([baseline_A, baseline_B], ignore_index=True)
    save_df_csv(baseline, os.path.join(RESULTS_DIR, "baseline_metrics.csv"))
    print("\n=== (Part B) Baseline metrics (LogReg no reg) ===")
    print(baseline.to_string(index=False))

    # ----- Part C -----
    # LogisticRegression with L2 (C-scan)
    logreg_l2_A = run_logreg_L2_grid(
        X_A_train, y_A_train, X_A_test, y_A_test, "Dataset A"
    )
    logreg_l2_B = run_logreg_L2_grid(
        X_B_train, y_B_train, X_B_test, y_B_test, "Dataset B"
    )
    save_df_csv(logreg_l2_A, os.path.join(RESULTS_DIR, "logreg_l2_results_A.csv"))
    save_df_csv(logreg_l2_B, os.path.join(RESULTS_DIR, "logreg_l2_results_B.csv"))

    # Summary L2
    sum_A = add_generalization_gap(logreg_l2_A).sort_values("test_auc", ascending=False)
    sum_B = add_generalization_gap(logreg_l2_B).sort_values("test_auc", ascending=False)
    print("\nLogisticRegression L2 — Best test AUC (Dataset A)")
    print(sum_A.head(5).to_string(index=False))
    print("\nLogisticRegression L2 — Best test AUC (Dataset B)")
    print(sum_B.head(5).to_string(index=False))

    # RandomForest with max_depth control
    rf_A = run_rf_depth_grid(X_A_train, y_A_train, X_A_test, y_A_test, "Dataset A")
    rf_B = run_rf_depth_grid(X_B_train, y_B_train, X_B_test, y_B_test, "Dataset B")
    save_df_csv(rf_A, os.path.join(RESULTS_DIR, "rf_results_A.csv"))
    save_df_csv(rf_B, os.path.join(RESULTS_DIR, "rf_results_B.csv"))

    # Summary RF
    rf_sum_A = add_generalization_gap(rf_A).sort_values("test_auc", ascending=False)
    rf_sum_B = add_generalization_gap(rf_B).sort_values("test_auc", ascending=False)
    print("\nRandomForest — Best test AUC (Dataset A)")
    print(rf_sum_A.head(5).to_string(index=False))
    print("\nRandomForest — Best test AUC (Dataset B)") 
    print(rf_sum_B.head(5).to_string(index=False))

    print(f"\nRun complete. Files saved in: {os.path.abspath(RESULTS_DIR)}")

if __name__ == "__main__":
    main()
