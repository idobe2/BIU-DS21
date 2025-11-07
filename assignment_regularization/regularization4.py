import os
import numpy as np
import pandas as pd
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
    average_precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

RESULTS_DIR = "results"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def round_cols(df):
    for c in df.columns:
        if c not in (
            "model",
            "dataset",
            "C",
            "max_depth",
            "class_weight",
            "threshold_rule",
        ):
            df[c] = df[c].astype(float).round(3)
    return df


def evaluate_all(y_true, y_pred, y_proba_pos):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_proba_pos),
        "avg_precision": average_precision_score(y_true, y_proba_pos),
    }


def plot_roc(y_true, y_proba, title, path_png):
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


def to_row(prefix, metrics: dict):
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def eval_model(name, model, X_train, y_train, X_test, y_test, thr=0.5):
    model.fit(X_train, y_train)
    ytr_proba = model.predict_proba(X_train)[:, 1]
    ytr_pred = (ytr_proba >= thr).astype(int)
    tr = evaluate_all(y_train, ytr_pred, ytr_proba)
    yte_proba = model.predict_proba(X_test)[:, 1]
    yte_pred = (yte_proba >= thr).astype(int)
    te = evaluate_all(y_test, yte_pred, yte_proba)
    row = {"model": name, **to_row("train", tr), **to_row("test", te)}
    return row, yte_proba


def build_datasets():
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
    X_B, y_B = make_classification(
        n_samples=4000,
        n_features=30,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        weights=[0.95, 0.05],
        flip_y=0.0,
        n_clusters_per_class=3,
        random_state=42,
    )
    XA_tr, XA_te, ya_tr, ya_te = train_test_split(
        X_A, y_A, test_size=0.30, stratify=y_A, random_state=42
    )
    XB_tr, XB_te, yb_tr, yb_te = train_test_split(
        X_B, y_B, test_size=0.30, stratify=y_B, random_state=42
    )
    return XA_tr, XA_te, ya_tr, ya_te, XB_tr, XB_te, yb_tr, yb_te


def run_baseline(X_train, y_train, X_test, y_test, dataset_name, class_weight=None):
    lr = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
        class_weight=class_weight,
    )
    row, y_proba_test = eval_model(
        "LogReg (no regularization)", lr, X_train, y_train, X_test, y_test, thr=0.5
    )
    df = pd.DataFrame([row])
    df.insert(0, "dataset", dataset_name)
    df.insert(2, "class_weight", str(class_weight))
    roc_png = os.path.join(
        RESULTS_DIR,
        f"roc_{dataset_name.replace(' ', '_')}_baseline_{'balanced' if class_weight else 'none'}.png",
    )
    pr_png = os.path.join(
        RESULTS_DIR,
        f"pr_{dataset_name.replace(' ', '_')}_baseline_{'balanced' if class_weight else 'none'}.png",
    )
    plot_roc(y_test, y_proba_test, f"{dataset_name} - ROC (LogReg no reg)", roc_png)
    plot_pr(y_test, y_proba_test, f"{dataset_name} - PR (LogReg no reg)", pr_png)
    return round_cols(df), y_proba_test


def run_logreg_L2_grid(
    X_train, y_train, X_test, y_test, dataset_name, C_grid=None, class_weight=None
):
    if C_grid is None:
        C_grid = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3]
    rows = []
    for C in C_grid:
        clf = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=C,
            max_iter=2000,
            random_state=42,
            class_weight=class_weight,
        )
        row, _ = eval_model(f"LogReg_L2_C={C}", clf, X_train, y_train, X_test, y_test)
        row["dataset"] = dataset_name
        row["C"] = C
        row["class_weight"] = str(class_weight)
        rows.append(row)
    df = pd.DataFrame(rows)[
        [
            "dataset",
            "model",
            "C",
            "class_weight",
            "train_accuracy",
            "train_precision",
            "train_recall",
            "train_f1",
            "train_auc",
            "train_avg_precision",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_auc",
            "test_avg_precision",
        ]
    ]
    return round_cols(df)


def run_rf_depth_grid(X_train, y_train, X_test, y_test, dataset_name, depth_grid=None):
    if depth_grid is None:
        depth_grid = [2, 4, 6, 8, None]
    rows = []
    for d in depth_grid:
        clf = RandomForestClassifier(
            n_estimators=60,
            max_depth=d,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt",
        )
        row, _ = eval_model(
            f"RandomForest_depth={d}", clf, X_train, y_train, X_test, y_test
        )
        row["dataset"] = dataset_name
        row["max_depth"] = "None" if d is None else d
        rows.append(row)
    df = pd.DataFrame(rows)[
        [
            "dataset",
            "model",
            "max_depth",
            "train_accuracy",
            "train_precision",
            "train_recall",
            "train_f1",
            "train_auc",
            "train_avg_precision",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_auc",
            "test_avg_precision",
        ]
    ]
    return round_cols(df)


def add_generalization_gap(df):
    df = df.copy()
    df["generalization_gap_auc"] = (
        df["train_auc"].astype(float) - df["test_auc"].astype(float)
    ).round(3)
    return df


def threshold_sweep(y_true, y_proba, rule="best_f1", recall_floor=0.8):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.0)
    best = None
    for p, r, t in zip(precision, recall, thresholds):
        y_pred = (y_proba >= t).astype(int)
        m = evaluate_all(y_true, y_pred, y_proba)
        if rule == "best_f1":
            key, cond = m["f1"], True
        elif rule == "recall_floor":
            key, cond = m["f1"], (m["recall"] >= recall_floor)
        else:
            raise ValueError("Unknown rule")
        if cond and (best is None or key > best["f1"]):
            best = {"threshold": float(t), **m}
    return best


def main():
    ensure_dir(RESULTS_DIR)

    # Part A – datasets
    XA_tr, XA_te, ya_tr, ya_te, XB_tr, XB_te, yb_tr, yb_te = build_datasets()

    # Part B – baselines + plots
    baseline_A, _ = run_baseline(XA_tr, ya_tr, XA_te, ya_te, "Dataset A")
    baseline_B, probs_B = run_baseline(XB_tr, yb_tr, XB_te, yb_te, "Dataset B")
    baseline = pd.concat([baseline_A, baseline_B], ignore_index=True)
    baseline.to_csv(os.path.join(RESULTS_DIR, "baseline_metrics.csv"), index=False)

    # Part C – L2 grid
    l2_A = run_logreg_L2_grid(XA_tr, ya_tr, XA_te, ya_te, "Dataset A")
    l2_B = run_logreg_L2_grid(XB_tr, yb_tr, XB_te, yb_te, "Dataset B")
    l2_A.to_csv(os.path.join(RESULTS_DIR, "logreg_l2_results_A.csv"), index=False)
    l2_B.to_csv(os.path.join(RESULTS_DIR, "logreg_l2_results_B.csv"), index=False)

    # Part C – RF depth grid
    rf_A = run_rf_depth_grid(XA_tr, ya_tr, XA_te, ya_te, "Dataset A")
    rf_B = run_rf_depth_grid(XB_tr, yb_tr, XB_te, yb_te, "Dataset B")
    rf_A.to_csv(os.path.join(RESULTS_DIR, "rf_results_A.csv"), index=False)
    rf_B.to_csv(os.path.join(RESULTS_DIR, "rf_results_B.csv"), index=False)

    # Summaries (printed)
    print("Top L2 (A):")
    print(
        add_generalization_gap(l2_A)
        .sort_values("test_auc", ascending=False)
        .head(5)
        .to_string(index=False)
    )
    print("\nTop L2 (B):")
    print(
        add_generalization_gap(l2_B)
        .sort_values("test_auc", ascending=False)
        .head(5)
        .to_string(index=False)
    )
    print("\nTop RF (A):")
    print(
        add_generalization_gap(rf_A)
        .sort_values("test_auc", ascending=False)
        .head(5)
        .to_string(index=False)
    )
    print("\nTop RF (B):")
    print(
        add_generalization_gap(rf_B)
        .sort_values("test_auc", ascending=False)
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
