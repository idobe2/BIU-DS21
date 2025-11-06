import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


RESULTS_DIR = "results"
RANDOM_STATE = 42
N_SAMPLES = 4000

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_df_csv(df, path):
    df.to_csv(path, index=False)

def to_df(X, y, prefix):
    cols = [f"{prefix}f{i+1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    return df

# Return all requested metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
def evaluate_all(y_true, y_pred, y_proba_pos):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_proba_pos) if len(np.unique(y_true)) == 2 else np.nan
    return acc, prec, rec, f1, auc

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

# LogisticRegression with no regularization (penalty='none').
def try_logreg_no_reg():
    try:
        return LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
    except Exception:
        return LogisticRegression(penalty="l2", solver="lbfgs", C=1e12, max_iter=1000, random_state=RANDOM_STATE)

def eval_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    # train
    ytr_pred = model.predict(X_train)
    ytr_proba = model.predict_proba(X_train)[:, 1]
    tr_acc, tr_prec, tr_rec, tr_f1, tr_auc = evaluate_all(y_train, ytr_pred, ytr_proba)

    # test
    yte_pred = model.predict(X_test)
    yte_proba = model.predict_proba(X_test)[:, 1]
    te_acc, te_prec, te_rec, te_f1, te_auc = evaluate_all(y_test, yte_pred, yte_proba)

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
    return row, yte_proba  # return test probabilities for ROC/PR plotting

def round_cols(df):
    for c in df.columns:
        if c not in ("model", "dataset", "C", "max_depth"):
            df[c] = df[c].astype(float).round(3)
    return df


# ===== חלק א — יצירת שני הסטים בדיוק לפי ההנחיות =====

def build_datasets():
    # A: מאוזן, רועש: 15 פיצ'רים – כולם אינפורמטיביים; flip_y=0.3
    X_A, y_A = make_classification(
        n_samples=N_SAMPLES,
        n_features=15,
        n_informative=15,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        weights=None,        # מאוזן
        flip_y=0.30,         # רעש תוויות גבוה
        random_state=RANDOM_STATE
    )

    # B: לא מאוזן מאוד (95/5), מורכב מאוד, 5 אינפורמטיביים + 30 לא-רלוונטיים = 35 פיצ'רים
    X_B, y_B = make_classification(
        n_samples=N_SAMPLES,
        n_features=35,       # 5 informative + 30 irrelevant
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        weights=[0.95, 0.05],
        flip_y=0.0,          # ללא רעש ב-B
        n_clusters_per_class=3,  # להדגשת "מורכב מאוד"
        random_state=RANDOM_STATE
    )

    # פיצולים עם stratify
    X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(
        X_A, y_A, test_size=0.30, stratify=y_A, random_state=RANDOM_STATE
    )
    X_B_train, X_B_test, y_B_train, y_B_test = train_test_split(
        X_B, y_B, test_size=0.30, stratify=y_B, random_state=RANDOM_STATE
    )

    # שמירות אופציונליות לשימוש חוזר
    ensure_dir(RESULTS_DIR)
    save_df_csv(to_df(X_A_train, y_A_train, "A_"), os.path.join(RESULTS_DIR, "dataset_A_train.csv"))
    save_df_csv(to_df(X_A_test,  y_A_test,  "A_"), os.path.join(RESULTS_DIR, "dataset_A_test.csv"))
    save_df_csv(to_df(X_B_train, y_B_train, "B_"), os.path.join(RESULTS_DIR, "dataset_B_train.csv"))
    save_df_csv(to_df(X_B_test,  y_B_test,  "B_"), os.path.join(RESULTS_DIR, "dataset_B_test.csv"))

    print("A shape:", X_A.shape, "A balance:", Counter(y_A))
    print("B shape:", X_B.shape, "B balance:", Counter(y_B))
    print("A train/test:", Counter(y_A_train), Counter(y_A_test))
    print("B train/test:", Counter(y_B_train), Counter(y_B_test))

    return (X_A_train, X_A_test, y_A_train, y_A_test,
            X_B_train, X_B_test, y_B_train, y_B_test)


# ===== חלק ב — מודל בסיסי: LogisticRegression בלי רגולריזציה + ROC/PR =====

def run_baseline(X_train, y_train, X_test, y_test, dataset_name):
    lr = try_logreg_no_reg()
    row, y_proba_test = eval_model("LogReg (no regularization)", lr, X_train, y_train, X_test, y_test)

    # טבלת מדדים (שורה אחת עבור הסט)
    df = pd.DataFrame([row])
    df.insert(0, "dataset", dataset_name)
    df = round_cols(df)

    # גרפים (על test בלבד, כנדרש)
    roc_png = os.path.join(RESULTS_DIR, f"roc_{dataset_name.replace(' ', '_')}_baseline.png")
    pr_png  = os.path.join(RESULTS_DIR, f"pr_{dataset_name.replace(' ', '_')}_baseline.png")
    plot_roc(y_test, y_proba_test, f"{dataset_name} - ROC (LogReg, no reg)", roc_png)
    plot_pr(y_test, y_proba_test,  f"{dataset_name} - Precision-Recall (LogReg, no reg)", pr_png)

    return df

# ===== חלק ג — LogisticRegression עם L2 (סריקת C) + RandomForest עם שליטה ב-max_depth =====

def run_logreg_L2_grid(X_train, y_train, X_test, y_test, dataset_name, C_grid=None):
    if C_grid is None:
        C_grid = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3]
    rows = []
    for C in C_grid:
        clf = LogisticRegression(penalty="l2", solver="lbfgs", C=C, max_iter=1000, random_state=RANDOM_STATE)
        row, _ = eval_model(f"LogReg_L2_C={C}", clf, X_train, y_train, X_test, y_test)
        row["dataset"] = dataset_name
        row["C"] = C
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df[["dataset", "model", "C",
             "train_accuracy", "train_precision", "train_recall", "train_f1", "train_auc",
             "test_accuracy",  "test_precision",  "test_recall",  "test_f1",  "test_auc"]]
    return round_cols(df)

def run_rf_depth_grid(X_train, y_train, X_test, y_test, dataset_name, depth_grid=None):
    if depth_grid is None:
        depth_grid = [2, 4, 6, 8, None]  # None = ללא הגבלת עומק
    rows = []
    for d in depth_grid:
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=d, random_state=RANDOM_STATE, n_jobs=-1
        )
        row, _ = eval_model(f"RandomForest_depth={d}", clf, X_train, y_train, X_test, y_test)
        row["dataset"] = dataset_name
        row["max_depth"] = "None" if d is None else d
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df[["dataset", "model", "max_depth",
             "train_accuracy", "train_precision", "train_recall", "train_f1", "train_auc",
             "test_accuracy",  "test_precision",  "test_recall",  "test_f1",  "test_auc"]]
    return round_cols(df)

def add_generalization_gap(df):
    df = df.copy()
    df["generalization_gap_auc"] = (df["train_auc"] - df["test_auc"]).astype(float).round(3)
    return df


# ===== main =====

def main():
    ensure_dir(RESULTS_DIR)

    # חלק א
    (X_A_train, X_A_test, y_A_train, y_A_test,
     X_B_train, X_B_test, y_B_train, y_B_test) = build_datasets()

    # חלק ב – מודל בסיסי + גרפים
    baseline_A = run_baseline(X_A_train, y_A_train, X_A_test, y_A_test, "Dataset A")
    baseline_B = run_baseline(X_B_train, y_B_train, X_B_test, y_B_test, "Dataset B")
    baseline = pd.concat([baseline_A, baseline_B], ignore_index=True)
    save_df_csv(baseline, os.path.join(RESULTS_DIR, "baseline_metrics.csv"))
    print("\n=== Baseline metrics (LogReg no reg) ===")
    print(baseline.to_string(index=False))

    # חלק ג – LogisticRegression עם L2 (סריקת C)
    logreg_l2_A = run_logreg_L2_grid(X_A_train, y_A_train, X_A_test, y_A_test, "Dataset A")
    logreg_l2_B = run_logreg_L2_grid(X_B_train, y_B_train, X_B_test, y_B_test, "Dataset B")
    save_df_csv(logreg_l2_A, os.path.join(RESULTS_DIR, "logreg_l2_results_A.csv"))
    save_df_csv(logreg_l2_B, os.path.join(RESULTS_DIR, "logreg_l2_results_B.csv"))

    # סיכומי L2 לפי AUC(test)
    sum_A = add_generalization_gap(logreg_l2_A).sort_values("test_auc", ascending=False)
    sum_B = add_generalization_gap(logreg_l2_B).sort_values("test_auc", ascending=False)
    print("\n=== LogisticRegression L2 — best by test AUC (Dataset A) ===")
    print(sum_A.head(5).to_string(index=False))
    print("\n=== LogisticRegression L2 — best by test AUC (Dataset B) ===")
    print(sum_B.head(5).to_string(index=False))

    # חלק ג – RandomForest עם שליטה ב-max_depth
    rf_A = run_rf_depth_grid(X_A_train, y_A_train, X_A_test, y_A_test, "Dataset A")
    rf_B = run_rf_depth_grid(X_B_train, y_B_train, X_B_test, y_B_test, "Dataset B")
    save_df_csv(rf_A, os.path.join(RESULTS_DIR, "rf_results_A.csv"))
    save_df_csv(rf_B, os.path.join(RESULTS_DIR, "rf_results_B.csv"))

    # סיכומי RF לפי AUC(test)
    rf_sum_A = add_generalization_gap(rf_A).sort_values("test_auc", ascending=False)
    rf_sum_B = add_generalization_gap(rf_B).sort_values("test_auc", ascending=False)
    print("\n=== RandomForest — best by test AUC (Dataset A) ===")
    print(rf_sum_A.head(5).to_string(index=False))
    print("\n=== RandomForest — best by test AUC (Dataset B) ===")
    print(rf_sum_B.head(5).to_string(index=False))

    print(f"\nקבצים נוצרו בתיקייה: {os.path.abspath(RESULTS_DIR)}")
    print("גרפים לחלק ב' (ROC/PR) נשמרו בשם: roc_*_baseline.png, pr_*_baseline.png")
    print("טבלאות נשמרו: baseline_metrics.csv, logreg_l2_results_*.csv, rf_results_*.csv")


if __name__ == "__main__":
    main()
