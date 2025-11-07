from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

# Train/test split (keep class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ----- Q19: Random Forest -----
print("\nRandom Forest")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")

# Sort feature importances
rf_importances = sorted(
    zip(feature_names, rf_model.feature_importances_), key=lambda x: x[1], reverse=True
)
print("Random Forest Feature Importances:")
for name, imp in rf_importances:
    print(f"  {name}: {imp:.4f}")

# ----- Q20: XGBoost -----
print("\nXGBoost & Comparison")
xgb_model = XGBClassifier(
    random_state=42, eval_metric="mlogloss"  # multi-class log-loss
)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb * 100:.2f}%")

xgb_importances = sorted(
    zip(feature_names, xgb_model.feature_importances_), key=lambda x: x[1], reverse=True
)
print("XGBoost Feature Importances:")
for name, imp in xgb_importances:
    print(f"  {name}: {imp:.4f}")
