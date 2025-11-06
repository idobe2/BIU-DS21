import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_classification

# יצירת דאטה-סט סיווג
X_q30, y_q30 = make_classification(n_samples=50, n_features=1, n_informative=1,
                                   n_redundant=0, n_clusters_per_class=1, random_state=4)
X_sorted = np.sort(X_q30, axis=0)

# 1. מודל נכון: רגרסיה לוגיסטית (ממזער Cross-Entropy)
model_log = LogisticRegression()
model_log.fit(X_q30, y_q30)
y_pred_log = model_log.predict_proba(X_sorted)[:, 1]

# 2. מודל שגוי: רגרסיה ליניארית (ממזער MSE)
model_lin = LinearRegression()
model_lin.fit(X_q30, y_q30)
y_pred_lin = model_lin.predict(X_sorted)

# הצגת הגרף
plt.figure(figsize=(10, 6))
plt.scatter(X_q30, y_q30, color='black', zorder=20, alpha=0.7, label='Data (0/1)')

plt.plot(X_sorted, y_pred_log, 'b-', 
         label='Logistic Regression (Loss: Cross-Entropy)', lw=3)
plt.plot(X_sorted, y_pred_lin, 'r--', 
         label='Linear Regression (Loss: MSE)', lw=3)

plt.title("Q30: Effect of Loss Function on a Classification Problem")
plt.ylabel("Prediction (Probability / Value)")
plt.xlabel("Feature")
plt.legend()
plt.grid(True)
plt.show()