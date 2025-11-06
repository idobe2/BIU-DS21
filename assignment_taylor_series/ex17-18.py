import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


# Dataset creation
X_q17, y_q17 = make_classification(n_samples=100, n_features=1, n_informative=1,
                                   n_redundant=0, n_clusters_per_class=1, random_state=42)

# Sort X for better plotting
sort_indices = np.argsort(X_q17.ravel())
X_sorted = X_q17[sort_indices]


# Solution 17
log_reg = LogisticRegression()
log_reg.fit(X_q17, y_q17)
y_pred_log = log_reg.predict_proba(X_sorted)[:, 1] # Probability estimates


# Solution 18
log_reg_no_intercept = LogisticRegression(fit_intercept=False)
log_reg_no_intercept.fit(X_q17, y_q17)
y_pred_no_intercept = log_reg_no_intercept.predict_proba(X_sorted)[:, 1]

# Plotting results for questions 17 and 18
plt.figure(figsize=(10, 6))
plt.scatter(X_q17, y_q17, color='black', zorder=20, alpha=0.5, label='Data (0/1)')

plt.plot(X_sorted, y_pred_log, color='blue', 
         linewidth=3, label='Q17: Logistic Regression (Cross-Entropy)')

plt.plot(X_sorted, y_pred_no_intercept, color='red', linestyle='--',
         linewidth=3, label='Q18: fit_intercept = False')

plt.title("Logistic Regression")
plt.ylabel("Probability")
plt.xlabel("Feature (e.g., 'Weight')")
plt.legend()
plt.grid(True)
plt.show()