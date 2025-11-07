import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss, mean_squared_error, accuracy_score

# binary classification (1 feature)
X, y = make_classification(n_samples=50, n_features=1, n_informative=1,
                           n_redundant=0, n_clusters_per_class=1, random_state=4)
X_sorted = np.sort(X, axis=0)

# "correct" for classification: Logistic Regression (optimizes Cross-Entropy)
log_clf = LogisticRegression()
log_clf.fit(X, y)
p_log = log_clf.predict_proba(X)[:, 1]              # probabilities in [0,1]
yhat_log = (p_log >= 0.5).astype(int)

# "wrong" for classification: Linear Regression (optimizes MSE)
lin_clf = LinearRegression()
lin_clf.fit(X, y)
p_lin = lin_clf.predict(X)                           # not bounded to [0,1]
p_lin_clip = np.clip(p_lin, 0, 1)                    # only for computing log-loss fairly
yhat_lin = (p_lin >= 0.5).astype(int)

# print simple metrics to compare losses and decisions
print("Effect of Loss Function")
print(f"Logistic (Cross-Entropy):  log_loss={log_loss(y, p_log):.4f}, "
      f"MSE={mean_squared_error(y, p_log):.4f}, Acc={accuracy_score(y, yhat_log):.3f}")
print(f"Linear   (MSE on outputs): log_loss={log_loss(y, p_lin_clip):.4f} (after clipping), "
      f"MSE={mean_squared_error(y, p_lin):.4f}, Acc={accuracy_score(y, yhat_lin):.3f}")
print(f"Note: linear predictions min={p_lin.min():.2f}, max={p_lin.max():.2f}  (not probabilities)")

# plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', zorder=20, alpha=0.7, label='Data (0/1)')

plt.plot(X_sorted, log_clf.predict_proba(X_sorted)[:, 1], 'b-', lw=3,
         label='Logistic Regression (Cross-Entropy)')
plt.plot(X_sorted, lin_clf.predict(X_sorted), 'r--', lw=3,
         label='Linear Regression (MSE)')

plt.title("Effect of Loss Function on a Small Classification Problem")
plt.xlabel("Feature")
plt.ylabel("Prediction (Probability / Value)")
plt.grid(True)
plt.legend()
plt.show()
