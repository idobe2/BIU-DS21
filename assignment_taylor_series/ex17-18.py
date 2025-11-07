import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error

# ----- simple "height & weight" data -----
rng = np.random.default_rng(42)
n_per_class = 120

# class 0: shorter
h0 = rng.normal(170, 7, n_per_class)  # height (cm)
w0 = rng.normal(70, 10, n_per_class)  # weight (kg)
y0 = np.zeros(n_per_class, dtype=int)

# class 1: taller
h1 = rng.normal(180, 7, n_per_class)  # height (cm)
w1 = rng.normal(85, 10, n_per_class)  # weight (kg)
y1 = np.ones(n_per_class, dtype=int)

X = np.column_stack([np.concatenate([h0, h1]), np.concatenate([w0, w1])])
y = np.concatenate([y0, y1])

# ----- Q17: logistic regression -----
clf_int = LogisticRegression(fit_intercept=True, solver="lbfgs")
clf_int.fit(X, y)
p_int = clf_int.predict_proba(X)[:, 1]

# print losses
print("Q17 (fit_intercept=True)")
print("  Cross-Entropy (log_loss):", log_loss(y, p_int))
print("  MSE on probabilities    :", mean_squared_error(y, p_int))

# ----- Q18: fit_intercept=False and compare -----
clf_no = LogisticRegression(fit_intercept=False, solver="lbfgs")
clf_no.fit(X, y)
p_no = clf_no.predict_proba(X)[:, 1]
print("\nQ18 (fit_intercept=False)")
print("  Cross-Entropy (log_loss):", log_loss(y, p_no))
print("  MSE on probabilities    :", mean_squared_error(y, p_no))

# ----- plot points + both decision boundaries -----
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], s=20, alpha=0.6, label="Class 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], s=20, alpha=0.6, label="Class 1")

x1_min, x1_max = X[:, 0].min() - 2, X[:, 0].max() + 2
xx = np.linspace(x1_min, x1_max, 200)

w_int = clf_int.coef_[0]
b_int = clf_int.intercept_[0]
yy_int = -(w_int[0] / w_int[1]) * xx - b_int / w_int[1]
plt.plot(xx, yy_int, label="fit_intercept=True", linewidth=2)

w_no = clf_no.coef_[0]
b_no = 0.0  # no intercept
yy_no = -(w_no[0] / w_no[1]) * xx - b_no / w_no[1]
plt.plot(xx, yy_no, "--", label="fit_intercept=False", linewidth=2)

plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("Logistic Regression: effect of fit_intercept")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
