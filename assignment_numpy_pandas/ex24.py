import numpy as np

rng = np.random.default_rng(42)

# Computing the covariance matrix of a NumPy array with 3 features (columns)
X = rng.normal(size=(10, 3))
C = np.cov(X, rowvar=False, ddof=1)
print("Covariance matrix:\n", C)