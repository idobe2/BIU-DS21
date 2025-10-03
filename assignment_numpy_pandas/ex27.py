import numpy as np

rng = np.random.default_rng(42)

# Explained variance ratio from SVD on centered data
X = rng.normal(size=(50, 4))
Xc = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

# Explained variance per component = S^2 / (n-1) according to the relationship between SVD and covariance
n_samples = Xc.shape[0]
explained_var = (S**2) / (n_samples - 1)
explained_ratio = explained_var / explained_var.sum()
print("Explained variance ratio per component:\n", explained_ratio)