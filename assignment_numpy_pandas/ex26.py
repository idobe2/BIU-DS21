import numpy as np

rng = np.random.default_rng(42)

X = rng.normal(size=(100, 5))
C = np.cov(X, rowvar=False, ddof=1)
eigvals, eigvecs = np.linalg.eigh(C)

# Sort eigenvalues and eigenvectors in descending order
idx = np.argsort(eigvals)[::-1]
eigvals_sorted = eigvals[idx]
eigvecs_sorted = eigvecs[:, idx]
print("Sorted eigenvalues:\n", eigvals_sorted)