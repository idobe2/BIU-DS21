import numpy as np

rng = np.random.default_rng(42)

# Eigenvalues and eigenvectors of a (5x5) covariance matrix
X = rng.normal(size=(100, 5))  # Sample enough to ensure stable covariance
C = np.cov(X, rowvar=False, ddof=1)
eigvals, eigvecs = np.linalg.eigh(C)  # Suitable for symmetric/hermitian matrices
print("Eigenvalues:\n", eigvals)