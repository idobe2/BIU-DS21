import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# Scree plot using SVD on centered data
X = rng.normal(size=(100, 5))
Xc = X - X.mean(axis=0)
U2, S2, Vt2 = np.linalg.svd(Xc, full_matrices=False)
eigs_approx = (S2**2) / (Xc.shape[0] - 1) # Eigenvalues approximation from SVD
plt.figure()
plt.plot(np.arange(1, len(eigs_approx) + 1), np.sort(eigs_approx)[::-1], marker='o')
plt.title("Scree Plot (Eigenvalues)")
plt.xlabel("Component")
plt.ylabel("Eigenvalue")
plt.tight_layout()
plt.show()