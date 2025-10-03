import numpy as np

rng = np.random.default_rng(1)

# Create data with one feature having much larger variance
X = np.hstack([
    10 * rng.normal(size=(100, 1)),
    rng.normal(size=(100, 3))
])
X_c = X - X.mean(axis=0)
Uh, Sh, Vth = np.linalg.svd(X_c, full_matrices=False)
ratio = (Sh**2)/(X_c.shape[0]-1)
ratio /= ratio.sum()
print("Explained variance ratio:\n", ratio)