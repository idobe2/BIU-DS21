import numpy as np

rng = np.random.default_rng(42)

# Create random data and compute SVD to analyze energy retention in top components
X = rng.normal(size=(10, 20))
U3, S3, Vt3 = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
total_energy = (S3**2).sum()
top3_energy = (S3[:3]**2).sum()
print("Total energy =", total_energy, "\nTop-3 energy =", top3_energy,
      "\nRatio kept =", top3_energy / total_energy)