import numpy as np

# Initializing the matrix M
M = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [5, 7, 9]
])

# Performing SVD
U, S, Vh = np.linalg.svd(M)

print("----- SVD Results -----")
print("Matrix U (Left Singular Vectors):\n", U)
print("\nSingular Values (S):\n", S)
print("\nMatrix Vh (Right Singular Vectors Transposed):\n", Vh)