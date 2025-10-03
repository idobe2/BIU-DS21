import numpy as np

rng = np.random.default_rng(42)

# Create a 5x5 matrix filled with random floats between 0 and 1 and print its transpose
matrix = rng.random((5, 5))
print("Transpose:\n", matrix.T)