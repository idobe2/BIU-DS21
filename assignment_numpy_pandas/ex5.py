import numpy as np

rng = np.random.default_rng(42)

# Create a 5x5 matrix filled with random floats between 0 and 1 and compute the sum of all elements
rand_mat = rng.random((5, 5))
print("sum =", rand_mat.sum())