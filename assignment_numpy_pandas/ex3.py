import numpy as np

rng = np.random.default_rng(42)

# Create a 4x4 matrix filled with random floats between 0 and 1
rand_mat = rng.random((4, 4))
print("Random Matrix:\n", rand_mat)