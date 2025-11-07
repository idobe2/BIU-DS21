import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Create synthetic 2D data with correlation
np.random.seed(42)
N = 300
x = np.random.normal(0, 5, size=N)
y = 2 * x + np.random.normal(0, 2, size=N)
data = np.column_stack([x, y])

# Perform PCA
pca = PCA(n_components=2)
pca.fit(data)

# Display results
print("----- PCA Results -----")
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Principal Components (Axes):\n {pca.components_}")

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], alpha=0.5, label="Original Data")
# Plotting principal components
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    comp = comp * np.sqrt(var) * 3  # Scale vector for better visibility
    plt.plot(
        [0, comp[0]],
        [0, comp[1]],
        "r-",
        lw=3,
        label=f"Component {i+1} (Var: {pca.explained_variance_ratio_[i]:.2f})",
    )
plt.title("PCA on 2D Synthetic Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()
