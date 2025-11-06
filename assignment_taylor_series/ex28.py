import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# יצירת נתונים דו-ממדיים עם קורלציה (מבוסס מצגת [cite: 291-298])
np.random.seed(42)
N = 300
x = np.random.normal(0, 5, size=N)
y = 2*x + np.random.normal(0, 2, size=N)
data = np.column_stack([x, y])

# הפעלת PCA [cite: 300-301]
pca = PCA(n_components=2)
pca.fit(data)

# הצגת התוצאות
print("--- Q28: PCA Results ---")
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Principal Components (Axes):\n {pca.components_}")

# הצגת גרף
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], alpha=0.5, label='Original Data')
# שרטוט הרכיבים העיקריים
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    comp = comp * np.sqrt(var) * 3 # שרטוט וקטור ארוך יותר
    plt.plot([0, comp[0]], [0, comp[1]], 'r-', lw=3,
             label=f'Component {i+1} (Var: {pca.explained_variance_ratio_[i]:.2f})')
plt.title("Q28: PCA on 2D Synthetic Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()