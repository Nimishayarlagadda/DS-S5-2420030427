
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

X, y = make_blobs(
    n_samples=100,
    centers=3,
    cluster_std=1.5,
    random_state=42
)

print("Hierarchical Clustering")

linked = linkage(X, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(linked)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
