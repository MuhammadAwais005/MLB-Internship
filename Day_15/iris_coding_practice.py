from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


GRAPH_DIR = Path(__file__).resolve().parent / "graph"
GRAPH_DIR.mkdir(exist_ok=True)


# Load the Iris dataset and create a DataFrame

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(
    iris.target, categories=iris.target_names
)

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape (rows, columns):")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nSummary statistics:")
print(df.describe())

print("\nSpecies counts:")
print(df["species"].value_counts())


# K-Means and PCA work with numeric features, not the species column.
X = df[iris.feature_names]

# Standardize features so measurements on larger scales do not dominate.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# Elbow Method: compare inertia for different K values

inertias = []
k_values = range(1, 11)

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(list(k_values), inertias, marker="o")
plt.title("Elbow Method for Iris Dataset")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(GRAPH_DIR / "practice_elbow_method.png")
plt.show()

best_k = 3

# Apply K-Means and visualize two selected original features

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

print(f"\nK-Means cluster counts for K={best_k}:")
print(df["cluster"].value_counts().sort_index())

# Sepal length and sepal width are used for this 2D original-feature plot.
plt.figure(figsize=(7, 5))
scatter = plt.scatter(
    df["sepal length (cm)"],
    df["sepal width (cm)"],
    c=df["cluster"],
    cmap="viridis",
    s=45,
    alpha=0.8
)
plt.title("K-Means Clusters Using Original Iris Features")
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.colorbar(scatter, label="K-Means cluster")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "practice_kmeans_features.png")
plt.show()



# Apply PCA to reduce four features to two components

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["cluster"] = df["cluster"]
pca_df["species"] = df["species"]

print("\nPCA explained variance ratio:")
print(pca.explained_variance_ratio_)
print(
    "Total variance retained by 2 components:",
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)

# Plot the PCA projection, colored by K-Means cluster.
plt.figure(figsize=(7, 5))
for cluster_id in sorted(pca_df["cluster"].unique()):
    points = pca_df[pca_df["cluster"] == cluster_id]
    plt.scatter(points["PC1"], points["PC2"], label=f"Cluster {cluster_id}", s=45)

plt.title("PCA Visualization Colored by K-Means Cluster")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.tight_layout()
plt.savefig(GRAPH_DIR / "practice_pca_clusters.png")
plt.show()

# Optional comparison: show the known species in the same PCA space.
# These labels are used only for comparison, not to train K-Means or PCA.
plt.figure(figsize=(7, 5))
for species in iris.target_names:
    points = pca_df[pca_df["species"] == species]
    plt.scatter(points["PC1"], points["PC2"], label=species, s=45)

plt.title("PCA Visualization Colored by Actual Species")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.tight_layout()
plt.savefig(GRAPH_DIR / "practice_pca_species.png")
plt.show()

print("\nPractice complete.")
