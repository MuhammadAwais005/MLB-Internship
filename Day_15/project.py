
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score


GRAPH_DIR = Path(__file__).resolve().parent / "graph"
GRAPH_DIR.mkdir(exist_ok=True)


#Load the dataset and convert it to a DataFrame

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df["species"] = pd.Categorical.from_codes(
    iris.target, categories=iris.target_names
)

print("=" * 60)
print("IRIS FLOWER CLUSTERING & VISUALIZATION")
print("=" * 60)

print("\nFirst five rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nData types and non-null values:")
df.info()

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nNumber of samples per actual species:")
print(df["species"].value_counts())


#Prepare features

feature_names = iris.feature_names
X = df[feature_names]

#Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#Elbow Method

k_values = range(1, 11)
inertias = []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.title("Elbow Method: Choosing K")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(GRAPH_DIR / "project_elbow_method.png")
plt.show()


#Choose K and train the final K-Means model

# We choose K=3 to compare the clusters with the three known species.
chosen_k = 3

kmeans = KMeans(
    n_clusters=chosen_k,
    random_state=42,
    n_init=10
)

# IMPORTANT: Create the cluster column before using it in plots.
df["cluster"] = kmeans.fit_predict(X_scaled)

print(f"\nNumber of clusters selected: {chosen_k}")
print("\nNumber of flowers in each cluster:")
print(df["cluster"].value_counts().sort_index())


#Visualize K-Means clusters using original features

plt.figure(figsize=(8, 5))

scatter = plt.scatter(
    df["petal length (cm)"],
    df["petal width (cm)"],
    c=df["cluster"],
    cmap="viridis",
    s=45,
    alpha=0.8
)

plt.title("K-Means Clusters: Petal Length vs Petal Width")
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "project_kmeans_petals.png")
plt.show()


#Apply PCA to reduce four features to two components

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

variance = pca.explained_variance_ratio_
total_variance = variance.sum() * 100

print("\nPCA explained variance:")
print("PC1:", round(variance[0] * 100, 2), "%")
print("PC2:", round(variance[1] * 100, 2), "%")
print("Total variance retained:", round(total_variance, 2), "%")


#Visualize PCA-transformed data by K-Means cluster

plt.figure(figsize=(8, 5))

for cluster_id in sorted(df["cluster"].unique()):
    points = df[df["cluster"] == cluster_id]

    plt.scatter(
        points["PC1"],
        points["PC2"],
        label=f"Cluster {cluster_id}",
        s=45,
        alpha=0.8
    )

plt.title("PCA Projection Colored by K-Means Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="K-Means Clusters")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "project_pca_clusters.png")
plt.show()


#Compare the PCA view with actual species

# Species labels are used only for comparison and visualization.
# K-Means and PCA above do not use the species labels.

plt.figure(figsize=(8, 5))

for species in iris.target_names:
    points = df[df["species"] == species]

    plt.scatter(
        points["PC1"],
        points["PC2"],
        label=species,
        s=45,
        alpha=0.8
    )

plt.title("PCA Projection Colored by Actual Species")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Actual species")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "project_pca_species.png")
plt.show()


#Print observations and a comparison metric

ari = adjusted_rand_score(iris.target, df["cluster"])

print("\n" + "=" * 60)
print("PROJECT OBSERVATIONS")
print("=" * 60)

print(f"1. K-Means was configured to form {chosen_k} clusters.")
print("2. Compare the cluster plot with the actual-species plot.")
print("3. Iris-setosa is generally easier to separate; versicolor and")
print("   virginica can overlap, so clusters may not match species perfectly.")
print(
    "4. The first two PCA components retain",
    round(total_variance, 2),
    "% of variance."
)
print("5. PCA makes four-dimensional data easier to visualize in 2D,")
print("   but some information may be lost in the projection.")
print("6. Adjusted Rand Index vs actual species:", round(ari, 3))
print("   ARI is a comparison metric; it does not mean K-Means used labels.")

print("\nMini project complete.")
