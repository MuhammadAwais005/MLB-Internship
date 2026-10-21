# Day 15: Iris Clustering and PCA

This project explores the Iris dataset with K-Means clustering and Principal
Component Analysis (PCA). The scripts use the four numeric flower measurements.
The known species labels are kept out of model fitting and are used only to
compare with the resulting clusters.

## What is clustering?

Clustering is an unsupervised learning method that groups data points based on
similarity in their features. K-Means assigns each point to one of `K` groups
by finding cluster centers and reducing the distances between points and their
assigned centers. It does not need pre-existing labels.

In this project, the four Iris measurements are standardized before K-Means is
applied. Standardization puts the measurements on a comparable scale so that a
feature with larger numeric values does not dominate the distance calculation.

## What is PCA?

Principal Component Analysis is a dimensionality-reduction method. It
transforms the original features into new, uncorrelated axes called principal
components. The first component captures as much of the data's variation as
possible, and the next captures as much of the remaining variation as possible.

Here, PCA reduces the four standardized Iris measurements to two components so
the data can be plotted in 2D. In the project run, these components retain
about **95.81%** of the total variance (PC1: **72.96%**, PC2: **22.85%**).
PCA is used to create a view of the data; it does not use the species labels.

## How was the best value of K determined?

Both scripts calculate K-Means inertia for values of `K` from 1 to 10 and save
an elbow-method chart. Inertia measures how tightly points fit within their
assigned clusters. It always decreases as more clusters are added, so the
useful choice is around the bend in the curve, where adding clusters starts
producing smaller improvements.

The scripts use **K = 3** for the final clustering. This is a practical choice
based on the elbow curve and allows comparison with the Iris dataset's three
known species. The species labels are not used to select or train the clusters.
The code sets this value explicitly; it does not automatically detect the
elbow.

## Insights from the visualizations

- The elbow chart shows how the reduction in inertia changes as `K` increases,
  helping identify a reasonable number of clusters.
- In the original-feature and PCA cluster plots, Iris-setosa is generally
  easier to distinguish, while versicolor and virginica can overlap.
- The PCA plots show the cluster assignments and actual species in the same
  two-dimensional space. Comparing them illustrates that discovered clusters
  do not necessarily match the known species perfectly.
- In the project run, the Adjusted Rand Index comparing clusters with species
  is **0.62**. This is a comparison score, not a training input.
- The PCA view retains most, but not all, of the variation in the original
  four features.

## Run the scripts

Install Python with `pandas`, `matplotlib`, and `scikit-learn`, then run either
script from this folder:

```powershell
python .\project.py
python .\iris_coding_practice.py
```

Each script displays its charts and saves PNG copies in the `graph` folder.
The project charts are `project_elbow_method.png`, `project_kmeans_petals.png`,
`project_pca_clusters.png`, and `project_pca_species.png`. The practice charts
use the same descriptive names with the `practice_` prefix.
