"""
KMeans clustering: choosing k with the elbow method and the silhouette score,
and why a distance-based algorithm cares about feature scaling.

Synthetic data is a deliberate choice here. make_blobs gives us a *known* true
number of clusters, so we can check whether each selection method actually
recovers it. No real dataset lets you verify that -- if you knew the right
number of clusters you would not be clustering.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_and_show(fig, filename):
    """Write the figure to plots/ and then display it.

    Saving before showing means the script produces the same artifacts whether
    it is run interactively or headless (MPLBACKEND=Agg), where show() is a no-op.
    """
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=120, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 1. Generate synthetic data with a known number of clusters
# ---------------------------------------------------------------------------
TRUE_K = 4

X, y_true = make_blobs(
    n_samples=500,
    centers=TRUE_K,
    n_features=2,
    cluster_std=1.0,
    random_state=42,
)

print("=" * 70)
print("KMEANS CLUSTERING")
print("=" * 70)
print(f"Dataset shape: {X.shape}")
print(f"True number of clusters (hidden from the algorithm): {TRUE_K}")

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap="viridis", s=25, alpha=0.8)
ax.set_title(f"Ground truth: {TRUE_K} blobs")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
fig.tight_layout()
save_and_show(fig, "01_ground_truth.png")


# ---------------------------------------------------------------------------
# 2. Sweep k and record both selection criteria
# ---------------------------------------------------------------------------
# Inertia = sum of squared distances from each point to its assigned centroid.
# It is defined for k=1. Silhouette compares within-cluster tightness to the
# distance to the nearest *other* cluster, so it needs at least 2 clusters.
K_RANGE = range(1, 11)

inertias = []
silhouettes = []  # None for k=1, which has no "other cluster" to compare against

for k in K_RANGE:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X, labels) if k > 1 else None)

print("\n" + "-" * 70)
print(f"{'k':>3}  {'Inertia':>12}  {'Drop vs k-1':>12}  {'Silhouette':>11}")
print("-" * 70)
for i, k in enumerate(K_RANGE):
    drop = "" if i == 0 else f"{inertias[i - 1] - inertias[i]:12.1f}"
    sil = "" if silhouettes[i] is None else f"{silhouettes[i]:11.4f}"
    print(f"{k:>3}  {inertias[i]:12.1f}  {drop:>12}  {sil:>11}")

# The whole point of the table above: inertia never stops falling, but the size
# of each fall collapses after the true k. Silhouette actually turns around.
best_k_silhouette = max(
    (k for k in K_RANGE if k > 1),
    key=lambda k: silhouettes[list(K_RANGE).index(k)],
)
best_silhouette = silhouettes[list(K_RANGE).index(best_k_silhouette)]

print("-" * 70)
print(f"Inertia is minimized at k={list(K_RANGE)[-1]} "
      f"({inertias[-1]:.1f}) -- and would keep falling to 0 at k=n_samples.")
print(f"Silhouette is maximized at k={best_k_silhouette} ({best_silhouette:.4f}).")
print(f"True k = {TRUE_K}.")


# ---------------------------------------------------------------------------
# 3. Plot the elbow and the silhouette side by side
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(list(K_RANGE), inertias, "o-", color="steelblue")
axes[0].axvline(TRUE_K, color="red", linestyle="--", alpha=0.7,
                label=f"true k = {TRUE_K}")
axes[0].set_title("Elbow method: inertia vs k")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia (within-cluster sum of squares)")
axes[0].legend()
axes[0].grid(alpha=0.3)

k_sil = [k for k in K_RANGE if k > 1]
sil_vals = [s for s in silhouettes if s is not None]
axes[1].plot(k_sil, sil_vals, "o-", color="darkorange")
axes[1].axvline(TRUE_K, color="red", linestyle="--", alpha=0.7,
                label=f"true k = {TRUE_K}")
axes[1].scatter([best_k_silhouette], [best_silhouette], s=180,
                facecolors="none", edgecolors="green", linewidths=2,
                label=f"max at k = {best_k_silhouette}")
axes[1].set_title("Silhouette score vs k")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Mean silhouette score")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.suptitle("Inertia always falls; silhouette has a real optimum")
fig.tight_layout()
save_and_show(fig, "02_elbow_and_silhouette.png")


# ---------------------------------------------------------------------------
# 4. Fit at the chosen k and show the clusters with their centroids
# ---------------------------------------------------------------------------
kmeans = KMeans(n_clusters=best_k_silhouette, n_init=10, random_state=42)
labels = kmeans.fit_predict(X)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=25, alpha=0.8)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           c="red", marker="X", s=250, edgecolors="black", label="centroids")
ax.set_title(f"KMeans with k={best_k_silhouette}")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()
fig.tight_layout()
save_and_show(fig, "03_clusters_with_centroids.png")

# Adjusted Rand Index compares two labelings while ignoring how the labels are
# named -- cluster "0" matching true group "2" is not penalized. 1.0 is a
# perfect partition, 0.0 is what random assignment scores on average.
print(f"\nAdjusted Rand Index vs ground truth: "
      f"{adjusted_rand_score(y_true, labels):.4f}")


# ---------------------------------------------------------------------------
# 5. Why scaling matters: KMeans measures Euclidean distance
# ---------------------------------------------------------------------------
# Stretch feature 1 by 50x, as if it had been recorded in a different unit
# (grams vs kilograms, cents vs euros). The cluster structure is untouched --
# only the axis is. But squared distance is dominated by the large-range axis,
# so the centroids drift toward splitting that axis instead.
X_stretched = X.copy()
X_stretched[:, 1] *= 50.0

labels_raw = KMeans(n_clusters=TRUE_K, n_init=10, random_state=42).fit_predict(X_stretched)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_stretched)
labels_scaled = KMeans(n_clusters=TRUE_K, n_init=10, random_state=42).fit_predict(X_scaled)

ari_raw = adjusted_rand_score(y_true, labels_raw)
ari_scaled = adjusted_rand_score(y_true, labels_scaled)

print("\n" + "-" * 70)
print("Effect of feature scaling (feature 2 multiplied by 50)")
print("-" * 70)
print(f"ARI on unscaled stretched data: {ari_raw:.4f}")
print(f"ARI after StandardScaler:       {ari_scaled:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(X_stretched[:, 0], X_stretched[:, 1], c=labels_raw,
                cmap="viridis", s=25, alpha=0.8)
axes[0].set_title(f"Unscaled: ARI = {ari_raw:.3f}")
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2 (x50)")

axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_scaled,
                cmap="viridis", s=25, alpha=0.8)
axes[1].set_title(f"StandardScaler: ARI = {ari_scaled:.3f}")
axes[1].set_xlabel("Feature 1 (scaled)")
axes[1].set_ylabel("Feature 2 (scaled)")

fig.suptitle("KMeans is distance-based, so unit choice changes the answer")
fig.tight_layout()
save_and_show(fig, "04_scaling_effect.png")


# ---------------------------------------------------------------------------
# 6. n_init: KMeans is not deterministic from a single start
# ---------------------------------------------------------------------------
# KMeans converges to a *local* minimum that depends on where the centroids
# started. sklearn runs the whole algorithm n_init times and keeps the run with
# the lowest inertia. Setting n_init=1 exposes the variance this hides.
single_start_inertias = []
for seed in range(10):
    km = KMeans(n_clusters=TRUE_K, n_init=1, init="random", random_state=seed)
    km.fit(X)
    single_start_inertias.append(km.inertia_)

best_of_ten = KMeans(n_clusters=TRUE_K, n_init=10, random_state=42).fit(X).inertia_

print("\n" + "-" * 70)
print("Local minima: 10 single-start runs with random initialization")
print("-" * 70)
print(f"Inertia range across seeds: {min(single_start_inertias):.1f} "
      f"to {max(single_start_inertias):.1f}")
print(f"Spread: {max(single_start_inertias) - min(single_start_inertias):.1f}")
print(f"n_init=10 with k-means++ init: {best_of_ten:.1f}")


# ---------------------------------------------------------------------------
# 7. Takeaways
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TAKEAWAYS")
print("=" * 70)
print("""
1. Inertia cannot be used as an objective to minimize over k. It falls
   monotonically and hits exactly 0 when k equals the number of samples
   (every point becomes its own centroid). The elbow is a visual heuristic
   for where the marginal gain collapses -- not an optimum.

2. Silhouette has a genuine maximum because it penalizes both loose clusters
   and clusters that sit too close to each other. That makes it the more
   defensible criterion, at the cost of being O(n^2) in memory to compute.

3. KMeans measures squared Euclidean distance, so any feature with a wider
   numeric range dominates the objective. Rescaling the units of a feature
   changes the clustering even though the underlying structure did not move.
   Scale first unless every feature is already in the same unit.

4. KMeans finds a local minimum determined by its initialization. sklearn's
   default n_init=10 plus k-means++ hides that; n_init=1 with random init
   exposes it.

5. KMeans assumes roughly spherical, similarly sized clusters -- that is what
   "assign to the nearest centroid" implies geometrically. make_blobs produces
   exactly that, which is why it works so cleanly here. Elongated or nested
   shapes need a different algorithm (DBSCAN, spectral clustering).
""")
