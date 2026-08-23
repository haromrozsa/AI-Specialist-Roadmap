"""
PCA + KMeans together on the digits dataset.

Two questions this script answers with measurements rather than assertions:

  1. Does reducing dimensionality before clustering help, hurt, or do nothing?
  2. Is "always StandardScaler first" actually right?

Both are commonly stated as rules. Both depend on the data, and this dataset is
a case where the second rule is wrong.

The true labels are used *only* to score the result afterwards. Nothing in the
clustering pipeline ever sees them -- that is what makes it unsupervised.
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

N_CLUSTERS = 10  # one per digit -- a choice justified by the domain, not by the data


def save_and_show(fig, filename):
    """Write the figure to plots/ and then display it."""
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=120, bbox_inches="tight")
    plt.show()


def cluster_and_score(X_input, y_true, label):
    """Run KMeans on already-transformed data and report timing plus scores."""
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    start = time.perf_counter()
    labels = kmeans.fit_predict(X_input)
    elapsed = time.perf_counter() - start
    return {
        "label": label,
        "dims": X_input.shape[1],
        "seconds": elapsed,
        # ARI compares the partition to the true labels while ignoring label
        # names. It is a diagnostic here, not an objective -- see takeaway 4.
        "ari": adjusted_rand_score(y_true, labels),
        # Silhouette is computed in the same space the clustering happened in,
        # so values are only comparable across rows with the same dimensionality.
        "silhouette": silhouette_score(X_input, labels),
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------------
digits = load_digits()
X = digits.data
y = digits.target

print("=" * 78)
print("PCA + KMEANS PIPELINE")
print("=" * 78)
print(f"Dataset: {X.shape[0]} images, {X.shape[1]} pixel features, "
      f"{len(np.unique(y))} true classes")
print(f"Clustering into k={N_CLUSTERS} (chosen from domain knowledge: "
      f"ten digits)")


# ---------------------------------------------------------------------------
# 2. Three pipelines, same clustering step
# ---------------------------------------------------------------------------
# Order matters and is not arbitrary:
#   scale -> PCA -> cluster
# PCA maximizes variance, so whichever features have the largest numeric range
# dominate the components. If features carry different units, scaling has to
# happen before PCA or PCA is measuring units rather than structure. Scaling
# after PCA would defeat the point -- it would re-inflate the low-variance
# components PCA just demoted.
pca_95 = PCA(n_components=0.95, random_state=42)
X_pca = pca_95.fit_transform(X)
n_components_95 = X_pca.shape[1]

scaled_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95, random_state=42)),
])
X_scaled_pca = scaled_pipeline.fit_transform(X)

results = [
    cluster_and_score(X, y, "Raw 64-D pixels"),
    cluster_and_score(X_pca, y, f"PCA 95% ({n_components_95}-D)"),
    cluster_and_score(X_scaled_pca, y,
                      f"StandardScaler + PCA 95% ({X_scaled_pca.shape[1]}-D)"),
]

print("\n" + "-" * 78)
print(f"{'Pipeline':<38} {'Dims':>5} {'ARI':>8} {'Silhouette':>12} {'Fit (s)':>9}")
print("-" * 78)
for r in results:
    print(f"{r['label']:<38} {r['dims']:>5} {r['ari']:>8.4f} "
          f"{r['silhouette']:>12.4f} {r['seconds']:>9.3f}")
print("-" * 78)

best = max(results, key=lambda r: r["ari"])
print(f"Best ARI: {best['label']} ({best['ari']:.4f})")
print(f"\nPCA at 95% variance: {X.shape[1]} -> {n_components_95} dimensions "
      f"({X.shape[1] / n_components_95:.1f}x reduction)")
print(f"StandardScaler + PCA at 95%: {X.shape[1]} -> {X_scaled_pca.shape[1]} "
      f"dimensions ({X.shape[1] / X_scaled_pca.shape[1]:.1f}x reduction)")


# ---------------------------------------------------------------------------
# 3. Sweep the component count
# ---------------------------------------------------------------------------
# One threshold tells you nothing about the shape of the tradeoff. Sweeping it
# shows where the useful signal actually lives.
COMPONENT_COUNTS = [2, 5, 10, 15, 20, 30, 40, 64]

sweep_ari = []
sweep_time = []

for k in COMPONENT_COUNTS:
    X_k = PCA(n_components=k, random_state=42).fit_transform(X)
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    start = time.perf_counter()
    labels_k = km.fit_predict(X_k)
    sweep_time.append(time.perf_counter() - start)
    sweep_ari.append(adjusted_rand_score(y, labels_k))

print("\n" + "-" * 78)
print(f"{'Components':>11}  {'ARI':>8}  {'KMeans fit (s)':>15}")
print("-" * 78)
for k, ari, secs in zip(COMPONENT_COUNTS, sweep_ari, sweep_time):
    print(f"{k:>11}  {ari:>8.4f}  {secs:>15.3f}")

best_sweep_idx = int(np.argmax(sweep_ari))
print("-" * 78)
print(f"Peak ARI at {COMPONENT_COUNTS[best_sweep_idx]} components "
      f"({sweep_ari[best_sweep_idx]:.4f}); "
      f"all 64 components give {sweep_ari[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(COMPONENT_COUNTS, sweep_ari, "o-", color="steelblue")
axes[0].axhline(sweep_ari[-1], color="gray", linestyle="--", alpha=0.7,
                label=f"all 64 dims ({sweep_ari[-1]:.3f})")
axes[0].scatter([COMPONENT_COUNTS[best_sweep_idx]], [sweep_ari[best_sweep_idx]],
                s=180, facecolors="none", edgecolors="green", linewidths=2,
                label=f"peak at {COMPONENT_COUNTS[best_sweep_idx]} PCs")
axes[0].set_title("Clustering agreement vs number of components")
axes[0].set_xlabel("PCA components")
axes[0].set_ylabel("Adjusted Rand Index vs true digits")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(COMPONENT_COUNTS, sweep_time, "o-", color="darkorange")
axes[1].set_title("KMeans fit time vs number of components\n"
                  "(flat and non-monotonic: overhead dominates at n=1797)")
axes[1].set_xlabel("PCA components")
axes[1].set_ylabel("Seconds")
axes[1].set_ylim(bottom=0)
axes[1].grid(alpha=0.3)

fig.suptitle("Reducing dimensions costs almost no accuracy; "
             "this dataset is too small to show a speed benefit")
fig.tight_layout()
save_and_show(fig, "09_components_vs_quality.png")


# ---------------------------------------------------------------------------
# 4. Where the clustering agrees and disagrees with the truth
# ---------------------------------------------------------------------------
pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X)
cluster_labels = results[1]["labels"]  # the PCA-95% pipeline

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=15, alpha=0.7)
axes[0].set_title("Coloured by true digit")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")

axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap="tab10",
                s=15, alpha=0.7)
axes[1].set_title(f"Coloured by KMeans cluster (ARI = {results[1]['ari']:.3f})")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")

fig.suptitle("Same projection, two labelings -- the colours differ because "
             "cluster IDs are arbitrary")
fig.tight_layout()
save_and_show(fig, "10_true_vs_cluster_labels.png")

# A contingency table shows which digits the clustering actually confuses.
# Rows are true digits, columns are cluster IDs. A clean result has one
# dominant entry per row; a split row means one digit landed in two clusters.
contingency = np.zeros((10, N_CLUSTERS), dtype=int)
for true_label, cluster in zip(y, cluster_labels):
    contingency[true_label, cluster] += 1

print("\n" + "-" * 78)
print("Contingency table: rows = true digit, columns = cluster ID")
print("-" * 78)
print("      " + "".join(f"{c:>6}" for c in range(N_CLUSTERS)))
for digit in range(10):
    row = contingency[digit]
    print(f"  {digit}: " + "".join(f"{v:>6}" for v in row))

print("\nPurity of each true digit (largest single cluster / total):")
for digit in range(10):
    row = contingency[digit]
    print(f"  digit {digit}: {row.max() / row.sum():.1%} "
          f"(n={row.sum()}, spread over {int((row > 0).sum())} clusters)")

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(contingency, cmap="Blues")
ax.set_xlabel("KMeans cluster ID")
ax.set_ylabel("True digit")
ax.set_xticks(range(N_CLUSTERS))
ax.set_yticks(range(10))
ax.set_title("Which digits end up in which clusters")
for i in range(10):
    for j in range(N_CLUSTERS):
        if contingency[i, j] > 0:
            ax.text(j, i, contingency[i, j], ha="center", va="center",
                    fontsize=8,
                    color="white" if contingency[i, j] > contingency.max() / 2 else "black")
fig.colorbar(im, ax=ax, label="images")
fig.tight_layout()
save_and_show(fig, "11_contingency_matrix.png")


# ---------------------------------------------------------------------------
# 5. Takeaways
# ---------------------------------------------------------------------------
raw_result, pca_result, scaled_result = results
purity = contingency.max(axis=1) / contingency.sum(axis=1)
worst_digit = int(np.argmin(purity))

# Pre-formatted so the paragraphs below wrap predictably instead of breaking
# wherever an interpolated number happens to end.
worst_purity = f"{purity[worst_digit]:.1%}"
worst_others = int((contingency[worst_digit] > 0).sum()) - 1
r_dims, p_dims, s_dims = (r["dims"] for r in results)
r_ari = f"{raw_result['ari']:.4f}"
p_ari = f"{pca_result['ari']:.4f}"
s_ari = f"{scaled_result['ari']:.4f}"
best_c = COMPONENT_COUNTS[best_sweep_idx]
best_a = f"{sweep_ari[best_sweep_idx]:.4f}"
last_a = f"{sweep_ari[-1]:.4f}"
ratio = f"{r_dims / p_dims:.1f}"
mean_t = f"{np.mean(sweep_time):.2f}"
n_samples = X.shape[0]

print("\n" + "=" * 78)
print("TAKEAWAYS")
print("=" * 78)
print(f"""
1. Order the pipeline scale -> PCA -> cluster. PCA maximizes variance, so a
   feature measured in a larger unit would dominate the components purely
   because of its unit. Scaling has to come first for PCA to measure structure
   rather than units. Scaling after PCA would re-inflate exactly the
   low-variance components PCA had just demoted.

2. But "always scale" is not a rule -- it is a decision about units. Here all
   64 features are pixel intensities on the same 0-16 scale. Standardizing
   them scored ARI {s_ari} against {p_ari} without it, and needed {s_dims}
   components to reach 95% variance instead of {p_dims} -- dividing the
   near-constant border pixels by their tiny standard deviation promotes
   noise into signal.

3. Compression was free, not beneficial -- read the numbers, not the hope.
   Going from {r_dims} to {p_dims} dimensions moved ARI from {r_ari} to
   {p_ari}, a change in the fourth decimal. The sweep is where the real
   signal is: ARI peaks at {best_c} components ({best_a}) and is *lower*
   with all 64 ({last_a}), so dropping the low-variance tail does remove
   some noise -- a small effect, but a real one. The honest summary is that
   PCA cost nothing here and bought a {ratio}x smaller representation.

   Fit time did not separate the pipelines: every run lands near {mean_t}s
   regardless of dimensionality, and the sweep times are not monotonic. At
   {n_samples} samples the work is dominated by fixed overhead, so this
   dataset cannot demonstrate PCA's speed benefit -- that argument needs
   data where the distance computation actually hurts.

4. ARI is a diagnostic, not an objective. Optimizing a clustering against
   known labels is just supervised learning with extra steps -- if the labels
   existed, you would train a classifier. ARI is used here only because the
   digits dataset happens to have ground truth, which lets us sanity-check
   that the clusters correspond to something real.

5. Clustering is not classification. The contingency table shows the failure
   mode plainly: KMeans has no concept of "digit". Digit {worst_digit} is the
   worst case here -- only {worst_purity} of its images land in a single
   cluster, with the rest scattered across {worst_others} others, because it
   is written in visually distinct ways that sit far apart in pixel space.
   Meanwhile one cluster can absorb two digits that happen to look alike.
   The algorithm finds compact regions in pixel space, and that is not the
   semantic partition.
""")
