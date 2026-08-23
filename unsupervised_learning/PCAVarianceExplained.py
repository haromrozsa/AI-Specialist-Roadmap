"""
PCA on the digits dataset: variance explained, the scree plot, what the
principal components actually look like, and what reconstruction loses.

The digits dataset is 8x8 grayscale images flattened to 64 features. That is a
good fit for this demo because the features are pixels: we can render a
component as an image, and we can render a reconstruction as an image. It turns
"95% of the variance is explained" from a number into something you can look at.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(42)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_and_show(fig, filename):
    """Write the figure to plots/ and then display it."""
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=120, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------------
digits = load_digits()
X = digits.data          # (1797, 64), pixel intensities in [0, 16]
y = digits.target        # only used for labelling plots, never for fitting
n_samples, n_features = X.shape

print("=" * 70)
print("PCA -- VARIANCE EXPLAINED")
print("=" * 70)
print(f"Dataset shape: {X.shape}  ({n_samples} images, 8x8 = {n_features} pixels)")
print(f"Pixel value range: [{X.min():.0f}, {X.max():.0f}]")

# Note on scaling: every feature here is a pixel intensity on the same 0-16
# scale, so StandardScaler is the wrong move. Dividing each pixel by its own
# standard deviation would inflate the near-constant border pixels -- which are
# almost always 0 and carry no information -- up to the same weight as the
# informative center pixels. PCA centers the data itself; that is all this
# dataset needs. Standardize when features have different *units*, not reflexively.
zero_variance_pixels = int((X.std(axis=0) == 0).sum())
print(f"Pixels with zero variance across the dataset: {zero_variance_pixels}")


# ---------------------------------------------------------------------------
# 2. Fit PCA with all components and read the variance spectrum
# ---------------------------------------------------------------------------
pca_full = PCA(n_components=n_features, random_state=42)
pca_full.fit(X)

ratios = pca_full.explained_variance_ratio_
cumulative = np.cumsum(ratios)

print("\n" + "-" * 70)
print("Components needed to reach a variance threshold")
print("-" * 70)
for threshold in (0.80, 0.90, 0.95, 0.99):
    # searchsorted finds the first index where cumulative >= threshold; +1
    # converts a 0-based index into a component count.
    n_needed = int(np.searchsorted(cumulative, threshold)) + 1
    print(f"  {threshold:.0%} of variance: {n_needed:>2} of {n_features} components "
          f"({n_needed / n_features:.0%} of the original dimensions)")

print("\nFirst 10 components individually:")
for i in range(10):
    print(f"  PC{i + 1:<2}  {ratios[i]:6.2%}   cumulative {cumulative[i]:6.2%}")


# ---------------------------------------------------------------------------
# 3. Scree plot and cumulative variance
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].bar(range(1, 21), ratios[:20], color="steelblue")
axes[0].set_title("Scree plot: variance explained per component")
axes[0].set_xlabel("Principal component")
axes[0].set_ylabel("Proportion of variance explained")
axes[0].set_xticks(range(1, 21, 2))
axes[0].grid(alpha=0.3, axis="y")

axes[1].plot(range(1, n_features + 1), cumulative, "-", color="darkorange", linewidth=2)
for threshold, color in ((0.80, "green"), (0.90, "purple"), (0.95, "red")):
    n_needed = int(np.searchsorted(cumulative, threshold)) + 1
    axes[1].axhline(threshold, color=color, linestyle="--", alpha=0.6)
    axes[1].axvline(n_needed, color=color, linestyle=":", alpha=0.6)
    axes[1].annotate(f"{threshold:.0%} @ {n_needed} PCs",
                     xy=(n_needed, threshold), xytext=(n_needed + 3, threshold - 0.06),
                     color=color, fontsize=9)
axes[1].set_title("Cumulative variance explained")
axes[1].set_xlabel("Number of components")
axes[1].set_ylabel("Cumulative proportion of variance")
axes[1].set_ylim(0, 1.02)
axes[1].grid(alpha=0.3)

fig.suptitle("The variance is concentrated in the first few components")
fig.tight_layout()
save_and_show(fig, "05_scree_and_cumulative.png")


# ---------------------------------------------------------------------------
# 4. What a principal component actually is
# ---------------------------------------------------------------------------
# Each component is a 64-length vector -- one weight per pixel -- so it can be
# reshaped back to 8x8 and viewed as an image. These are not digits. They are
# the directions of greatest variation, and every image is reconstructed as the
# mean image plus a weighted sum of these.
fig, axes = plt.subplots(2, 6, figsize=(13, 5))

axes[0, 0].imshow(pca_full.mean_.reshape(8, 8), cmap="gray")
axes[0, 0].set_title("mean image", fontsize=10)
axes[0, 0].axis("off")

for i, ax in enumerate(axes.ravel()[1:]):
    ax.imshow(pca_full.components_[i].reshape(8, 8), cmap="RdBu_r")
    ax.set_title(f"PC{i + 1}\n{ratios[i]:.1%}", fontsize=10)
    ax.axis("off")

fig.suptitle("Principal components rendered as images "
             "(red = positive weight, blue = negative)")
fig.tight_layout()
save_and_show(fig, "06_components_as_images.png")


# ---------------------------------------------------------------------------
# 5. Reconstruction: what is actually lost at each compression level
# ---------------------------------------------------------------------------
# inverse_transform projects back from the reduced space into the original
# 64-dimensional pixel space. The result is the closest possible approximation
# of the original image using only that many components.
COMPONENT_COUNTS = [1, 2, 5, 10, 20, 40, 64]
SAMPLE_INDICES = [0, 1, 2, 3]  # digits 0, 1, 2, 3

reconstructions = {}
recon_errors = {}

for k in COMPONENT_COUNTS:
    pca_k = PCA(n_components=k, random_state=42)
    X_reduced = pca_k.fit_transform(X)
    X_reconstructed = pca_k.inverse_transform(X_reduced)
    reconstructions[k] = X_reconstructed
    # Mean squared error per pixel, averaged over the whole dataset.
    recon_errors[k] = float(np.mean((X - X_reconstructed) ** 2))

print("\n" + "-" * 70)
print(f"{'Components':>10}  {'Variance kept':>14}  {'Recon MSE/pixel':>16}  {'Compression':>12}")
print("-" * 70)
for k in COMPONENT_COUNTS:
    print(f"{k:>10}  {cumulative[k - 1]:>13.2%}  {recon_errors[k]:>16.4f}  "
          f"{n_features / k:>11.1f}x")

fig, axes = plt.subplots(len(SAMPLE_INDICES), len(COMPONENT_COUNTS) + 1,
                         figsize=(14, 8))

for row, idx in enumerate(SAMPLE_INDICES):
    axes[row, 0].imshow(X[idx].reshape(8, 8), cmap="gray")
    axes[row, 0].axis("off")
    if row == 0:
        axes[row, 0].set_title("original", fontsize=10)

    for col, k in enumerate(COMPONENT_COUNTS, start=1):
        axes[row, col].imshow(reconstructions[k][idx].reshape(8, 8), cmap="gray")
        axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(f"{k} PC\n{cumulative[k - 1]:.0%}", fontsize=10)

fig.suptitle("Reconstruction from k components -- the digit becomes readable "
             "long before 100% of the variance is kept")
fig.tight_layout()
save_and_show(fig, "07_reconstruction.png")


# ---------------------------------------------------------------------------
# 6. Projecting onto two components for visualization
# ---------------------------------------------------------------------------
# Two components keep only a fraction of the variance, which is why this plot
# is a rough map and not a classifier. Some digit classes separate cleanly;
# others (4/7/9) overlap heavily, and that overlap is real information loss,
# not a plotting artifact.
pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X)

fig, ax = plt.subplots(figsize=(9, 7))
scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=15, alpha=0.7)
ax.set_title(f"Digits projected onto 2 components "
             f"({pca_2d.explained_variance_ratio_.sum():.1%} of variance)")
ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})")
fig.colorbar(scatter, ax=ax, label="true digit", ticks=range(10))
fig.tight_layout()
save_and_show(fig, "08_two_component_projection.png")


# ---------------------------------------------------------------------------
# 7. Takeaways
# ---------------------------------------------------------------------------
n_95 = int(np.searchsorted(cumulative, 0.95)) + 1
print("\n" + "=" * 70)
print("TAKEAWAYS")
print("=" * 70)
print(f"""
1. PCA finds orthogonal directions of maximum variance, ordered. On digits,
   {n_95} of {n_features} components carry 95% of the variance -- a
   {n_features / n_95:.1f}x reduction with most of the signal intact.

2. explained_variance_ratio_ is the honest measure of what a component is
   worth. The scree plot's steep drop is why dimensionality reduction works at
   all: the tail components mostly encode noise and near-constant pixels.

3. A component is not a feature. It is a weighted combination of all
   {n_features} pixels, and its weights can be negative. "PC1 = {ratios[0]:.2%}"
   tells you how much variation it captures, not what it means. This is the
   real cost of PCA: you trade interpretability for compactness.

4. Variance explained is not the same as usefulness. Reconstruction at 10
   components already looks like the right digit to a human, well before the
   95% threshold. Pick the component count by measuring the downstream task,
   not by defaulting to 95%.

5. Do not standardize reflexively. These pixels share one unit and one scale,
   and {zero_variance_pixels} of them are constant across the dataset --
   StandardScaler would either divide by zero variance or amplify near-empty
   border pixels to the weight of informative ones. PCA already centers.
   Standardize when features have different units.
""")
