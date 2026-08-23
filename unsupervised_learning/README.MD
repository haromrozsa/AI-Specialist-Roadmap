# ML Fundamentals – Unsupervised Learning (KMeans & PCA)

# Session Summary: Clustering Without Labels and Reducing Dimensions

Three scripts covering the two workhorse unsupervised algorithms: **KMeans** for
clustering and **PCA** for dimensionality reduction, then the two combined into
one pipeline.

The theme running through all three is that the standard recipes — "use the
elbow method", "keep 95% of the variance", "always StandardScaler first" — are
heuristics with failure modes, not rules. Each script measures the thing the
heuristic claims instead of asserting it.

| Script | Data | Covers |
|---|---|---|
| `KMeansClustering.py` | `make_blobs`, 500 samples, 2 features, true k = 4 | Elbow method, silhouette score, feature scaling, local minima |
| `PCAVarianceExplained.py` | `load_digits`, 1797 images, 64 pixel features | Scree plot, cumulative variance, components as images, reconstruction |
| `PCAKMeansPipeline.py` | `load_digits` | `scale → PCA → cluster` ordering, component sweep, ARI, contingency table |

Plots are written to `plots/` as well as shown, so the scripts produce the same
artifacts run interactively or headless.

## Running

```bash
cd unsupervised_learning
python KMeansClustering.py
python PCAVarianceExplained.py
python PCAKMeansPipeline.py
```

Requires `scikit-learn`, `numpy`, and `matplotlib`. No other dependencies, no
downloads — both datasets ship with scikit-learn.

## What I Did During the Session

1. **Built the k-selection comparison on synthetic data**
   - `make_blobs` is a deliberate choice: it has a *known* true k, so both
     selection methods can be checked against the right answer. No real dataset
     allows that — if you knew the number of clusters you would not be clustering
   - Swept k = 1..10 recording inertia and silhouette, and printed the
     per-step drop in inertia alongside it so the elbow is visible as numbers
     and not only as a curve

2. **Demonstrated the scaling failure mode**
   - Multiplied one feature by 50, as if it had been recorded in a different
     unit, leaving the cluster structure untouched
   - Clustered before and after `StandardScaler` and scored both with Adjusted
     Rand Index against the generated ground truth

3. **Exposed KMeans' dependence on initialization**
   - Ran 10 single-start fits (`n_init=1, init="random"`) and reported the
     spread in final inertia against the `n_init=10` + k-means++ default

4. **Measured PCA's variance spectrum on digits**
   - Reported components needed for 80 / 90 / 95 / 99% of variance
   - Rendered the first 11 components as 8×8 images — a component is a weight
     per pixel, so it can be looked at directly
   - Reconstructed digits from k ∈ {1, 2, 5, 10, 20, 40, 64} components via
     `inverse_transform`, with per-pixel reconstruction MSE for each

5. **Combined the two and tested the pipeline conventions**
   - Compared raw 64-D, `PCA(0.95)`, and `StandardScaler + PCA(0.95)` on ARI,
     silhouette, and fit time
   - Swept the component count from 2 to 64 to see the shape of the tradeoff
     rather than trusting one threshold
   - Built a contingency table of true digit vs. cluster ID to show *where*
     the clustering disagrees with the semantic labels

## Results

Environment for every number below: Python 3.8.8, scikit-learn 0.24.1,
numpy 1.22.4, matplotlib 3.4.1. All scripts use fixed seeds and reproduce
exactly, except the fit times in the pipeline sweep — see the caveat there.

### KMeans on synthetic blobs (500 samples, 2 features, true k = 4)

```
  k       Inertia   Drop vs k-1   Silhouette
  1       33903.9
  2       15737.1       18166.8       0.5955
  3        3426.3       12310.8       0.7613
  4         948.9        2477.4       0.7911   <- true k
  5         856.1          92.8       0.6646
  6         768.0          88.1       0.5360
  7         682.2          85.9       0.4313
  8         607.0          75.1       0.3384
  9         542.8          64.2       0.3521
 10         494.6          48.3       0.3598
```

This is the whole argument in one table. **Inertia is minimized at k=10** and
would keep falling to 0 at k=500 — it cannot select k. What it does show is the
*drop* column collapsing by a factor of 27 after k=4 (2477 → 93), which is the
elbow, read as numbers rather than eyeballed off a curve.

**Silhouette peaks at k=4** — the true value — and then falls, because past k=4
it starts splitting genuine clusters and the "distance to the nearest other
cluster" term collapses. At k=4 the recovered partition is perfect:
**ARI = 1.0000**.

Feature scaling, after multiplying one feature by 50 (a unit change that leaves
the structure untouched):

```
ARI on unscaled stretched data: 0.7728
ARI after StandardScaler:       1.0000
```

A pure change of units cost 23% of the agreement. Nothing about the data moved.

Initialization sensitivity, 10 single-start fits with random seeding:

```
Inertia range across seeds: 948.9 to 3344.4  (spread 2395.5)
n_init=10 with k-means++:   948.9
```

The worst random start lands **3.5x** off the best. sklearn's defaults hide
this completely.

### PCA on digits (1797 images, 64 pixel features)

Components required to reach each variance threshold:

```
  80% of variance: 13 of 64 components (20% of the original dimensions)
  90% of variance: 21 of 64 components (33% of the original dimensions)
  95% of variance: 29 of 64 components (45% of the original dimensions)
  99% of variance: 41 of 64 components (64% of the original dimensions)
```

The first components carry far more than their share, which is the entire
premise of dimensionality reduction:

```
  PC1   14.89%   cumulative 14.89%
  PC2   13.62%   cumulative 28.51%
  PC3   11.79%   cumulative 40.30%
  PC4    8.41%   cumulative 48.71%
  PC5    5.78%   cumulative 54.50%
```

Reconstruction quality against compression:

```
Components   Variance kept   Recon MSE/pixel   Compression
         1         14.89%           15.9777         64.0x
         2         28.51%           13.4210         32.0x
         5         54.50%            8.5424         12.8x
        10         73.82%            4.9143          6.4x
        20         89.43%            1.9844          3.2x
        40         98.82%            0.2215          1.6x
        64        100.00%            0.0000          1.0x
```

The interesting part of that table is the mismatch between the two middle
columns and what the images actually look like. At **10 components** the digit
is already clearly readable in `plots/07_reconstruction.png` — but only 74% of
the variance is kept, well short of the 95% that convention would demand. The
95% threshold is a default, not a finding.

At **64 components** reconstruction MSE is exactly 0. That is not a result, it
is arithmetic: with as many components as original dimensions PCA is a lossless
rotation of the data. Nothing has been reduced.

### PCA + KMeans on digits (k = 10)

```
Pipeline                                Dims      ARI   Silhouette   Fit (s)
Raw 64-D pixels                           64   0.6650       0.1824     0.192
PCA 95% (29-D)                            29   0.6639       0.1949     0.181
StandardScaler + PCA 95% (40-D)           40   0.4491       0.1518     0.210
```

Two findings, one of them against the conventional advice.

**PCA was free, not beneficial.** 64 → 29 dimensions moved ARI by 0.0011 — the
fourth decimal, i.e. nothing. The component sweep shows where the real (small)
effect is:

```
 Components       ARI
          2    0.3892
          5    0.5549
         10    0.6526
         15    0.6674
         20    0.6688
         30    0.6699   <- peak
         40    0.6673
         64    0.6638
```

Clustering in 30 dimensions beats clustering in all 64, so dropping the
low-variance tail does remove noise as well as size. It is a 0.006 effect —
real, but not the dramatic win the technique is often sold as. The honest
summary is that PCA cost nothing and bought a 2.2x smaller representation.

**`StandardScaler` actively hurt** — ARI 0.4491 against 0.6639 without it, a
32% relative drop, and it *increased* the components needed for 95% variance
from 29 to 40. All 64 features are pixel intensities on one 0–16 scale, and 3
are constant across the dataset, so dividing each pixel by its own standard
deviation promotes near-empty border pixels to the weight of informative center
ones. "Scale first" is a decision about units, not a reflex.

Cluster purity per true digit, from the contingency table:

```
digit 0: 99.4% over 2 clusters      digit 5: 75.3% over 5 clusters
digit 1: 54.4% over 6 clusters      digit 6: 97.8% over 4 clusters
digit 2: 83.6% over 7 clusters      digit 7: 95.0% over 4 clusters
digit 3: 85.2% over 5 clusters      digit 8: 56.9% over 8 clusters
digit 4: 89.5% over 4 clusters      digit 9: 77.2% over 5 clusters
```

Digits 0, 6, and 7 are nearly clean. Digits 1 and 8 fall apart — barely half
their images share a cluster. That spread is the difference between "compact in
pixel space" and "the same digit", which no amount of tuning removes.

**Timing caveat:** the fit times above do not separate the pipelines. Every
configuration lands near 0.18s regardless of dimensionality, and the sweep times
are not even monotonic. At 1797 samples the run is dominated by fixed overhead,
so this dataset cannot demonstrate PCA's speed benefit — that argument needs
data where the distance computation actually hurts. The timing panel in
`plots/09_components_vs_quality.png` is included as evidence of the non-result,
not as a trend.

### Plots produced

| File | Shows |
|---|---|
| `01_ground_truth.png` | The generated blobs with their true labels |
| `02_elbow_and_silhouette.png` | Inertia falling forever vs. silhouette peaking |
| `03_clusters_with_centroids.png` | Final KMeans assignment and centroid positions |
| `04_scaling_effect.png` | Same data clustered before and after scaling |
| `05_scree_and_cumulative.png` | Scree plot and cumulative variance with thresholds |
| `06_components_as_images.png` | The mean image and the first 11 components as 8×8 images |
| `07_reconstruction.png` | Four digits reconstructed at seven compression levels |
| `08_two_component_projection.png` | Digits projected to 2-D, coloured by true class |
| `09_components_vs_quality.png` | ARI and fit time against component count |
| `10_true_vs_cluster_labels.png` | The same projection under true vs. cluster labels |
| `11_contingency_matrix.png` | Which digits land in which clusters |

## What I Learned

1. **Inertia cannot be used to select k, and the elbow is not an optimum**

   Inertia is the sum of squared distances from each point to its centroid. It
   decreases monotonically in k and reaches exactly 0 when k equals the number
   of samples, because every point becomes its own centroid. So "minimize
   inertia" has a trivial, useless answer. The elbow is a *visual* judgement
   about where the marginal gain collapses — genuinely useful, but a heuristic
   read off a curve, and ambiguous the moment the curve is smooth.

2. **Silhouette has a real optimum, and that is why it is the better criterion**

   For each point it compares the mean distance to its own cluster (a) against
   the mean distance to the nearest other cluster (b), scoring `(b − a) / max(a, b)`.
   Because it penalizes both loose clusters *and* clusters that sit too close to
   each other, it does not improve without bound as k grows — it turns around.
   On the blobs it turned around at exactly the true k=4, which is the check
   that synthetic data makes possible. The cost is that it is O(n²) in pairwise
   distances, which is why the elbow survives on large datasets.

3. **KMeans is distance-based, so units are part of the model**

   Squared Euclidean distance means a feature with a wider numeric range
   dominates the objective. Re-expressing one feature in different units — the
   same information, a different scale — changes the clustering. This is not a
   subtle degradation: ARI fell from 1.0000 to 0.7728 purely because one feature
   was multiplied by 50. The centroids reorganize to split the large-range axis.
   The fix is to scale first unless every feature already shares a unit.

4. **KMeans converges to a local minimum, and `n_init` hides that**

   The algorithm alternates assign-then-update until nothing moves, which
   guarantees convergence but only to a local optimum determined by the
   initialization. sklearn's defaults (`n_init=10` plus k-means++ seeding, which
   spreads initial centroids apart) make it look deterministic. Running with
   `n_init=1, init="random"` across seeds exposes the spread that is actually
   there: the worst seed converged to inertia 3344.4 against 948.9 for the best,
   a 3.5x difference on data with an unambiguous correct answer.

5. **A principal component is not a feature**

   Each component is a linear combination of all 64 pixels with signed weights.
   Rendering them as images makes this concrete: they are not digits, they are
   the directions along which the dataset varies most, and every image is
   reconstructed as the mean image plus a weighted sum of them. `PC1 = 14.89%`
   says how much variation the direction captures, not what it means. Losing
   the ability to say "this feature mattered" is the real price of PCA, and it
   is why PCA and interpretability requirements pull against each other.

6. **"Variance explained" and "useful" are different quantities**

   Reconstruction at 10 components looks right to a human at 74% variance.
   Whether 10 is the correct choice depends on the downstream task, which means
   the component count is a hyperparameter to tune against that task, not a
   threshold to read off a cumulative curve.

7. **Scale before PCA — but only when scaling is warranted at all**

   The ordering `scale → PCA → cluster` is forced: PCA maximizes variance, so a
   feature in a larger unit would dominate the components purely because of its
   unit. Scaling *after* PCA would be worse than useless, re-inflating the
   low-variance components PCA had just demoted.

   But the digits dataset shows the ordering rule and the scaling decision are
   separate questions. All 64 features are pixel intensities on one 0–16 scale,
   and 3 of them are constant across the entire dataset. `StandardScaler`
   divides each pixel by its own standard deviation, which promotes
   near-constant border pixels — pure noise — to the same weight as the
   informative center pixels. Measured cost: ARI 0.4491 with standardization
   against 0.6639 without, and 40 components needed for 95% variance instead of
   29. Standardize when features have different *units*, not reflexively.

8. **ARI is a diagnostic, not an objective**

   Adjusted Rand Index compares two partitions while ignoring label names, so
   cluster "0" matching true digit "7" costs nothing; it is corrected for chance,
   so random assignment scores ≈ 0 rather than something misleadingly positive.
   It is used here only because digits happens to carry ground truth. Tuning a
   clustering to maximize agreement with known labels is supervised learning
   with extra steps — if the labels existed, you would train a classifier.

9. **Clustering is not classification, and the contingency table shows why**

   KMeans has no concept of "digit". It finds compact regions in pixel space,
   and that partition is not the semantic one. Digit 1 is the clearest case:
   only 54.4% of its images share a cluster, the rest scattered across 5 more,
   because a 1 is written in visually distinct ways that sit far apart in pixel
   space. Digit 8 is nearly as bad at 56.9%, while digits 0, 6, and 7 come out
   above 95%. No hyperparameter fixes this — it is what the objective optimizes.
   Cluster IDs are also arbitrary — the two panels of
   `plots/10_true_vs_cluster_labels.png` use different colours for the same
   grouping, which is exactly the label-permutation problem ARI exists to
   correct for.

## What Was Deliberately Left Out

This is a teaching demo, so scope was cut to keep the concepts and drop the
plumbing:

- **Other clustering families** — DBSCAN, hierarchical, and spectral clustering
  would contrast usefully with KMeans' spherical-cluster assumption, but each is
  its own topic
- **Non-linear dimensionality reduction** — t-SNE and UMAP produce better 2-D
  pictures than PCA but are visualization tools, not invertible transforms, and
  they cannot do the reconstruction that makes the PCA lesson land
- **Tests, type hints, structured logging** — per the teaching-demo standard in
  the repository's `CLAUDE.md`
