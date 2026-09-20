"""
k-Nearest Neighbours: the model that does no work until you ask it a question.

One script, seven figures, five questions:

1. What does k actually control? Drawn, by plotting the decision boundary at
   k = 1, 5, 15 and 50 on the same 2-D data, where k=1 is a memorised map of the
   training set and k=50 is nearly a straight line.
2. Where is the bias/variance trade-off in k? Swept, k = 1..100, with training,
   test and cross-validated accuracy on one axis. k-NN is the rare model whose
   complexity knob runs *backwards*: small k is the complex model.
3. Does feature scaling matter? Measured, by multiplying one feature by 1000 and
   watching the classifier collapse. k-NN is a pure distance model, so this is
   the most extreme case of it in the repository.
4. What does `weights="distance"` buy? Compared against `uniform` across the
   same k sweep.
5. Why is k-NN called a lazy learner, and what does that cost? Timed, by fitting
   and predicting at growing training-set sizes so the cost that moves from
   training to inference is a number rather than a phrase.

Then the same model is put on the Titanic split that
logistic_regression/TitanicDatasetLogisticRegression.py, trees_and_boosting/ and
shap_explainability/ all use (test_size=0.2, stratify, random_state=42), so its
accuracy is directly comparable to the numbers already recorded in this
repository rather than being a lone figure with nothing to sit next to.

The 2-D work uses make_moons for the same reason svm_kernels/ does: a decision
boundary you cannot draw teaches nothing about a model whose entire definition
is "which training points are near".
"""

import os
import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_moons
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Set random seed for reproducibility
np.random.seed(42)
RANDOM_STATE = 42

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_and_show(fig, filename):
    """Write the figure to plots/ and then display it.

    Saving before showing means the script produces the same artifacts whether
    it is run interactively or headless (MPLBACKEND=Agg), where show() is a no-op.
    """
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=120, bbox_inches="tight")
    plt.show()


def plot_boundary(ax, model, X, y, title, resolution=160):
    """Shade the mesh by predicted class, then scatter the points on top.

    sklearn 0.24 has no DecisionBoundaryDisplay, so the mesh is built by hand.
    It also makes the mechanic explicit: every pixel below is a full k-NN query
    against the training set, which is exactly why the timing section matters --
    and why `resolution` is 160 and not 300. At 300 this is 90,000 queries per
    panel, and on the environment these figures were generated on that is ~40
    seconds of wall clock for a single subplot.
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm", levels=1)
    ax.contour(xx, yy, Z, colors="k", linewidths=0.8, levels=[0.5])
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=25)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# 0. The 2-D dataset every geometric figure below is drawn on
# ---------------------------------------------------------------------------
X2d, y2d = make_moons(n_samples=300, noise=0.25, random_state=RANDOM_STATE)
X2d_train, X2d_test, y2d_train, y2d_test = train_test_split(
    X2d, y2d, test_size=0.3, stratify=y2d, random_state=RANDOM_STATE
)

print("=" * 78)
print("k-NEAREST NEIGHBOURS")
print("=" * 78)
print(f"\n2-D data: make_moons  n={len(X2d)}  noise=0.25")
print(f"Train: {X2d_train.shape}   Test: {X2d_test.shape}")

# ---------------------------------------------------------------------------
# 1. What k does to the boundary
# ---------------------------------------------------------------------------
# k-NN has no parameters to fit. "Training" stores the data; all of the work
# happens at predict time, and k is the only thing deciding how much of the
# neighbourhood gets a vote.
print("\n" + "-" * 78)
print("1. k and the shape of the boundary")
print("-" * 78)

K_PANELS = [1, 5, 15, 50]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
for ax, k in zip(axes, K_PANELS):
    knn = KNeighborsClassifier(n_neighbors=k).fit(X2d_train, y2d_train)
    train_acc = accuracy_score(y2d_train, knn.predict(X2d_train))
    test_acc = accuracy_score(y2d_test, knn.predict(X2d_test))
    plot_boundary(
        ax,
        knn,
        X2d_train,
        y2d_train,
        f"k = {k}\ntrain {train_acc:.3f}   test {test_acc:.3f}",
    )
    print(f"  k={k:3d}   train {train_acc:.4f}   test {test_acc:.4f}")

fig.suptitle(
    "k is a smoothing knob, and it runs backwards: k=1 is the COMPLEX model\n"
    "k=1 memorises every training point (train accuracy 1.000 by construction); "
    "k=50 averages the neighbourhood into an almost straight cut.",
    fontsize=12,
)
fig.tight_layout()
save_and_show(fig, "01_k_boundary_grid.png")

# The k=1 panel is the one to stare at. Its training accuracy is 1.000 and it
# always will be: the nearest neighbour of a training point is itself, at
# distance zero. A training score of 1.000 here measures nothing at all.

# ---------------------------------------------------------------------------
# 2. The bias/variance sweep
# ---------------------------------------------------------------------------
print("\n" + "-" * 78)
print("2. k sweep: train vs test vs 5-fold CV")
print("-" * 78)

k_values = list(range(1, 101))
train_scores, test_scores, cv_scores = [], [], []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X2d_train, y2d_train)
    train_scores.append(accuracy_score(y2d_train, knn.predict(X2d_train)))
    test_scores.append(accuracy_score(y2d_test, knn.predict(X2d_test)))
    cv_scores.append(
        cross_val_score(
            KNeighborsClassifier(n_neighbors=k), X2d_train, y2d_train, cv=5
        ).mean()
    )

best_cv_k = k_values[int(np.argmax(cv_scores))]
print(f"  Best k by 5-fold CV on the training split: k={best_cv_k} "
      f"(CV {max(cv_scores):.4f})")
print(f"  Its test accuracy: {test_scores[k_values.index(best_cv_k)]:.4f}")
print(f"  k=1  train {train_scores[0]:.4f}  test {test_scores[0]:.4f}  "
      f"CV {cv_scores[0]:.4f}")
print(f"  k=100 train {train_scores[-1]:.4f}  test {test_scores[-1]:.4f}  "
      f"CV {cv_scores[-1]:.4f}   (n_train={len(X2d_train)})")

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(k_values, train_scores, label="train", lw=2)
ax.plot(k_values, test_scores, label="test", lw=2)
ax.plot(k_values, cv_scores, label="5-fold CV (train split only)", lw=2, ls="--")
ax.axvline(best_cv_k, color="k", ls=":", lw=1.5,
           label=f"CV-selected k = {best_cv_k}")
ax.set_xlabel("k  (number of neighbours voting)")
ax.set_ylabel("accuracy")
ax.set_title(
    "Model complexity decreases left to right.\n"
    "Train accuracy starts at 1.000 for free; only the CV curve is informative.",
    fontsize=12,
)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
save_and_show(fig, "02_k_sweep.png")

# ---------------------------------------------------------------------------
# 3. Feature scaling, broken on purpose
# ---------------------------------------------------------------------------
# Every model that measures distance inherits the units of its features. k-NN is
# the purest case: the neighbour set itself changes when one axis is stretched,
# so the model does not degrade gracefully, it looks at different points.
print("\n" + "-" * 78)
print("3. Feature scaling: one feature multiplied by 1000")
print("-" * 78)

SCALE_FACTOR = 1000.0
X2d_train_broken = X2d_train.copy()
X2d_test_broken = X2d_test.copy()
X2d_train_broken[:, 1] *= SCALE_FACTOR
X2d_test_broken[:, 1] *= SCALE_FACTOR

k_fixed = 15
raw_ok = KNeighborsClassifier(n_neighbors=k_fixed).fit(X2d_train, y2d_train)
raw_broken = KNeighborsClassifier(n_neighbors=k_fixed).fit(
    X2d_train_broken, y2d_train
)
scaled_broken = Pipeline(
    [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k_fixed))]
).fit(X2d_train_broken, y2d_train)

acc_raw_ok = accuracy_score(y2d_test, raw_ok.predict(X2d_test))
acc_raw_broken = accuracy_score(y2d_test, raw_broken.predict(X2d_test_broken))
acc_scaled_broken = accuracy_score(y2d_test, scaled_broken.predict(X2d_test_broken))

print(f"  k={k_fixed}, comparable feature scales, no scaler : {acc_raw_ok:.4f}")
print(f"  k={k_fixed}, feature 2 x{SCALE_FACTOR:.0f}, no scaler  : "
      f"{acc_raw_broken:.4f}   (drop {acc_raw_ok - acc_raw_broken:+.4f})")
print(f"  k={k_fixed}, feature 2 x{SCALE_FACTOR:.0f}, StandardScaler: "
      f"{acc_scaled_broken:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
plot_boundary(axes[0], raw_ok, X2d_train, y2d_train,
              f"Comparable scales, no scaler\ntest {acc_raw_ok:.3f}")
plot_boundary(axes[1], raw_broken, X2d_train_broken, y2d_train,
              f"Feature 2 x{SCALE_FACTOR:.0f}, no scaler\ntest {acc_raw_broken:.3f}")
plot_boundary(axes[2], scaled_broken, X2d_train_broken, y2d_train,
              f"Feature 2 x{SCALE_FACTOR:.0f}, StandardScaler\n"
              f"test {acc_scaled_broken:.3f}")
fig.suptitle(
    "The middle panel is not a worse boundary, it is a boundary in a different space.\n"
    "One axis 1000x larger means the other contributes ~nothing to the distance, "
    "so the model reads horizontal stripes.",
    fontsize=12,
)
fig.tight_layout()
save_and_show(fig, "03_scaling_matters.png")

# The scaler lives inside a Pipeline, so under cross-validation the mean and
# standard deviation are recomputed from each fold's training rows only. Fitting
# one scaler on the whole training set before cross-validating leaks the fold's
# own distribution into its preprocessing.

# ---------------------------------------------------------------------------
# 4. uniform vs distance weighting
# ---------------------------------------------------------------------------
# weights="distance" gives closer neighbours a larger vote (1/d). One consequence
# is structural: with distance weighting the nearest neighbour of a training
# point is itself at d=0, which sklearn treats as infinite weight, so training
# accuracy is pinned at 1.000 for *every* k, not just k=1.
print("\n" + "-" * 78)
print("4. weights='uniform' vs weights='distance'")
print("-" * 78)

k_sub = list(range(1, 61))
uni_test, dist_test, uni_train, dist_train = [], [], [], []
for k in k_sub:
    u = KNeighborsClassifier(n_neighbors=k, weights="uniform").fit(
        X2d_train, y2d_train)
    d = KNeighborsClassifier(n_neighbors=k, weights="distance").fit(
        X2d_train, y2d_train)
    uni_test.append(accuracy_score(y2d_test, u.predict(X2d_test)))
    dist_test.append(accuracy_score(y2d_test, d.predict(X2d_test)))
    uni_train.append(accuracy_score(y2d_train, u.predict(X2d_train)))
    dist_train.append(accuracy_score(y2d_train, d.predict(X2d_train)))

print(f"  uniform : best test {max(uni_test):.4f} at k="
      f"{k_sub[int(np.argmax(uni_test))]}")
print(f"  distance: best test {max(dist_test):.4f} at k="
      f"{k_sub[int(np.argmax(dist_test))]}")
print(f"  distance training accuracy is {min(dist_train):.4f}-"
      f"{max(dist_train):.4f} across all k (self-distance 0 -> infinite weight)")
print(f"  mean test difference (distance - uniform): "
      f"{np.mean(np.array(dist_test) - np.array(uni_test)):+.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(k_sub, uni_train, label="uniform", lw=2)
axes[0].plot(k_sub, dist_train, label="distance", lw=2)
axes[0].set_title("Training accuracy\n'distance' is pinned at 1.000 for every k")
axes[1].plot(k_sub, uni_test, label="uniform", lw=2)
axes[1].plot(k_sub, dist_test, label="distance", lw=2)
axes[1].set_title("Test accuracy\nthe part that actually decides anything")
for ax in axes:
    ax.set_xlabel("k")
    ax.set_ylabel("accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
fig.suptitle(
    "Distance weighting never lets the training score fall, which is exactly why "
    "a training score cannot be used to pick k.",
    fontsize=12,
)
fig.tight_layout()
save_and_show(fig, "04_weighting.png")

# ---------------------------------------------------------------------------
# 5. The lazy learner's bill, timed
# ---------------------------------------------------------------------------
# The standard description is that k-NN does no work at training time and pays
# for it at inference. That is true of the algorithm, and false of sklearn's
# default: `algorithm="auto"` picks a kd_tree on low-dimensional data, and
# building that tree is real work that scales with n. So both algorithms are
# timed here rather than only the default, because the textbook sentence and the
# library's behaviour do not agree.
print("\n" + "-" * 78)
print("5. Lazy learning: where the time actually goes")
print("-" * 78)

sizes = [500, 2_000, 8_000, 32_000, 128_000]
X_query, _ = make_moons(n_samples=1_000, noise=0.25, random_state=7)
timings = {}

for algo in ["kd_tree", "brute"]:
    fit_times, predict_times = [], []
    for n in sizes:
        Xb, yb = make_moons(n_samples=n, noise=0.25, random_state=RANDOM_STATE)
        knn = KNeighborsClassifier(n_neighbors=15, algorithm=algo)
        t0 = time.perf_counter()
        knn.fit(Xb, yb)
        fit_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        knn.predict(X_query)
        predict_times.append(time.perf_counter() - t0)
        print(f"  {algo:8s} n_train={n:7,d}   fit {fit_times[-1]*1000:8.2f} ms   "
              f"predict(1000 rows) {predict_times[-1]*1000:9.2f} ms")
    timings[algo] = (fit_times, predict_times)

print(f"\n  kd_tree fit  grew {timings['kd_tree'][0][-1] / timings['kd_tree'][0][0]:.0f}x "
      f"from n=500 to n=128,000 -- the 'lazy' learner is not lazy by default.")
print(f"  kd_tree predict grew "
      f"{timings['kd_tree'][1][-1] / timings['kd_tree'][1][0]:.2f}x over the same range.")
print(f"  brute   predict grew "
      f"{timings['brute'][1][-1] / timings['brute'][1][0]:.1f}x -- this is the linear "
      f"scan the textbook describes.")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for algo, color in [("kd_tree", "steelblue"), ("brute", "crimson")]:
    f, p = timings[algo]
    axes[0].plot(sizes, np.array(f) * 1000, "o-", lw=2, color=color, label=algo)
    axes[1].plot(sizes, np.array(p) * 1000, "o-", lw=2, color=color, label=algo)
axes[0].set_title("fit()\nkd_tree builds an index; brute just stores the array")
axes[1].set_title("predict() on 1000 rows\nthe index is what keeps this flat")
for ax in axes:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training rows")
    ax.set_ylabel("milliseconds (log scale)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
fig.suptitle(
    "'k-NN does no work at training time' is a statement about the algorithm, "
    "not about sklearn's default.\n"
    "algorithm='auto' chose kd_tree here, which moves the cost back to fit().",
    fontsize=12,
)
fig.tight_layout()
save_and_show(fig, "05_lazy_learner_timing.png")

# Caveat on the absolute numbers: on the machine these were measured on, a
# predict() call carries roughly 0.45 ms of fixed per-query-row overhead, so the
# kd_tree predict curve is sitting on a floor rather than showing the index's
# true asymptotic cost. The shapes -- kd_tree fit growing, brute predict growing
# linearly -- are the part that transfers; the milliseconds are not.

# ---------------------------------------------------------------------------
# 6. The curse of dimensionality, measured rather than asserted
# ---------------------------------------------------------------------------
# "k-NN suffers in high dimensions" is repeated everywhere. The mechanism is that
# distances concentrate: as dimensions are added, the nearest and the farthest
# training point end up almost equally far away, so "nearest" stops meaning
# anything. Both halves are measured below, on data where the two real features
# never change and every added column is pure noise.
print("\n" + "-" * 78)
print("6. Curse of dimensionality: 2 real features + d noise columns")
print("-" * 78)

noise_dims = [0, 2, 5, 10, 20, 50, 100, 200]
cod_acc, dist_ratio = [], []
rng = np.random.RandomState(RANDOM_STATE)

for d in noise_dims:
    if d == 0:
        Xtr, Xte = X2d_train, X2d_test
    else:
        Xtr = np.hstack([X2d_train, rng.randn(len(X2d_train), d)])
        Xte = np.hstack([X2d_test, rng.randn(len(X2d_test), d)])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=15)),
    ]).fit(Xtr, y2d_train)
    cod_acc.append(accuracy_score(y2d_test, pipe.predict(Xte)))

    # Distance concentration: for each test point, nearest / farthest training
    # distance. It starts near 0 and climbs towards 1 as dimensions are added.
    Xtr_s = StandardScaler().fit_transform(Xtr)
    Xte_s = StandardScaler().fit(Xtr).transform(Xte)
    dmat = np.linalg.norm(Xte_s[:, None, :] - Xtr_s[None, :, :], axis=2)
    dist_ratio.append(np.mean(dmat.min(axis=1) / dmat.max(axis=1)))
    print(f"  +{d:3d} noise dims   test accuracy {cod_acc[-1]:.4f}   "
          f"mean(nearest/farthest) {dist_ratio[-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(noise_dims, cod_acc, "o-", lw=2, color="crimson")
axes[0].axhline(0.5, color="k", ls=":", lw=1.5, label="coin flip")
axes[0].set_xlabel("noise dimensions added")
axes[0].set_ylabel("test accuracy")
axes[0].set_title("The two informative features never change.\n"
                  "Only noise is added, and accuracy still collapses.")
axes[0].legend()
axes[1].plot(noise_dims, dist_ratio, "o-", lw=2, color="darkslateblue")
axes[1].set_xlabel("noise dimensions added")
axes[1].set_ylabel("mean  nearest distance / farthest distance")
axes[1].set_ylim(0, 1)
axes[1].set_title("Why: distances concentrate.\n"
                  "As this ratio approaches 1, 'nearest' stops being a signal.")
for ax in axes:
    ax.grid(alpha=0.3)
fig.suptitle(
    "The curse of dimensionality is not about accuracy directly, it is about the "
    "distance metric losing contrast.",
    fontsize=12,
)
fig.tight_layout()
save_and_show(fig, "06_curse_of_dimensionality.png")

# ---------------------------------------------------------------------------
# 7. Titanic: the same split as the rest of the repository
# ---------------------------------------------------------------------------
# Everything above is on synthetic 2-D data. To place k-NN next to the logistic
# regression, tree, forest and boosting numbers already recorded here, it has to
# run on the same rows, the same seven features and the same seed.
print("\n" + "-" * 78)
print("7. Titanic (same split as logistic_regression/ and trees_and_boosting/)")
print("-" * 78)

titanic = sns.load_dataset("titanic")
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
NUMERICAL = ["age", "sibsp", "parch", "fare"]
CATEGORICAL = ["pclass", "sex", "embarked"]

X = titanic[FEATURES]
y = titanic["survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"  Train: {X_train.shape}   Test: {X_test.shape}")

# StandardScaler is inside the preprocessor here, unlike in the tree scripts that
# share this split. A tree does not care about units; k-NN cannot work without
# them being comparable, as section 3 measured.
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            NUMERICAL,
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="first", handle_unknown="error")),
            ]),
            CATEGORICAL,
        ),
    ]
)

knn_pipe = Pipeline([("prep", preprocessor), ("knn", KNeighborsClassifier())])
grid = GridSearchCV(
    knn_pipe,
    {
        "knn__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31, 51],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],  # Manhattan vs Euclidean
    },
    cv=5,
    scoring="accuracy",
    n_jobs=1,  # spawning workers is slower than the fits here
)
grid.fit(X_train, y_train)

best = grid.best_estimator_
titanic_acc = accuracy_score(y_test, best.predict(X_test))
titanic_auc = roc_auc_score(y_test, best.predict_proba(X_test)[:, 1])

print(f"  Best params : {grid.best_params_}")
print(f"  CV accuracy : {grid.best_score_:.4f}")
print(f"  Test accuracy: {titanic_acc:.4f}")
print(f"  Test ROC-AUC : {titanic_auc:.4f}")

# One unscaled k-NN on the same split, to show the section-3 lesson is not an
# artefact of synthetic data. Titanic's `fare` runs 0-512 while `sibsp` runs 0-8.
unscaled_prep = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), NUMERICAL),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="first", handle_unknown="error")),
            ]),
            CATEGORICAL,
        ),
    ]
)
unscaled = Pipeline([
    ("prep", unscaled_prep),
    ("knn", KNeighborsClassifier(**{
        k.replace("knn__", ""): v for k, v in grid.best_params_.items()
    })),
]).fit(X_train, y_train)
unscaled_acc = accuracy_score(y_test, unscaled.predict(X_test))
print(f"  Same hyperparameters, NO scaler: {unscaled_acc:.4f}   "
      f"(scaler is worth {titanic_acc - unscaled_acc:+.4f} here)")

cv_table = pd.DataFrame(grid.cv_results_)[
    ["param_knn__n_neighbors", "param_knn__weights", "param_knn__p",
     "mean_test_score"]
].sort_values("mean_test_score", ascending=False)
print("\n  Top 5 hyperparameter combinations by CV accuracy:")
print(cv_table.head(5).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for weights in ["uniform", "distance"]:
    for p, ls in [(1, "--"), (2, "-")]:
        mask = ((cv_table["param_knn__weights"] == weights)
                & (cv_table["param_knn__p"] == p))
        sub = cv_table[mask].sort_values("param_knn__n_neighbors")
        axes[0].plot(
            sub["param_knn__n_neighbors"].astype(int).to_numpy(),
            sub["mean_test_score"].to_numpy(),
            ls, marker="o", lw=1.8,
            label=f"{weights}, {'Manhattan' if p == 1 else 'Euclidean'}",
        )
axes[0].set_xlabel("k")
axes[0].set_ylabel("5-fold CV accuracy (training split)")
axes[0].set_title("Titanic: the full grid\nAll four curves land within ~0.03")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

bars = ["k-NN\n(tuned, scaled)", "k-NN\n(tuned, unscaled)", "Logistic\nregression*",
        "Random forest*"]
vals = [titanic_acc, unscaled_acc, 0.804, 0.821]
colors = ["seagreen", "indianred", "steelblue", "steelblue"]
axes[1].bar(bars, vals, color=colors)
for i, v in enumerate(vals):
    axes[1].text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=11)
axes[1].set_ylim(0.6, 0.90)
axes[1].set_ylabel("test accuracy")
axes[1].set_title("Same split, same seed, same seven features\n"
                  "* numbers recorded in trees_and_boosting/README.MD")
axes[1].grid(alpha=0.3, axis="y")
fig.tight_layout()
save_and_show(fig, "07_titanic_comparison.png")

print("\n" + "=" * 78)
print("Figures written to plots/")
print("=" * 78)
