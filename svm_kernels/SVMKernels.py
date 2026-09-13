"""
Support Vector Machines: the margin, the kernel trick, and the two knobs.

One script, six figures, four questions:

1. Why does a linear SVM fail on data that is not linearly separable, and what
   exactly does "the kernel trick" do about it? Answered by lifting
   make_circles into a third dimension by hand and showing a flat plane cut it,
   then checking that an RBF SVC finds the same boundary without ever building
   that third dimension.
2. What is a support vector? Drawn, by plotting the margin as the -1 / 0 / +1
   contours of the decision function and circling the points that touch it.
3. What do C and gamma actually control? Swept one at a time, then jointly as a
   cross-validated heatmap, so the two-knob interaction is visible rather than
   described.
4. Why does every SVM tutorial scale the features first? Measured, by breaking
   one feature's scale on purpose.

Everything runs on 2-D synthetic data (make_moons, make_circles) because the
whole point of an SVM is the shape of its decision boundary, and a boundary you
cannot draw teaches nothing. That is a deliberate departure from the previous two
tasks in this repository (trees_and_boosting/, shap_explainability/), which both
reused the Titanic split for comparability.

Unlike a tree, an SVM is a distance-based model: the margin is measured in the
feature space's own units, so StandardScaler is not optional here. It is applied
inside a Pipeline throughout, so the scaler is fitted on training folds only and
cross-validation stays honest.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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


def plot_boundary(ax, model, X, y, title, show_margin=False):
    """Shade the decision regions of a fitted model over the span of X.

    With show_margin=True the raw decision_function is contoured at -1, 0 and +1
    instead of only at the class flip, which is what makes the margin -- the band
    the SVM is actually maximising -- visible. The support vectors are circled.
    """
    pad = 0.5
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 300),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    zz = model.decision_function(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz > 0, alpha=0.20, cmap="coolwarm")

    if show_margin:
        ax.contour(
            xx, yy, zz,
            levels=[-1, 0, 1],
            colors="k",
            linestyles=["--", "-", "--"],
            linewidths=[0.8, 1.4, 0.8],
        )
        svc = model[-1] if isinstance(model, Pipeline) else model
        sv = model[:-1].transform(X) if isinstance(model, Pipeline) else X
        sv = sv[svc.support_]
        # Support vectors live in the scaled space; map them back for plotting.
        if isinstance(model, Pipeline):
            sv = model[:-1].inverse_transform(sv)
        ax.scatter(sv[:, 0], sv[:, 1], s=110, facecolors="none",
                   edgecolors="k", linewidths=1.1, zorder=3)

    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=22,
               edgecolors="k", linewidths=0.3, zorder=2)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def rbf_svc(C=1.0, gamma="scale"):
    """An RBF SVM with its scaler, as one leakage-safe estimator."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=C, gamma=gamma)),
    ])


# ---------------------------------------------------------------------------
# 1. The kernel trick, done by hand
# ---------------------------------------------------------------------------
# make_circles is the cleanest possible counter-example to a linear model: one
# class is a ring around the other, so no straight line can separate them. But
# the two classes DO differ in one thing -- their distance from the origin. Add
# that distance as a third feature and the problem becomes linearly separable by
# a flat plane. That lift is what a kernel does; the trick is that the kernel
# computes inner products in the lifted space without ever building it.

X_circ, y_circ = make_circles(n_samples=300, factor=0.35, noise=0.09,
                              random_state=RANDOM_STATE)

lin_circ = Pipeline([("scale", StandardScaler()),
                     ("svc", SVC(kernel="linear", C=1.0))]).fit(X_circ, y_circ)
rbf_circ = rbf_svc(C=1.0).fit(X_circ, y_circ)

print("=" * 74)
print("1. THE KERNEL TRICK")
print("=" * 74)
print("make_circles: one class ringed around the other, 300 points")
print("  linear kernel, training accuracy: %.4f"
      % accuracy_score(y_circ, lin_circ.predict(X_circ)))
print("  RBF kernel,    training accuracy: %.4f"
      % accuracy_score(y_circ, rbf_circ.predict(X_circ)))

fig = plt.figure(figsize=(15, 4.4))

ax = fig.add_subplot(1, 3, 1)
plot_boundary(ax, lin_circ, X_circ, y_circ, "Linear kernel: no line can do it")

# The explicit lift: phi(x) = (x1, x2, x1^2 + x2^2).
ax = fig.add_subplot(1, 3, 2, projection="3d")
r = X_circ[:, 0] ** 2 + X_circ[:, 1] ** 2
ax.scatter(X_circ[:, 0], X_circ[:, 1], r, c=y_circ, cmap="coolwarm", s=18,
           edgecolors="k", linewidths=0.2)
plane = np.full((2, 2), (r[y_circ == 0].min() + r[y_circ == 1].max()) / 2)
gx, gy = np.meshgrid([X_circ[:, 0].min(), X_circ[:, 0].max()],
                     [X_circ[:, 1].min(), X_circ[:, 1].max()])
ax.plot_surface(gx, gy, plane, alpha=0.35, color="#2ca02c", zorder=1)
ax.set_title(r"Lifted by hand: $\phi(x)=(x_1,x_2,x_1^2+x_2^2)$", fontsize=10)
ax.set_zlabel(r"$x_1^2+x_2^2$", fontsize=8)
ax.set_xticks([])
ax.set_yticks([])
ax.view_init(elev=16, azim=-60)
ax.grid(False)
for pane in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
    pane.pane.set_visible(False)

ax = fig.add_subplot(1, 3, 3)
plot_boundary(ax, rbf_circ, X_circ, y_circ,
              "RBF kernel: the same cut, without building the 3rd dimension")

save_and_show(fig, "01_kernel_trick.png")


# ---------------------------------------------------------------------------
# 2. Linear vs RBF on moons, with the margin and the support vectors drawn
# ---------------------------------------------------------------------------
# The dashed lines are the +/-1 contours of the decision function: the margin.
# Circled points are the support vectors -- the only training points that carry
# any weight in the fitted model. Delete every other point and refit, and the
# boundary does not move. That is the property the name is pointing at.

X, y = make_moons(n_samples=300, noise=0.25, random_state=RANDOM_STATE)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

lin = Pipeline([("scale", StandardScaler()),
                ("svc", SVC(kernel="linear", C=1.0))]).fit(X_train, y_train)
rbf = rbf_svc(C=1.0).fit(X_train, y_train)

print()
print("=" * 74)
print("2. LINEAR VS RBF ON MOONS (n=300, noise=0.25, 70/30 split)")
print("=" * 74)
for name, model in [("linear", lin), ("rbf", rbf)]:
    n_sv = model["svc"].n_support_.sum()
    print("  %-7s train %.4f  test %.4f  support vectors %3d / %d"
          % (name,
             accuracy_score(y_train, model.predict(X_train)),
             accuracy_score(y_test, model.predict(X_test)),
             n_sv, len(X_train)))

# The claim that the support vectors ARE the model, checked rather than asserted:
# refit on nothing but them and compare predictions point by point. The scaler is
# fitted once on the full training set in both cases, so the only thing that
# changes is which rows the SVC itself sees.
scaler = StandardScaler().fit(X_train)
Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)
# gamma is pinned to a number rather than left at 'scale', because 'scale' is
# 1/(n_features * X.var()) -- it would be recomputed from the smaller subset and
# quietly change the kernel, which is not the thing being tested here.
g = 1.0 / (Xs_train.shape[1] * Xs_train.var())
full = SVC(kernel="rbf", C=1.0, gamma=g).fit(Xs_train, y_train)
sv_idx = full.support_
sv_only = SVC(kernel="rbf", C=1.0, gamma=g).fit(Xs_train[sv_idx], y_train[sv_idx])
agree = (full.predict(Xs_test) == sv_only.predict(Xs_test)).mean()
print("  refit on the %d support vectors alone: %.4f of test predictions identical"
      % (len(sv_idx), agree))
print("  max |decision_function| difference over the test set: %.2e"
      % np.abs(full.decision_function(Xs_test)
               - sv_only.decision_function(Xs_test)).max())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
plot_boundary(axes[0], lin, X_train, y_train,
              "Linear kernel  (test %.3f)" % accuracy_score(y_test, lin.predict(X_test)),
              show_margin=True)
plot_boundary(axes[1], rbf, X_train, y_train,
              "RBF kernel  (test %.3f)" % accuracy_score(y_test, rbf.predict(X_test)),
              show_margin=True)
fig.suptitle("Solid = boundary, dashed = margin, circled = support vectors", fontsize=11)
save_and_show(fig, "02_linear_vs_rbf_margin.png")


# ---------------------------------------------------------------------------
# 3. C: how much the margin is allowed to be violated
# ---------------------------------------------------------------------------
# C is the price paid per margin violation. Small C buys a wide margin by
# tolerating misclassified points -- so most points end up inside the margin and
# become support vectors. Large C refuses violations, narrowing the margin until
# it contorts around individual noisy points. The support-vector count is the
# most direct read-out of that trade-off.

C_values = [0.01, 0.1, 1, 10, 100, 1000]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

print()
print("=" * 74)
print("3. C SWEEP (RBF, gamma='scale')")
print("=" * 74)
print("  %8s %8s %8s %14s" % ("C", "train", "test", "support vecs"))
sv_counts, c_test = [], []
for ax, C in zip(axes.ravel(), C_values):
    m = rbf_svc(C=C).fit(X_train, y_train)
    tr = accuracy_score(y_train, m.predict(X_train))
    te = accuracy_score(y_test, m.predict(X_test))
    n_sv = m["svc"].n_support_.sum()
    sv_counts.append(n_sv)
    c_test.append(te)
    print("  %8g %8.4f %8.4f %10d /%3d" % (C, tr, te, n_sv, len(X_train)))
    plot_boundary(ax, m, X_train, y_train,
                  "C = %g   test %.3f   %d SVs" % (C, te, n_sv), show_margin=True)
fig.suptitle("C controls the margin/violation trade-off (RBF, gamma='scale')", fontsize=12)
save_and_show(fig, "03_C_sweep.png")

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.semilogx(C_values, sv_counts, "o-", color="#1f77b4", label="support vectors")
ax.set_xlabel("C (log scale)")
ax.set_ylabel("support vectors", color="#1f77b4")
ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.semilogx(C_values, c_test, "s--", color="#d62728", label="test accuracy")
ax2.set_ylabel("test accuracy", color="#d62728")
ax.set_title("More C, fewer support vectors -- the model leans on less of the data")
save_and_show(fig, "04_C_vs_support_vectors.png")


# ---------------------------------------------------------------------------
# 4. gamma: how far a single training point's influence reaches
# ---------------------------------------------------------------------------
# The RBF kernel is exp(-gamma * ||x - x'||^2). gamma is the inverse width of
# that bell: small gamma means every point influences the boundary from far away
# (the boundary goes nearly linear), large gamma means influence dies within a
# tiny radius, and the model degenerates into islands drawn around individual
# training points. That is overfitting you can literally see.

gammas = [0.01, 0.1, 1, 10, 100, 1000]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

print()
print("=" * 74)
print("4. GAMMA SWEEP (RBF, C=1)")
print("=" * 74)
print("  %8s %8s %8s" % ("gamma", "train", "test"))
for ax, g in zip(axes.ravel(), gammas):
    m = rbf_svc(C=1.0, gamma=g).fit(X_train, y_train)
    tr = accuracy_score(y_train, m.predict(X_train))
    te = accuracy_score(y_test, m.predict(X_test))
    print("  %8g %8.4f %8.4f" % (g, tr, te))
    plot_boundary(ax, m, X_train, y_train,
                  "gamma = %g   train %.3f   test %.3f" % (g, tr, te))
fig.suptitle("gamma is the reach of one training point (RBF, C=1)", fontsize=12)
save_and_show(fig, "05_gamma_sweep.png")


# ---------------------------------------------------------------------------
# 5. C and gamma together
# ---------------------------------------------------------------------------
# Sweeping them one at a time hides the fact that they trade against each other:
# a too-large gamma can be partly rescued by a small C, and vice versa. The
# cross-validated grid shows the diagonal ridge of good combinations, which is
# why these two are always tuned jointly.

grid = GridSearchCV(
    rbf_svc(),
    {"svc__C": C_values, "svc__gamma": gammas},
    cv=5, n_jobs=1,
)
grid.fit(X_train, y_train)
scores = grid.cv_results_["mean_test_score"].reshape(len(C_values), len(gammas))

print()
print("=" * 74)
print("5. JOINT C/GAMMA SEARCH (5-fold CV on the training split)")
print("=" * 74)
print("  best params: %s" % grid.best_params_)
print("  best CV accuracy:  %.4f" % grid.best_score_)
print("  test accuracy:     %.4f"
      % accuracy_score(y_test, grid.best_estimator_.predict(X_test)))

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(scores, cmap="viridis", origin="lower", aspect="auto")
ax.set_xticks(range(len(gammas)))
ax.set_xticklabels(gammas)
ax.set_yticks(range(len(C_values)))
ax.set_yticklabels(C_values)
ax.set_xlabel("gamma")
ax.set_ylabel("C")
ax.set_title("5-fold CV accuracy over the C/gamma grid")
for i in range(len(C_values)):
    for j in range(len(gammas)):
        ax.text(j, i, "%.2f" % scores[i, j], ha="center", va="center",
                color="w" if scores[i, j] < scores.max() - 0.08 else "k", fontsize=8)
fig.colorbar(im, ax=ax, label="mean CV accuracy")
save_and_show(fig, "06_C_gamma_heatmap.png")


# ---------------------------------------------------------------------------
# 6. Why the scaler is not optional
# ---------------------------------------------------------------------------
# The RBF kernel measures a Euclidean distance. Multiply one feature by 1000 and
# that feature dominates the distance, so gamma is effectively enormous along one
# axis and nil along the other. A decision tree would not notice this at all --
# it splits one feature at a time, and any monotone rescaling gives the identical
# tree (see trees_and_boosting/DecisionTreeDepth.py). An SVM notices immediately.

X_bad_train, X_bad_test = X_train.copy(), X_test.copy()
X_bad_train[:, 1] *= 1000
X_bad_test[:, 1] *= 1000

unscaled = SVC(kernel="rbf", C=1.0).fit(X_bad_train, y_train)
scaled = rbf_svc(C=1.0).fit(X_bad_train, y_train)

print()
print("=" * 74)
print("6. FEATURE SCALING (feature 2 multiplied by 1000)")
print("=" * 74)
print("  no scaler:        test %.4f  (%d support vectors)"
      % (accuracy_score(y_test, unscaled.predict(X_bad_test)),
         unscaled.n_support_.sum()))
print("  StandardScaler:   test %.4f  (%d support vectors)"
      % (accuracy_score(y_test, scaled.predict(X_bad_test)),
         scaled["svc"].n_support_.sum()))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
plot_boundary(axes[0], unscaled, X_bad_train, y_train,
              "No scaler  (test %.3f)"
              % accuracy_score(y_test, unscaled.predict(X_bad_test)))
plot_boundary(axes[1], scaled, X_bad_train, y_train,
              "StandardScaler in a Pipeline  (test %.3f)"
              % accuracy_score(y_test, scaled.predict(X_bad_test)))
fig.suptitle("Same data, one feature scaled x1000: the RBF kernel is a distance",
             fontsize=11)
save_and_show(fig, "07_scaling_matters.png")

print()
print("Figures written to %s" % PLOTS_DIR)
