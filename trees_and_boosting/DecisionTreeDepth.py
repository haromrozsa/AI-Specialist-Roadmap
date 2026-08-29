"""
Decision trees: how one tree carves up the feature space, and how it overfits
when you let it grow without limit.

Two datasets, each chosen for what it can show:

* make_moons is two-dimensional, so the decision boundary can actually be drawn.
  A picture of the boundary is the fastest way to see the two defining properties
  of a tree -- every cut is axis-aligned, and an unconstrained tree will happily
  carve out a private box around a single noisy point.

* Titanic is the same dataset (and the same seven features) used in
  logistic_regression/TitanicDatasetLogisticRegression.py, so the accuracy a tree
  reaches here is directly comparable to a model already in this repository
  instead of being a number with nothing to sit beside.

Note what is NOT here: no StandardScaler. A tree splits on a threshold inside a
single feature, so any monotone rescaling of that feature produces the identical
tree. The logistic regression script needs the scaler; this one would be doing
arithmetic for nothing.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

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


# ---------------------------------------------------------------------------
# 1. What a tree's decision boundary looks like, and what depth does to it
# ---------------------------------------------------------------------------
print("=" * 70)
print("DECISION TREES: DEPTH AND OVERFITTING")
print("=" * 70)

X_moons, y_moons = make_moons(n_samples=400, noise=0.3, random_state=RANDOM_STATE)

Xm_train, Xm_test, ym_train, ym_test = train_test_split(
    X_moons, y_moons, test_size=0.3, stratify=y_moons, random_state=RANDOM_STATE
)


def plot_boundary(ax, model, X, y, title):
    """Shade the model's prediction over a mesh, then scatter the data on top."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.30, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=22, edgecolor="k", linewidth=0.4)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


print("\n--- Moons: boundary shape at four depths ---")
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, depth in zip(axes.ravel(), [1, 3, 5, None]):
    tree = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
    tree.fit(Xm_train, ym_train)
    train_acc = tree.score(Xm_train, ym_train)
    test_acc = tree.score(Xm_test, ym_test)
    label = "unlimited" if depth is None else str(depth)
    plot_boundary(
        ax,
        tree,
        X_moons,
        y_moons,
        f"max_depth={label}  |  leaves={tree.get_n_leaves()}\n"
        f"train {train_acc:.3f}   test {test_acc:.3f}",
    )
    print(
        f"depth={label:>9}  leaves={tree.get_n_leaves():>4}  "
        f"train={train_acc:.3f}  test={test_acc:.3f}"
    )

fig.suptitle(
    "A tree's boundary is always a staircase of axis-aligned cuts.\n"
    "Let it grow and it carves private boxes around individual noisy points.",
    fontsize=12,
)
fig.tight_layout()
save_and_show(fig, "01_moons_depth_grid.png")

# The depth=1 panel is worth staring at: a single split is one straight line,
# because a tree can only ever cut perpendicular to one axis at a time. Every
# curve a tree appears to draw is that staircase at a finer resolution.

# ---------------------------------------------------------------------------
# 2. Titanic: the same seven features the logistic regression script uses
# ---------------------------------------------------------------------------
titanic = sns.load_dataset("titanic")

# seaborn's frame ships eight columns that must not go into a model here:
#   alive        -- the target, spelled as a string. Including it is pure leakage.
#   class, who, adult_male, embark_town, alone
#                -- re-encodings of pclass / sex / age / embarked / sibsp+parch
#   deck         -- 688 of 891 values missing (77%)
# RandomForestEnsemble.py deliberately puts `alive` back for one experiment, to
# show what leakage looks like from the outside.
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
NUMERICAL = ["age", "sibsp", "parch", "fare"]
CATEGORICAL = ["pclass", "sex", "embarked"]

X = titanic[FEATURES]
y = titanic["survived"]

print("\n--- Titanic ---")
print(f"Rows: {len(titanic)}   features used: {len(FEATURES)}")
print(f"Class balance: {y.value_counts().to_dict()}  ({y.mean():.1%} survived)")
print(f"Missing values in the features: {X.isna().sum()[X.isna().sum() > 0].to_dict()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape}   Test: {X_test.shape}")

# Imputation and one-hot encoding live inside a ColumnTransformer so they can be
# wrapped in a Pipeline. That matters for cross-validation below: the median age
# has to be recomputed inside every fold, from that fold's training rows only.
# Computing it once over the whole training set and then cross-validating leaks
# information between folds and quietly inflates the score.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), NUMERICAL),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(drop="first", handle_unknown="error")),
                ]
            ),
            CATEGORICAL,
        ),
    ]
)


def make_tree_pipeline(**tree_kwargs):
    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE, **tree_kwargs)),
        ]
    )


# A second, fitted-on-train-only copy of the preprocessing, used where a raw
# array is genuinely needed (the pruning path in section 4, and readable feature
# names for the tree drawing in section 3). Fitting it on X_train and applying it
# to X_test is exactly what the Pipeline does internally -- the test rows never
# influence the imputer or the encoder.
fitted_prep = preprocessor.fit(X_train, y_train)
X_train_t = fitted_prep.transform(X_train)
X_test_t = fitted_prep.transform(X_test)

ohe = fitted_prep.named_transformers_["cat"].named_steps["onehot"]
FEATURE_NAMES = NUMERICAL + list(ohe.get_feature_names(CATEGORICAL))
print(f"After one-hot encoding: {len(FEATURE_NAMES)} columns -> {FEATURE_NAMES}")

# ---------------------------------------------------------------------------
# 3. The overfitting curve: training accuracy climbs, test accuracy peaks
# ---------------------------------------------------------------------------
print("\n--- Depth sweep (5-fold CV on the training set) ---")
print(f"{'depth':>6} {'leaves':>7} {'train':>8} {'test':>8} {'cv mean':>9} {'cv std':>8}")

depths = list(range(1, 16))
train_scores, test_scores, cv_means, cv_stds, leaf_counts = [], [], [], [], []

for depth in depths:
    pipe = make_tree_pipeline(max_depth=depth)
    pipe.fit(X_train, y_train)

    train_acc = pipe.score(X_train, y_train)
    test_acc = pipe.score(X_test, y_test)

    # cross_val_score on the Pipeline, not on X_train_t: every fold refits the
    # imputer and the encoder from scratch.
    cv = cross_val_score(
        make_tree_pipeline(max_depth=depth), X_train, y_train, cv=5, scoring="accuracy"
    )

    train_scores.append(train_acc)
    test_scores.append(test_acc)
    cv_means.append(cv.mean())
    cv_stds.append(cv.std())
    leaf_counts.append(pipe.named_steps["clf"].get_n_leaves())

    print(
        f"{depth:>6} {leaf_counts[-1]:>7} {train_acc:>8.3f} {test_acc:>8.3f} "
        f"{cv.mean():>9.3f} {cv.std():>8.3f}"
    )

train_scores = np.array(train_scores)
test_scores = np.array(test_scores)
cv_means = np.array(cv_means)
cv_stds = np.array(cv_stds)

best_depth = depths[int(np.argmax(cv_means))]
print(f"\nDepth chosen by cross-validation: {best_depth} (CV {cv_means.max():.3f})")
print(f"  its test accuracy: {test_scores[depths.index(best_depth)]:.3f}")
print(f"Depth with the best TEST accuracy: {depths[int(np.argmax(test_scores))]} "
      f"({test_scores.max():.3f})  <- not something you are allowed to select on")
print(f"Training accuracy at depth 15: {train_scores[-1]:.3f} "
      f"({leaf_counts[-1]} leaves for {len(X_train)} training rows)")
print(f"Train-minus-CV gap: {train_scores[0] - cv_means[0]:+.3f} at depth 1 -> "
      f"{train_scores[-1] - cv_means[-1]:+.3f} at depth 15")

# Read the three curves separately, because they do not all tell the same story:
#
#   training accuracy  climbs monotonically towards 1.0 and keeps climbing. A
#                      deeper tree can always cut the training data finer, so
#                      this curve can never diagnose anything on its own.
#   CV accuracy        rises to depth 4 and then falls away steadily. This is
#                      the overfitting curve, and it is measured on 712 rows
#                      through five folds, so it has the resolution to show it.
#   test accuracy      wanders between 0.76 and 0.81 with no clear peak. Not a
#                      contradiction -- the test set is 179 rows, so a single
#                      flipped prediction moves it by 0.006 and the whole visible
#                      range is roughly eight passengers.
#
# The practical consequence is the reason cross-validation exists: on a dataset
# this size the held-out set is too noisy to select a hyperparameter with, and
# picking the depth that happens to score best on it is fitting the test set by
# hand.
print(f"One test prediction is worth {1 / len(y_test):.4f} accuracy, so differences")
print("below about 0.02 between models on this split are noise, not results.")

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(depths, train_scores, "o-", label="Training accuracy", color="#d62728")
ax.plot(depths, test_scores, "s-", label="Test accuracy", color="#1f77b4")
ax.plot(depths, cv_means, "^--", label="5-fold CV accuracy (train only)", color="#2ca02c")
ax.fill_between(
    depths, cv_means - cv_stds, cv_means + cv_stds, alpha=0.18, color="#2ca02c"
)
ax.axvline(best_depth, color="grey", linestyle=":", label=f"CV-selected depth = {best_depth}")
ax.set_xlabel("max_depth")
ax.set_ylabel("Accuracy")
ax.set_title(
    "Training accuracy only ever goes up. Cross-validated accuracy peaks at\n"
    "depth 4 and decays -- that is the overfitting. The 179-row test curve is\n"
    "too noisy to show it, which is precisely why you select on CV instead."
)
ax.set_xticks(depths)
ax.legend()
ax.grid(alpha=0.3)
save_and_show(fig, "02_depth_vs_accuracy.png")

# ---------------------------------------------------------------------------
# 4. What the tree actually learned
# ---------------------------------------------------------------------------
shallow = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
shallow.fit(X_train_t, y_train)

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    shallow,
    feature_names=FEATURE_NAMES,
    class_names=["died", "survived"],
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax,
)
ax.set_title(
    "A depth-3 tree on Titanic. Each node prints its split, its gini impurity,\n"
    "how many training rows reached it, and how those rows were distributed.",
    fontsize=13,
)
save_and_show(fig, "03_tree_depth3.png")

root_feature = FEATURE_NAMES[shallow.tree_.feature[0]]
root_threshold = shallow.tree_.threshold[0]
print(f"\nRoot split: {root_feature} <= {root_threshold:.3f}")
print("Gini at the root:", round(shallow.tree_.impurity[0], 4))
print("A tree picks the split that drops impurity the most, so the root is the")
print("single most informative question you could ask about a passenger.")

# ---------------------------------------------------------------------------
# 5. Pruning: the principled way to stop a tree, instead of guessing a depth
# ---------------------------------------------------------------------------
# Cost-complexity pruning grows the tree out fully and then removes the subtrees
# whose contribution to accuracy is worth less than alpha per extra leaf. The
# pruning path enumerates every alpha at which some subtree collapses, so the
# candidates are the ones the data actually produces rather than a made-up grid.
print("\n--- Cost-complexity pruning ---")
path = DecisionTreeClassifier(random_state=RANDOM_STATE).cost_complexity_pruning_path(
    X_train_t, y_train
)
alphas = path.ccp_alphas[:-1]  # the last alpha collapses the tree to a single root node
print(f"{len(alphas)} candidate alphas between {alphas.min():.5f} and {alphas.max():.5f}")

prune_train, prune_test, prune_leaves, prune_cv = [], [], [], []
for alpha in alphas:
    tree = DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=alpha)
    tree.fit(X_train_t, y_train)
    prune_train.append(tree.score(X_train_t, y_train))
    prune_test.append(tree.score(X_test_t, y_test))
    prune_leaves.append(tree.get_n_leaves())
    prune_cv.append(
        cross_val_score(
            Pipeline(
                steps=[
                    ("prep", preprocessor),
                    (
                        "clf",
                        DecisionTreeClassifier(
                            random_state=RANDOM_STATE, ccp_alpha=alpha
                        ),
                    ),
                ]
            ),
            X_train,
            y_train,
            cv=5,
            scoring="accuracy",
        ).mean()
    )

prune_cv = np.array(prune_cv)
best_alpha = alphas[int(np.argmax(prune_cv))]
pruned = DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=best_alpha)
pruned.fit(X_train_t, y_train)
pruned_test_acc = pruned.score(X_test_t, y_test)

print(f"Best alpha by CV: {best_alpha:.5f}")
print(f"  leaves: {pruned.get_n_leaves()} (down from "
      f"{DecisionTreeClassifier(random_state=RANDOM_STATE).fit(X_train_t, y_train).get_n_leaves()} unpruned)")
print(f"  depth:  {pruned.get_depth()}")
print(f"  train accuracy: {pruned.score(X_train_t, y_train):.3f}")
print(f"  test accuracy:  {pruned_test_acc:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
ax1.plot(alphas, prune_train, "o-", markersize=3, label="Training accuracy", color="#d62728")
ax1.plot(alphas, prune_test, "s-", markersize=3, label="Test accuracy", color="#1f77b4")
ax1.plot(alphas, prune_cv, "^--", markersize=3, label="5-fold CV accuracy", color="#2ca02c")
ax1.axvline(best_alpha, color="grey", linestyle=":", label=f"CV-selected alpha = {best_alpha:.4f}")
ax1.set_xlabel("ccp_alpha (cost per leaf)")
ax1.set_ylabel("Accuracy")
ax1.set_title("Pruning harder costs training accuracy\nlong before it costs test accuracy")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

ax2.plot(alphas, prune_leaves, "o-", markersize=3, color="#9467bd")
ax2.axvline(best_alpha, color="grey", linestyle=":")
ax2.set_xlabel("ccp_alpha (cost per leaf)")
ax2.set_ylabel("Number of leaves")
ax2.set_yscale("log")
ax2.set_title("Leaves surviving at each alpha (log scale)")
ax2.grid(alpha=0.3)

fig.tight_layout()
save_and_show(fig, "04_cost_complexity_pruning.png")

# ---------------------------------------------------------------------------
# 6. Split criterion, and a baseline worth comparing against
# ---------------------------------------------------------------------------
print("\n--- gini vs entropy at the CV-selected depth ---")
for criterion in ["gini", "entropy"]:
    pipe = make_tree_pipeline(max_depth=best_depth, criterion=criterion)
    pipe.fit(X_train, y_train)
    cv = cross_val_score(
        make_tree_pipeline(max_depth=best_depth, criterion=criterion),
        X_train,
        y_train,
        cv=5,
    )
    print(
        f"{criterion:>8}: test={pipe.score(X_test, y_test):.3f}  "
        f"cv={cv.mean():.3f}  leaves={pipe.named_steps['clf'].get_n_leaves()}"
    )
print("Both measure the same thing -- how mixed a node is -- and they almost never")
print("disagree about which split is best. This is not a hyperparameter to agonise over.")

# The logistic regression the rest of this repository already has, refit on the
# identical split so the tree's number means something. It needs the scaler the
# tree does not.
logreg = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]
)
t0 = time.time()
logreg.fit(X_train, y_train)
logreg_time = time.time() - t0
logreg_acc = accuracy_score(y_test, logreg.predict(X_test))

majority_acc = max(y_test.mean(), 1 - y_test.mean())

print("\n" + "=" * 70)
print("BASELINES ON THE IDENTICAL TEST SPLIT")
print("=" * 70)
print(f"{'Predict the majority class':<34} {majority_acc:.3f}")
print(f"{'Logistic regression':<34} {logreg_acc:.3f}   ({logreg_time:.3f}s)")
print(f"{'Tree, CV-selected depth ' + str(best_depth):<34} "
      f"{test_scores[depths.index(best_depth)]:.3f}")
print(f"{'Tree, cost-complexity pruned':<34} {pruned_test_acc:.3f}")
print(f"{'Tree, depth 15 (deepest swept)':<34} {test_scores[-1]:.3f}   "
      f"(train {train_scores[-1]:.3f}, CV {cv_means[-1]:.3f})")
print("\nThe depth-15 tree scores respectably on this particular test split, and")
print("that is exactly the trap: its training accuracy is near 1.0 and its CV")
print("accuracy is the worst of any depth. It has memorised the training set, and")
print("179 test rows are not enough to catch it out.")
print("\nRandomForestEnsemble.py takes that memorising tree and shows why averaging")
print("a hundred of them turns high variance into a usable model.")
