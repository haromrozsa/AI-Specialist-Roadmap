"""
Random forests: why averaging a hundred overfit trees produces a model that is
not overfit.

DecisionTreeDepth.py ended on an uncomfortable result -- a fully grown tree hits
0.975 training accuracy and the worst cross-validated accuracy of any depth. It
has memorised the training set. The obvious fix is to make it smaller. Bagging
takes the opposite route: keep the trees deep and overfit, but grow many of them
on different bootstrap resamples and average the votes. Each tree is still wrong
in its own way; the ways cancel.

Four things get measured here rather than asserted:

  * that the variance really does collapse (section 2, the box plot)
  * that more trees is safe rather than a tuning risk, and that out-of-bag error
    is a free stand-in for a validation set (section 3)
  * that decorrelating the trees via max_features is what separates a *random*
    forest from plain bagged trees (section 4)
  * that the impurity-based feature_importances_ everybody quotes is biased, and
    permutation importance disagrees with it (section 5)

Section 6 then deliberately leaks the target into the features, because the
resulting plot is what leakage looks like from the outside and it is worth being
able to recognise on sight.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

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


print("=" * 70)
print("RANDOM FORESTS: BAGGING, DECORRELATION, AND FEATURE IMPORTANCE")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. The same boundary, one tree versus one hundred
# ---------------------------------------------------------------------------
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
    # predict_proba, not predict: the forest's averaged vote is a continuous
    # surface, and shading it shows the softening that hard labels hide.
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=20, alpha=0.45, cmap="coolwarm", vmin=0, vmax=1)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1.2)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=22, edgecolor="k", linewidth=0.4)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


single = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(Xm_train, ym_train)
forest = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE).fit(
    Xm_train, ym_train
)

print("\n--- Moons: one unlimited tree vs a 100-tree forest ---")
print(f"single tree   train={single.score(Xm_train, ym_train):.3f}  "
      f"test={single.score(Xm_test, ym_test):.3f}  leaves={single.get_n_leaves()}")
print(f"forest (100)  train={forest.score(Xm_train, ym_train):.3f}  "
      f"test={forest.score(Xm_test, ym_test):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
plot_boundary(
    axes[0], single, X_moons, y_moons,
    f"One unlimited tree ({single.get_n_leaves()} leaves)\n"
    f"train {single.score(Xm_train, ym_train):.3f}   test {single.score(Xm_test, ym_test):.3f}",
)
plot_boundary(
    axes[1], forest, X_moons, y_moons,
    f"100 trees averaged\n"
    f"train {forest.score(Xm_train, ym_train):.3f}   test {forest.score(Xm_test, ym_test):.3f}",
)
fig.suptitle(
    "Both models memorise the training set (train = 1.000). The forest is still\n"
    "made of hard staircases -- it just averages a hundred different ones, so the\n"
    "noise islands survive in the probabilities instead of in the decision.",
    fontsize=12,
)
fig.tight_layout()
save_and_show(fig, "05_bagging_boundary.png")

# ---------------------------------------------------------------------------
# 2. Titanic, and the measurement behind that picture
# ---------------------------------------------------------------------------
titanic = sns.load_dataset("titanic")

# Same seven features and the same dropped columns as DecisionTreeDepth.py --
# see the comment there for why alive/class/who/adult_male/embark_town/alone/deck
# are all excluded. Section 6 puts `alive` back on purpose.
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
NUMERICAL = ["age", "sibsp", "parch", "fare"]
CATEGORICAL = ["pclass", "sex", "embarked"]

X = titanic[FEATURES]
y = titanic["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

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

# Fitted on the training rows only; the test rows never touch the imputer or the
# encoder. Working with transformed arrays from here on keeps the bootstrap
# resampling in section 2 and the permutation importance in section 5 readable.
fitted_prep = preprocessor.fit(X_train, y_train)
X_train_t = fitted_prep.transform(X_train)
X_test_t = fitted_prep.transform(X_test)

ohe = fitted_prep.named_transformers_["cat"].named_steps["onehot"]
FEATURE_NAMES = NUMERICAL + list(ohe.get_feature_names(CATEGORICAL))

print("\n--- Titanic ---")
print(f"Train {X_train_t.shape}   Test {X_test_t.shape}   features {FEATURE_NAMES}")

# The variance experiment. Resample the training set 20 times and fit one
# unlimited tree per resample, then do the same with forests. The spread of the
# resulting test accuracies IS the variance of the procedure -- how much the
# model you end up with depends on which particular rows you happened to train on.
print("\n--- Variance under resampling (20 bootstrap resamples) ---")
N_REPEATS = 20
tree_scores, forest_scores = [], []

tree_preds, forest_preds = [], []

for i in range(N_REPEATS):
    Xb, yb = resample(X_train_t, y_train, replace=True, random_state=RANDOM_STATE + i)

    t = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(Xb, yb)
    f = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE).fit(Xb, yb)

    tree_scores.append(t.score(X_test_t, y_test))
    forest_scores.append(f.score(X_test_t, y_test))
    tree_preds.append(t.predict(X_test_t))
    forest_preds.append(f.predict(X_test_t))

tree_scores = np.array(tree_scores)
forest_scores = np.array(forest_scores)


def mean_pairwise_disagreement(predictions):
    """Average fraction of test rows on which two models of the same family differ.

    Accuracy spread is a blunt instrument for measuring variance: a 179-row test
    set has its own sampling noise, and that noise inflates the spread of both
    families equally. Disagreement between the models themselves has no such
    floor -- it measures only how much the fitted model depends on which rows it
    was trained on, which is exactly what variance means here.
    """
    preds = np.asarray(predictions)
    n = len(preds)
    rates = [
        np.mean(preds[i] != preds[j]) for i in range(n) for j in range(i + 1, n)
    ]
    return float(np.mean(rates))


tree_disagreement = mean_pairwise_disagreement(tree_preds)
forest_disagreement = mean_pairwise_disagreement(forest_preds)

print(f"{'':14} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} {'range':>8}")
for name, scores in [("single tree", tree_scores), ("100-tree forest", forest_scores)]:
    print(f"{name:14} {scores.mean():>8.3f} {scores.std():>8.4f} "
          f"{scores.min():>8.3f} {scores.max():>8.3f} "
          f"{scores.max() - scores.min():>8.3f}")
print(f"\nAccuracy std shrank {tree_scores.std() / forest_scores.std():.1f}x "
      f"({tree_scores.std():.4f} -> {forest_scores.std():.4f}) and the mean rose "
      f"{forest_scores.mean() - tree_scores.mean():+.3f}.")
print("\nThat accuracy spread understates the effect, because a 179-row test set")
print("carries sampling noise of its own that inflates both numbers. Comparing the")
print("models to each other instead removes that floor:")
print(f"  two single trees disagree on {tree_disagreement:.1%} of test rows")
print(f"  two forests   disagree on {forest_disagreement:.1%} of test rows "
      f"({tree_disagreement / forest_disagreement:.1f}x less)")
print("\nThat is the bargain: bagging trades a little bias for a large reduction in")
print("variance. Which rows you happened to train on stops determining what you")
print("predict -- and on data this noisy, that trade is worth taking.")

fig, ax = plt.subplots(figsize=(8, 6))
bp = ax.boxplot(
    [tree_scores, forest_scores],
    labels=["1 unlimited tree", "100-tree forest"],
    patch_artist=True,
    widths=0.5,
)
for patch, colour in zip(bp["boxes"], ["#d62728", "#1f77b4"]):
    patch.set_facecolor(colour)
    patch.set_alpha(0.45)
for i, scores in enumerate([tree_scores, forest_scores], start=1):
    ax.scatter(
        np.random.normal(i, 0.04, len(scores)), scores,
        s=18, color="k", alpha=0.6, zorder=3,
    )
ax.set_ylabel("Test accuracy")
ax.set_title(
    f"Same procedure, {N_REPEATS} different bootstrap resamples of the training set.\n"
    f"Accuracy spread narrows {tree_scores.std() / forest_scores.std():.1f}x, but the "
    f"sharper measure is disagreement:\ntwo single trees differ on "
    f"{tree_disagreement:.1%} of test rows, two forests on only {forest_disagreement:.1%}."
)
ax.grid(alpha=0.3, axis="y")
save_and_show(fig, "06_variance_reduction.png")

# ---------------------------------------------------------------------------
# 3. How many trees, and the free validation set that comes with bagging
# ---------------------------------------------------------------------------
# Each tree is trained on a bootstrap resample, so on average ~37% of the rows
# are left out of any given tree. Scoring each row using only the trees that did
# not see it gives an honest held-out estimate for free -- no split required.
print("\n--- n_estimators sweep, with out-of-bag error ---")
n_values = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300]
oob_errors, test_errors, train_errors = [], [], []

print("(sklearn warns at n=1,2,5: with that few trees some rows are in every")
print(" bootstrap sample and so have no out-of-bag vote. Expected, not a bug.)")

for n in n_values:
    rf = RandomForestClassifier(
        n_estimators=n, oob_score=True, bootstrap=True, random_state=RANDOM_STATE, n_jobs=-1
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # see the note above
        rf.fit(X_train_t, y_train)
    oob_errors.append(1 - rf.oob_score_)
    test_errors.append(1 - rf.score(X_test_t, y_test))
    train_errors.append(1 - rf.score(X_train_t, y_train))
    print(f"n={n:>4}  oob_err={oob_errors[-1]:.3f}  test_err={test_errors[-1]:.3f}  "
          f"train_err={train_errors[-1]:.3f}")

print(f"\nOOB error at n=300: {oob_errors[-1]:.3f}, test error {test_errors[-1]:.3f} "
      f"(difference {abs(oob_errors[-1] - test_errors[-1]):.3f})")
print("Neither curve turns upward. Adding trees to a forest cannot make it overfit --")
print("it is an averaging budget, not a capacity knob. GradientBoostingComparison.py")
print("shows the opposite behaviour, where more rounds genuinely do hurt.")

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(n_values, oob_errors, "o-", label="Out-of-bag error (free, no split)", color="#ff7f0e")
ax.plot(n_values, test_errors, "s-", label="Test error", color="#1f77b4")
ax.plot(n_values, train_errors, "^--", label="Training error", color="#d62728")
ax.set_xscale("log")
ax.set_xlabel("n_estimators (log scale)")
ax.set_ylabel("Error rate")
ax.set_title(
    "More trees never hurts -- both curves flatten rather than turning up.\n"
    "OOB error tracks test error closely enough to tune on."
)
ax.set_xticks(n_values)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.legend()
ax.grid(alpha=0.3)
save_and_show(fig, "07_n_estimators_oob.png")

# ---------------------------------------------------------------------------
# 4. What makes it a *random* forest rather than just bagged trees
# ---------------------------------------------------------------------------
# max_features caps how many features each split may consider. The textbook
# argument: at max_features = all, every tree greedily picks the same strong
# feature at the root, the trees end up near-identical, and averaging
# near-identical models buys nothing. Restricting the choice forces the trees to
# disagree, and disagreement is what averaging feeds on.
#
# The sweep below does not reproduce the accuracy benefit that argument predicts.
# Rather than quietly dropping the section, it is measured twice: once as
# accuracy (where the effect is absent) and once as tree-to-tree disagreement
# (where the mechanism is plainly there). The gap between those two results is
# the actual lesson.
print("\n--- max_features sweep (bagging vs random forest) ---")
n_features = X_train_t.shape[1]
mf_values = list(range(1, n_features + 1))
mf_oob, mf_test = [], []

for mf in mf_values:
    rf = RandomForestClassifier(
        n_estimators=200, max_features=mf, oob_score=True,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train_t, y_train)
    mf_oob.append(rf.oob_score_)
    mf_test.append(rf.score(X_test_t, y_test))
    tag = ""
    if mf == int(np.sqrt(n_features)):
        tag = "  <- sklearn's 'sqrt' default"
    if mf == n_features:
        tag = "  <- plain bagging, no feature subsampling"
    print(f"max_features={mf:>2}  oob={mf_oob[-1]:.3f}  test={mf_test[-1]:.3f}{tag}")

best_mf = mf_values[int(np.argmax(mf_oob))]
sqrt_mf = int(np.sqrt(n_features))
print(f"\nBest by OOB: max_features={best_mf} ({max(mf_oob):.3f}); "
      f"plain bagging (all {n_features}) scores {mf_oob[-1]:.3f}")
print(f"Spread across the whole sweep: {max(mf_oob) - min(mf_oob):.3f} OOB accuracy.")

# Is the decorrelation happening at all, or is the whole story wrong? Ask the
# trees directly: pull the individual estimators out of a forest and measure how
# often two of them predict different labels for the same passenger.
def within_forest_disagreement(max_features, n_estimators=100):
    f = RandomForestClassifier(
        n_estimators=n_estimators, max_features=max_features,
        random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(X_train_t, y_train)
    preds = np.array([est.predict(X_test_t) for est in f.estimators_])
    n = len(preds)
    return float(np.mean([
        np.mean(preds[i] != preds[j]) for i in range(n) for j in range(i + 1, n)
    ]))


dis_sqrt = within_forest_disagreement(sqrt_mf)
dis_all = within_forest_disagreement(n_features)
print(f"\nTree-to-tree disagreement inside one forest:")
print(f"  max_features={sqrt_mf} (sqrt default): {dis_sqrt:.1%} of test rows")
print(f"  max_features={n_features} (plain bagging): {dis_all:.1%} of test rows")
print("\nThe decorrelation is real and measurable -- the subsampled trees disagree")
print("measurably more. It just does not buy accuracy on this dataset, and the")
print("reason is the dataset, not the theory: 9 columns, 5 of them one-hot dummies")
print("with two or three usable splits between them. There is barely a redundant")
print("feature for subsampling to route around. The technique pays off on wide,")
print("correlated tabular data -- hundreds of columns measuring overlapping things --")
print("which is where random forests earned their reputation and is not this.")

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(mf_values, mf_oob, "o-", label="OOB accuracy", color="#ff7f0e")
ax.plot(mf_values, mf_test, "s--", label="Test accuracy", color="#1f77b4")
ax.axvline(sqrt_mf, color="#2ca02c", linestyle=":",
           label=f"sqrt({n_features}) = {sqrt_mf}, the default")
ax.axvline(n_features, color="grey", linestyle=":",
           label=f"all {n_features} features = plain bagging")
ax.set_xlabel("max_features (candidates considered per split)")
ax.set_ylabel("Accuracy")
ax.set_title(
    f"On this dataset max_features barely moves accuracy "
    f"({max(mf_oob) - min(mf_oob):.3f} OOB across the sweep).\n"
    f"The decorrelation still happens -- trees disagree {dis_sqrt:.0%} of the time at "
    f"sqrt vs {dis_all:.0%} at\nfull bagging -- but 9 mostly-dummy columns leave it "
    "nothing useful to route around."
)
ax.set_xticks(mf_values)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
save_and_show(fig, "08_max_features.png")

# ---------------------------------------------------------------------------
# 5. Feature importance: the number everyone quotes, and the one to trust
# ---------------------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=300, oob_score=True, random_state=RANDOM_STATE, n_jobs=-1
)
t0 = time.time()
rf.fit(X_train_t, y_train)
rf_fit_time = time.time() - t0
rf_test_acc = rf.score(X_test_t, y_test)

print("\n--- Feature importance, measured two ways ---")
print(f"Forest: 300 trees, oob={rf.oob_score_:.3f}, test={rf_test_acc:.3f}, "
      f"fit {rf_fit_time:.2f}s")

# feature_importances_ sums the impurity drop each feature produced across all
# splits. It is computed on the TRAINING data and it rewards features that had
# many chances to be chosen -- a continuous feature like fare offers hundreds of
# candidate thresholds, a binary one offers a single split. That is a bias
# towards high-cardinality features, not evidence of usefulness.
impurity_imp = rf.feature_importances_

# Permutation importance instead shuffles one column of the TEST set and measures
# how far accuracy falls. It answers the question people think they are asking:
# how much does the model's performance actually depend on this feature?
perm = permutation_importance(
    rf, X_test_t, y_test, n_repeats=30, random_state=RANDOM_STATE, scoring="accuracy"
)

order = np.argsort(perm.importances_mean)
print(f"\n{'feature':>12} {'impurity':>10} {'permutation':>14} {'perm std':>10}")
for i in order[::-1]:
    print(f"{FEATURE_NAMES[i]:>12} {impurity_imp[i]:>10.4f} "
          f"{perm.importances_mean[i]:>14.4f} {perm.importances_std[i]:>10.4f}")

imp_rank = [FEATURE_NAMES[i] for i in np.argsort(impurity_imp)[::-1]]
perm_rank = [FEATURE_NAMES[i] for i in np.argsort(perm.importances_mean)[::-1]]
print(f"\nImpurity ranking:    {imp_rank[:4]}")
print(f"Permutation ranking: {perm_rank[:4]}")
print("Quote the permutation ranking. The impurity ranking is measured on data the")
print("trees were fitted to, and it flatters continuous features for a reason that")
print("has nothing to do with whether they help.")

fig, ax = plt.subplots(figsize=(10, 6))
pos = np.arange(len(FEATURE_NAMES))
height = 0.4
ax.barh(pos + height / 2, impurity_imp[order], height,
        label="Impurity-based (train, biased)", color="#d62728", alpha=0.75)
ax.barh(pos - height / 2, perm.importances_mean[order], height,
        xerr=perm.importances_std[order], label="Permutation (test, honest)",
        color="#1f77b4", alpha=0.75, error_kw={"lw": 1})
ax.set_yticks(pos)
ax.set_yticklabels([FEATURE_NAMES[i] for i in order])
ax.set_xlabel("Importance (note: different units, compare the ordering not the height)")
ax.set_title(
    "Two importance measures, two different stories.\n"
    "Impurity importance inflates continuous features (age, fare) because they\n"
    "offer more candidate split points, not because they matter more."
)
ax.legend()
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
save_and_show(fig, "09_importance_impurity_vs_permutation.png")

# ---------------------------------------------------------------------------
# 6. What leakage looks like from the outside
# ---------------------------------------------------------------------------
# seaborn's Titanic frame carries `alive`, which is the survival target written
# as "yes"/"no". Every script here drops it. This section puts it back, on
# purpose, because the resulting numbers are the signature to learn to recognise.
print("\n" + "=" * 70)
print("DELIBERATE LEAKAGE DEMONSTRATION -- DO NOT COPY THIS")
print("=" * 70)

X_leak = titanic[FEATURES + ["alive"]].copy()
X_leak["alive"] = (X_leak["alive"] == "yes").astype(int)

Xl_train, Xl_test, yl_train, yl_test = train_test_split(
    X_leak, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

leak_prep = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), NUMERICAL + ["alive"]),
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
).fit(Xl_train, yl_train)

LEAK_NAMES = NUMERICAL + ["alive"] + list(
    leak_prep.named_transformers_["cat"].named_steps["onehot"].get_feature_names(CATEGORICAL)
)

rf_leak = RandomForestClassifier(
    n_estimators=300, oob_score=True, random_state=RANDOM_STATE, n_jobs=-1
)
rf_leak.fit(leak_prep.transform(Xl_train), yl_train)
leak_test_acc = rf_leak.score(leak_prep.transform(Xl_test), yl_test)

print(f"Test accuracy WITHOUT the leaked column: {rf_test_acc:.4f}")
print(f"Test accuracy WITH    the leaked column: {leak_test_acc:.4f}")
print(f"OOB score with the leaked column:        {rf_leak.oob_score_:.4f}")

leak_imp = rf_leak.feature_importances_
leak_order = np.argsort(leak_imp)
print(f"\n{'feature':>12} {'impurity importance':>22}")
for i in leak_order[::-1]:
    print(f"{LEAK_NAMES[i]:>12} {leak_imp[i]:>22.4f}")

print("\nThe signature to recognise:")
print("  1. an accuracy that is perfect, or implausibly close to it")
print("  2. one feature holding essentially all of the importance")
print("  3. every other feature collapsing to near zero")
print("Cross-validation does not catch this. OOB does not catch this. A held-out")
print("test set does not catch this -- the leaked column is in the test set too.")
print("The only thing that catches it is looking at your columns and asking where")
print("each one would come from at prediction time. `alive` is only knowable after")
print("the outcome you are trying to predict.")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.barh(np.arange(len(FEATURE_NAMES)), impurity_imp[np.argsort(impurity_imp)],
         color="#1f77b4", alpha=0.8)
ax1.set_yticks(np.arange(len(FEATURE_NAMES)))
ax1.set_yticklabels([FEATURE_NAMES[i] for i in np.argsort(impurity_imp)])
ax1.set_xlim(0, 1.0)
ax1.set_xlabel("Impurity importance")
ax1.set_title(f"Correct feature set\ntest accuracy {rf_test_acc:.3f}")
ax1.grid(alpha=0.3, axis="x")

colours = ["#d62728" if LEAK_NAMES[i] == "alive" else "#7f7f7f" for i in leak_order]
ax2.barh(np.arange(len(LEAK_NAMES)), leak_imp[leak_order], color=colours, alpha=0.85)
ax2.set_yticks(np.arange(len(LEAK_NAMES)))
ax2.set_yticklabels([LEAK_NAMES[i] for i in leak_order])
ax2.set_xlim(0, 1.0)
ax2.set_xlabel("Impurity importance")
ax2.set_title(f"With `alive` leaked in\ntest accuracy {leak_test_acc:.3f}")
ax2.grid(alpha=0.3, axis="x")

fig.suptitle(
    "Leakage is easy to spot once you know the shape: a near-perfect score and\n"
    "one bar that has eaten every other bar.",
    fontsize=12,
)
fig.tight_layout()
save_and_show(fig, "10_leakage_importance.png")

print("\n" + "=" * 70)
print(f"Honest random forest on this split: test accuracy {rf_test_acc:.3f}, "
      f"OOB {rf.oob_score_:.3f}")
print("GradientBoostingComparison.py builds the ensemble the other way round --")
print("sequentially, each tree correcting the last -- and the tuning advice inverts.")
print("=" * 70)
