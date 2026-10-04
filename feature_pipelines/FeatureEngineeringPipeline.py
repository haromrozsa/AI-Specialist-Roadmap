"""
Pipeline + ColumnTransformer: what the wrapper is actually buying you.

CLAUDE.md's code-review guidance names "data leakage (scaler or encoder fitted
before the train/test split)" as the first thing to look for in this repository,
and eight files here already use ColumnTransformer inside a Pipeline to avoid
exactly that. What none of them do is *measure the cost of getting it wrong*.
That is what this script is for. The question is not "how do I write a
Pipeline" -- trees_and_boosting/, shap_explainability/, knn_and_naive_bayes/ and
hyperparameter_search/ all answer that -- but "what does the Pipeline prevent,
in accuracy points, and when is the amount big enough to care about".

Three leaks, measured on the Titanic split that
logistic_regression/TitanicDatasetLogisticRegression.py, trees_and_boosting/,
shap_explainability/, knn_and_naive_bayes/ and hyperparameter_search/ all share
(test_size=0.2, stratify, random_state=42), so the honest baseline number sits
next to figures already recorded in this repository:

1. Preprocessor fitted on all rows before the split. The imputer's medians and
   the scaler's means are computed from test rows the model is about to be
   scored on. This is the shape CLAUDE.md describes and the shape that already
   exists in linear_regression/CaliforniaHousingLinearRegression.py:43-45.
2. Preprocessor fitted once on the whole training set before cross-validation.
   Subtler, and the more common one in real code, because the train/test split
   *was* respected -- the leak is between CV folds, where each validation fold
   was used to fit the transformer that prepares it.
3. The same leak as (1), repeated across shrinking training sets, because the
   size of the effect is the part that is usually asserted rather than measured.

The honest answer to (1) and (2) on this data is "almost nothing", and that is
written up as it came out rather than inflated into a scare number. Section 3
exists to find the conditions where it stops being almost nothing.
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
# The shared Titanic setup
# ---------------------------------------------------------------------------
titanic = sns.load_dataset("titanic")

FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
NUMERICAL = ["age", "sibsp", "parch", "fare"]
CATEGORICAL = ["pclass", "sex", "embarked"]

X = titanic[FEATURES]
y = titanic["survived"]

# `age` is the column that makes this worth measuring at all: 177 of 891 rows
# are missing, so the imputer has a real value to compute and therefore a real
# opportunity to compute it from the wrong rows.
print("=" * 70)
print("SETUP")
print("=" * 70)
print(f"rows: {len(X)}   missing age: {X['age'].isna().sum()}   "
      f"missing embarked: {X['embarked'].isna().sum()}")


def make_preprocessor(tolerant=False):
    """Median-impute + scale the numerics, mode-impute + one-hot the categoricals.

    A fresh instance every time, because the whole point below is to control
    exactly which rows each copy of this gets fitted on.

    `tolerant` exists for section 3. Fitting on a 30-row training set can miss a
    whole category, and the strict default -- correct for sections 1 and 2, and
    the setting that produced the finding in section 2b -- would abort the sweep
    rather than score it. sklearn refuses drop="first" together with
    handle_unknown="ignore" (dropping a level and zeroing unknowns would make
    the two indistinguishable), so the tolerant build keeps every level instead.
    """
    drop, handle_unknown = (None, "ignore") if tolerant else ("first", "error")
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERICAL,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(drop=drop, handle_unknown=handle_unknown)),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )


def make_model():
    return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"train: {len(X_train)}   test: {len(X_test)}")


# ---------------------------------------------------------------------------
# 1. The correct way, and leak #1: preprocessor fitted before the split
# ---------------------------------------------------------------------------
# Correct: the Pipeline is one estimator. Calling .fit() on it fits the
# ColumnTransformer on the training rows only; .predict() on the test rows
# calls .transform(), never .fit_transform(). There is no way to get the order
# wrong because there is no order to get wrong -- that is the entire argument
# for the wrapper, and it is an argument about human error, not about accuracy.
clean = Pipeline(steps=[("prep", make_preprocessor()), ("model", make_model())])
clean.fit(X_train, y_train)
clean_acc = clean.score(X_test, y_test)

# Leaky: fit the transformer on all 891 rows, *then* split. The medians and the
# scaler statistics now carry information from the 179 test rows.
leaked_all = make_preprocessor().fit_transform(X)
Xl_train, Xl_test, yl_train, yl_test = train_test_split(
    leaked_all, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
leaky_split_model = make_model().fit(Xl_train, yl_train)
leaky_split_acc = leaky_split_model.score(Xl_test, yl_test)

print("\n" + "=" * 70)
print("1. PREPROCESSOR FITTED BEFORE THE SPLIT")
print("=" * 70)
print(f"  Pipeline, fitted on train only : {clean_acc:.4f}")
print(f"  Transformer fitted on all rows : {leaky_split_acc:.4f}")
print(f"  difference                     : {leaky_split_acc - clean_acc:+.4f}")


# ---------------------------------------------------------------------------
# 2. Leak #2: preprocessing once, then cross-validating
# ---------------------------------------------------------------------------
# This one respects the train/test split and still leaks. `cross_val_score` on
# a pre-transformed array means every validation fold helped compute the
# statistics used to prepare it. Passing the Pipeline instead makes sklearn
# re-fit the transformer inside each fold, on that fold's training part only.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

clean_cv = cross_val_score(
    Pipeline(steps=[("prep", make_preprocessor()), ("model", make_model())]),
    X_train, y_train, cv=cv, scoring="accuracy",
)

X_train_pre = make_preprocessor().fit_transform(X_train)
leaky_cv = cross_val_score(make_model(), X_train_pre, y_train, cv=cv, scoring="accuracy")

print("\n" + "=" * 70)
print("2. PREPROCESSING BEFORE CROSS-VALIDATION")
print("=" * 70)
print(f"  Pipeline inside CV      : {clean_cv.mean():.4f} +/- {clean_cv.std():.4f}")
print(f"  Transformed once, then CV: {leaky_cv.mean():.4f} +/- {leaky_cv.std():.4f}")
print(f"  difference              : {leaky_cv.mean() - clean_cv.mean():+.4f}")
print("  per fold (clean -> leaky):")
for i, (c, l) in enumerate(zip(clean_cv, leaky_cv), start=1):
    print(f"    fold {i}: {c:.4f} -> {l:.4f}  ({l - c:+.4f})")


# ---------------------------------------------------------------------------
# 2b. The leak does not only inflate a score -- it suppresses an error
# ---------------------------------------------------------------------------
# This section was not planned. It came out of section 3 crashing on the first
# run, and it is the most useful thing in the script.
#
# `embarked` has three levels and Q is the rarest (77 of 891 rows). A small
# training set can miss it entirely. The correctly-fitted encoder then *raises*
# on the test set -- a loud, immediate, unmissable failure that tells you your
# training sample is not representative. The encoder fitted on all rows before
# the split already knows about Q, so it transforms the test set happily and
# reports an accuracy. The leak did not just make the number better; it deleted
# the evidence that something was wrong with the data.
print("\n" + "=" * 70)
print("2b. THE LEAK ALSO SUPPRESSES AN ERROR")
print("=" * 70)

# 30 training rows, and the first seed whose sample actually misses a level.
# Stratification is on the target, not on `embarked`, so whether Q survives is
# luck -- 6 of the 40 seeds in section 3 lost a category at this size.
for seed in range(100):
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    Xs_train, _, ys_train, _ = train_test_split(
        Xs_train, ys_train, train_size=30, stratify=ys_train, random_state=seed
    )
    if any(set(Xs_test[c].dropna()) - set(Xs_train[c].dropna()) for c in CATEGORICAL):
        break

print(f"  seed                    : {seed}")
print(f"  training embarked levels: {sorted(set(Xs_train['embarked'].dropna()))}")
print(f"  test embarked levels    : {sorted(set(Xs_test['embarked'].dropna()))}")

try:
    strict = Pipeline(steps=[("prep", make_preprocessor()), ("model", make_model())])
    strict.fit(Xs_train, ys_train)
    print(f"  correct Pipeline        : {strict.score(Xs_test, ys_test):.4f}")
except ValueError as exc:
    print(f"  correct Pipeline        : ValueError -- {exc}")

prep_leak = make_preprocessor().fit(X)
acc_quiet = (
    make_model()
    .fit(prep_leak.transform(Xs_train), ys_train)
    .score(prep_leak.transform(Xs_test), ys_test)
)
print(f"  leaked encoder          : {acc_quiet:.4f}  (no error raised)")


# ---------------------------------------------------------------------------
# 3. When does the leak stop being negligible?
# ---------------------------------------------------------------------------
# Leakage through a scaler or imputer is dilution-limited: the statistic is
# computed from train + test together, so the damage depends on how much the
# test rows can move it. With 712 training rows they move it very little. The
# sweep below shrinks the training set and repeats each size over 40 random
# splits, because at n=40 a single split's accuracy swings by more than the
# effect being measured and one number would be noise reported as a finding.
print("\n" + "=" * 70)
print("3. LEAK SIZE vs TRAINING-SET SIZE (40 random splits each)")
print("=" * 70)

SIZES = [30, 60, 120, 250, 500, len(X_train)]
N_REPEATS = 40
rows = []

for n_train in SIZES:
    diffs, cleans, leaks = [], [], []
    n_would_raise = 0
    for seed in range(N_REPEATS):
        # A fixed 179-row test set each time, so only the training size varies.
        Xa, Xb, ya, yb = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )
        if n_train < len(Xa):
            Xa, _, ya, _ = train_test_split(
                Xa, ya, train_size=n_train, stratify=ya, random_state=seed
            )

        # Did this training set miss a category the test set contains? Counted
        # rather than raised -- see section 2b for why this is the leak's most
        # under-appreciated effect.
        if any(set(Xb[c].dropna()) - set(Xa[c].dropna()) for c in CATEGORICAL):
            n_would_raise += 1

        pipe = Pipeline(
            steps=[("prep", make_preprocessor(tolerant=True)), ("model", make_model())]
        )
        pipe.fit(Xa, ya)
        acc_clean = pipe.score(Xb, yb)

        # The leak: statistics computed from this training set *and* the test
        # set it is about to be scored on.
        prep = make_preprocessor(tolerant=True).fit(pd.concat([Xa, Xb]))
        acc_leak = make_model().fit(prep.transform(Xa), ya).score(prep.transform(Xb), yb)

        cleans.append(acc_clean)
        leaks.append(acc_leak)
        diffs.append(acc_leak - acc_clean)

    rows.append(
        {
            "n_train": n_train,
            "clean": np.mean(cleans),
            "leaky": np.mean(leaks),
            "mean_gap": np.mean(diffs),
            "max_gap": np.max(np.abs(diffs)),
            "n_differed": int(np.sum(np.abs(diffs) > 1e-12)),
            "n_unseen_cat": n_would_raise,
        }
    )

sweep = pd.DataFrame(rows)
print(sweep.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\n  n_differed   = of {N_REPEATS} splits, how many gave a different answer at all.")
print(f"  n_unseen_cat = of {N_REPEATS} splits, how many had a test category the")
print("                 training set never saw -- the strict encoder raises there,")
print("                 the leaked one does not (section 2b).")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

labels = ["Pipeline\n(correct)", "Fitted before\nsplit", "Pipeline\nin CV", "Preprocessed\nthen CV"]
values = [clean_acc, leaky_split_acc, clean_cv.mean(), leaky_cv.mean()]
colors = ["#2a9d8f", "#e76f51", "#2a9d8f", "#e76f51"]
bars = axes[0].bar(labels, values, color=colors)
for bar, v in zip(bars, values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.4f}",
                 ha="center", fontsize=9)
axes[0].set_ylim(0.70, 0.85)
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Leak vs. no leak on the full Titanic split\n(left pair: test set; right pair: 5-fold CV)")
axes[0].grid(axis="y", alpha=0.3)

sizes = sweep["n_train"].to_numpy()
axes[1].plot(sizes, sweep["mean_gap"].to_numpy(), marker="o", color="#e76f51",
             label="mean gap (leaky - clean)")
axes[1].fill_between(sizes, 0, sweep["max_gap"].to_numpy(), alpha=0.15,
                     color="#e76f51", label="largest single-split gap")
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_xscale("log")
axes[1].set_xlabel("Training rows (log scale)")
axes[1].set_ylabel("Accuracy gap")
axes[1].set_title(f"How much the leak is worth, by training size\n({N_REPEATS} random splits per size)")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
save_and_show(fig, "leakage_cost.png")

print("\nSaved plots/leakage_cost.png")
