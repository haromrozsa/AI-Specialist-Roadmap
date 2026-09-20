"""
Naive Bayes: what the "naive" costs, and where it stops mattering.

One script, five figures, four questions:

1. How does Gaussian Naive Bayes do on a real tabular problem? Run on the same
   Titanic split as logistic_regression/, trees_and_boosting/ and
   shap_explainability/ (test_size=0.2, stratify, random_state=42), so the
   number lands next to the ones already recorded in this repository.
2. What does the independence assumption actually break? Measured, by copying
   one feature N times and watching the model count the same evidence N times.
   The probabilities are where the damage is worst, but -- contrary to the usual
   summary -- accuracy does not get off lightly either.
3. Are those probabilities usable as probabilities? Checked with a calibration
   curve against logistic regression on the same split.
4. Where is Naive Bayes genuinely the right answer? On text, where the feature
   count is enormous, the data is sparse, and the model trains in milliseconds.
   Including the textbook claim that NB beats logistic regression at small
   training sizes and loses at large ones -- tested rather than repeated.

No Naive Bayes model here is scaled, and that is not an oversight. Unlike k-NN in
the sibling script, Naive Bayes fits one distribution per feature independently,
so rescaling a feature rescales that feature's fitted mean and variance
identically and the posterior comes out unchanged. StandardScaler is not wrong
here, it is inert. The logistic regressions it is compared against *are* scaled,
because they are replicas of the baseline already recorded in
trees_and_boosting/README.MD and that one has a scaler.
"""

import os
import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
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


# ---------------------------------------------------------------------------
# 1. Gaussian Naive Bayes on the repository's Titanic split
# ---------------------------------------------------------------------------
print("=" * 78)
print("NAIVE BAYES")
print("=" * 78)
print("\n" + "-" * 78)
print("1. GaussianNB on Titanic (same split as logistic_regression/)")
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
print(f"  Class balance (train): {y_train.value_counts().to_dict()}")

# GaussianNB fits a mean and a variance per feature per class, which is an
# outright lie for the one-hot columns below -- a 0/1 indicator is Bernoulli, not
# Gaussian. sklearn has BernoulliNB and CategoricalNB for exactly this, but no
# single estimator handles a mixed frame, so the usual practical choice is to
# push everything through GaussianNB and accept the mis-specification. It is
# worth naming rather than hiding: see the results section of the README for
# what it costs here.
preprocessor = ColumnTransformer(
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

nb_pipe = Pipeline([("prep", preprocessor), ("clf", GaussianNB())]).fit(
    X_train, y_train)

# The logistic regression here is not a fresh baseline, it is a replica of the
# one in trees_and_boosting/GradientBoostingComparison.py, StandardScaler
# included, so the number it produces is the 0.804 already recorded in that
# README rather than a near-miss. Dropping the scaler costs it one test row.
# GaussianNB gets no scaler because a scaler would not change a single one of
# its predictions -- see the module docstring.
lr_pipe = Pipeline([
    ("prep", preprocessor),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
]).fit(X_train, y_train)

nb_proba = nb_pipe.predict_proba(X_test)[:, 1]
lr_proba = lr_pipe.predict_proba(X_test)[:, 1]
nb_pred = nb_pipe.predict(X_test)
lr_pred = lr_pipe.predict(X_test)

results = {}
for name, pred, proba in [("GaussianNB", nb_pred, nb_proba),
                          ("LogisticRegression", lr_pred, lr_proba)]:
    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "log_loss": log_loss(y_test, proba),
        "brier": brier_score_loss(y_test, proba),
    }
    r = results[name]
    print(f"  {name:20s} accuracy {r['accuracy']:.4f}   ROC-AUC {r['roc_auc']:.4f}"
          f"   log-loss {r['log_loss']:.4f}   Brier {r['brier']:.4f}")

cm = confusion_matrix(y_test, nb_pred)
print(f"\n  GaussianNB confusion matrix (rows=true, cols=pred):\n{cm}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for name, proba in [("GaussianNB", nb_proba), ("LogisticRegression", lr_proba)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[0].plot(fpr, tpr, lw=2,
                 label=f"{name}  AUC={results[name]['roc_auc']:.3f}")
axes[0].plot([0, 1], [0, 1], "k:", lw=1)
axes[0].set_xlabel("false positive rate")
axes[0].set_ylabel("true positive rate")
axes[0].set_title("Titanic ROC, same split as the rest of the repo")
axes[0].legend()
axes[0].grid(alpha=0.3)

im = axes[1].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(["pred died", "pred survived"])
axes[1].set_yticklabels(["true died", "true survived"])
axes[1].set_title(f"GaussianNB confusion matrix\naccuracy "
                  f"{results['GaussianNB']['accuracy']:.3f}")
fig.colorbar(im, ax=axes[1], fraction=0.046)
fig.tight_layout()
save_and_show(fig, "01_titanic_gaussian_nb.png")

# ---------------------------------------------------------------------------
# 2. Breaking the independence assumption on purpose
# ---------------------------------------------------------------------------
# Naive Bayes multiplies per-feature likelihoods as if the features were
# conditionally independent given the class. Duplicate a feature and the same
# evidence is multiplied in twice, three times, N times.
#
# The usual summary of this -- "NB is a bad probability estimator but the ranking
# survives, so accuracy is fine" -- is only half right, and the run below shows
# which half. ROC-AUC, which depends only on the ordering, does hold up. Accuracy
# does not, because accuracy reads the ordering through a fixed 0.5 threshold and
# the saturating probabilities drag rows across it.
print("\n" + "-" * 78)
print("2. Independence broken: the same feature, copied N times")
print("-" * 78)

X_train_num = preprocessor.fit_transform(X_train)
X_test_num = preprocessor.transform(X_test)
# Column 3 of the numeric block is `fare`, the feature with the widest spread.
FARE_COL = NUMERICAL.index("fare")

dup_counts = [0, 1, 2, 4, 8, 16]
dup_rows = []
for d in dup_counts:
    if d == 0:
        Xtr_d, Xte_d = X_train_num, X_test_num
    else:
        Xtr_d = np.hstack([X_train_num] + [X_train_num[:, [FARE_COL]]] * d)
        Xte_d = np.hstack([X_test_num] + [X_test_num[:, [FARE_COL]]] * d)

    nb_d = GaussianNB().fit(Xtr_d, y_train)
    lr_d = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]).fit(Xtr_d, y_train)
    p_nb = nb_d.predict_proba(Xte_d)[:, 1]
    p_lr = lr_d.predict_proba(Xte_d)[:, 1]
    conf = np.maximum(p_nb, 1 - p_nb)

    dup_rows.append({
        "copies": d,
        "nb_acc": accuracy_score(y_test, nb_d.predict(Xte_d)),
        "nb_auc": roc_auc_score(y_test, p_nb),
        "nb_logloss": log_loss(y_test, p_nb),
        "nb_mean_conf": conf.mean(),
        "nb_over_99": int((conf > 0.99).sum()),
        "lr_acc": accuracy_score(y_test, lr_d.predict(Xte_d)),
        "lr_logloss": log_loss(y_test, p_lr),
    })
    r = dup_rows[-1]
    print(f"  fare x{d + 1:2d}   NB: acc {r['nb_acc']:.4f}  AUC {r['nb_auc']:.4f}  "
          f"log-loss {r['nb_logloss']:6.4f}  mean conf {r['nb_mean_conf']:.4f}  "
          f">0.99 in {r['nb_over_99']:3d}/{len(y_test)} rows   |   "
          f"LR: acc {r['lr_acc']:.4f}  log-loss {r['lr_logloss']:.4f}")

dup = pd.DataFrame(dup_rows)

copies = dup["copies"].to_numpy()
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
axes[0].plot(copies, dup["nb_auc"].to_numpy(), "s--", lw=2,
             label="GaussianNB ROC-AUC")
axes[0].plot(copies, dup["nb_acc"].to_numpy(), "o-", lw=2,
             label="GaussianNB accuracy")
axes[0].set_title("Ranking survives, the decision does not\n"
                  "AUC barely moves; accuracy falls with the threshold crossings")
axes[0].set_ylim(0.5, 1.0)
axes[1].plot(copies, dup["nb_logloss"].to_numpy(), "o-", lw=2, color="crimson",
             label="GaussianNB")
axes[1].plot(copies, dup["lr_logloss"].to_numpy(), "o-", lw=2, color="steelblue",
             label="LogisticRegression")
axes[1].set_title("What breaks worst: the probabilities\n"
                  "Log-loss punishes confident mistakes")
axes[2].plot(copies, dup["nb_mean_conf"].to_numpy(), "o-", lw=2,
             color="darkorange")
axes[2].set_ylim(0.5, 1.02)
axes[2].set_title("Why it breaks: the same evidence, counted N times\n"
                  "Mean confidence max(p, 1-p) marches to 1.0")
for ax in axes:
    ax.set_xlabel("extra copies of `fare` added")
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
fig.suptitle(
    "Logistic regression splits a duplicated feature's coefficient between the "
    "copies. Naive Bayes multiplies the evidence instead.",
    fontsize=12,
)
fig.tight_layout()
save_and_show(fig, "02_independence_broken.png")

# Correlated features are the everyday version of this. Nobody literally copies a
# column, but `fare` and `pclass` on Titanic carry much of the same information,
# and NB counts both at full weight.
corr = pd.DataFrame(X_train_num[:, :len(NUMERICAL)], columns=NUMERICAL).corr()
print("\n  Correlations among the numeric features actually used:")
print(corr.round(3).to_string())

# ---------------------------------------------------------------------------
# 3. Calibration: is a 0.9 from Naive Bayes worth 0.9?
# ---------------------------------------------------------------------------
print("\n" + "-" * 78)
print("3. Calibration on the Titanic test split")
print("-" * 78)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for name, proba, color in [("GaussianNB", nb_proba, "crimson"),
                           ("LogisticRegression", lr_proba, "steelblue")]:
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=8,
                                            strategy="quantile")
    axes[0].plot(mean_pred, frac_pos, "o-", lw=2, color=color,
                 label=f"{name}  Brier={brier_score_loss(y_test, proba):.3f}")
    axes[1].hist(proba, bins=20, alpha=0.55, color=color, label=name)
    print(f"  {name:20s} Brier {brier_score_loss(y_test, proba):.4f}   "
          f"fraction of predictions outside [0.05, 0.95]: "
          f"{np.mean((proba < 0.05) | (proba > 0.95)):.1%}")

axes[0].plot([0, 1], [0, 1], "k:", lw=1.5, label="perfectly calibrated")
axes[0].set_xlabel("mean predicted probability")
axes[0].set_ylabel("observed fraction positive")
axes[0].set_title("Reliability diagram (quantile bins)\n"
                  "Below the diagonal = overconfident")
axes[0].legend(fontsize=9)
axes[1].set_xlabel("predicted P(survived)")
axes[1].set_ylabel("count")
axes[1].set_title("Where the predictions actually sit\n"
                  "Naive Bayes piles up against 0 and 1")
axes[1].legend(fontsize=9)
for ax in axes:
    ax.grid(alpha=0.3)
fig.suptitle(
    "Naive Bayes is a classifier whose probabilities should not be read as "
    "probabilities without calibrating them first.",
    fontsize=12,
)
fig.tight_layout()
save_and_show(fig, "03_calibration.png")

# ---------------------------------------------------------------------------
# 4. Text: the case Naive Bayes was made for
# ---------------------------------------------------------------------------
# Thousands of sparse features, few rows, and a model that needs one pass over
# the data. This is where NB stops being a baseline and starts being a
# reasonable answer.
print("\n" + "-" * 78)
print("4. MultinomialNB on 20 newsgroups")
print("-" * 78)

CATEGORIES = ["rec.autos", "sci.space", "talk.politics.guns", "comp.graphics"]
train_txt = fetch_20newsgroups(
    subset="train", categories=CATEGORIES, shuffle=True,
    random_state=RANDOM_STATE, remove=("headers", "footers", "quotes"))
test_txt = fetch_20newsgroups(
    subset="test", categories=CATEGORIES, shuffle=True,
    random_state=RANDOM_STATE, remove=("headers", "footers", "quotes"))
print(f"  Categories: {CATEGORIES}")
print(f"  Train docs: {len(train_txt.data)}   Test docs: {len(test_txt.data)}")
print("  Headers, footers and quoted text removed -- otherwise the task is "
      "partly solvable from email signatures alone.")

vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english")
Xtr_txt = vectorizer.fit_transform(train_txt.data)
Xte_txt = vectorizer.transform(test_txt.data)
print(f"  Vocabulary: {Xtr_txt.shape[1]:,} features   "
      f"density {Xtr_txt.nnz / np.prod(Xtr_txt.shape):.4%}")

text_models = {
    "MultinomialNB": MultinomialNB(alpha=0.05),
    "ComplementNB": ComplementNB(alpha=0.05),
    "LogisticRegression": LogisticRegression(max_iter=2000,
                                             random_state=RANDOM_STATE),
}
text_results = {}
for name, model in text_models.items():
    t0 = time.perf_counter()
    model.fit(Xtr_txt, train_txt.target)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(Xte_txt)
    pred_s = time.perf_counter() - t0
    acc = accuracy_score(test_txt.target, pred)
    text_results[name] = {"accuracy": acc, "fit_s": fit_s, "predict_s": pred_s}
    print(f"  {name:20s} accuracy {acc:.4f}   fit {fit_s*1000:8.2f} ms   "
          f"predict {pred_s*1000:7.2f} ms")

speedup = (text_results["LogisticRegression"]["fit_s"]
           / text_results["MultinomialNB"]["fit_s"])
print(f"  MultinomialNB fits {speedup:.1f}x faster than logistic regression here.")

# The tokens the model leans on, read straight out of feature_log_prob_. NB's
# whole parameter set is one log-probability per (class, token), which is why it
# is the most directly readable model in this repository.
mnb = text_models["MultinomialNB"]
feature_names = np.array(vectorizer.get_feature_names())
top_tokens = {}
for idx, category in enumerate(train_txt.target_names):
    others = [j for j in range(len(train_txt.target_names)) if j != idx]
    score = mnb.feature_log_prob_[idx] - mnb.feature_log_prob_[others].mean(axis=0)
    top = np.argsort(score)[-10:][::-1]
    top_tokens[category] = list(zip(feature_names[top], score[top]))
    print(f"\n  Most distinctive tokens for {category}:")
    print("    " + ", ".join(feature_names[top]))

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
names = list(text_results)
accs = [text_results[n]["accuracy"] for n in names]
times = [text_results[n]["fit_s"] * 1000 for n in names]
xpos = np.arange(len(names))
axes[0].bar(xpos - 0.2, accs, 0.4, label="test accuracy", color="seagreen")
axes[0].set_xticks(xpos)
axes[0].set_xticklabels(names, fontsize=9)
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("test accuracy")
for i, v in enumerate(accs):
    axes[0].text(i - 0.2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
ax_t = axes[0].twinx()
ax_t.bar(xpos + 0.2, times, 0.4, label="fit time (ms)", color="slategray")
ax_t.set_ylabel("fit time, ms (log)")
ax_t.set_yscale("log")
for i, v in enumerate(times):
    ax_t.text(i + 0.2, v * 1.1, f"{v:.0f}ms", ha="center", fontsize=10)
axes[0].set_title("4-way newsgroup classification\n"
                  "Comparable accuracy, orders of magnitude apart in fit time")

cat = train_txt.target_names[1]
tokens, scores = zip(*top_tokens[cat][::-1])
axes[1].barh(range(len(tokens)), scores, color="darkslateblue")
axes[1].set_yticks(range(len(tokens)))
axes[1].set_yticklabels(tokens)
axes[1].set_xlabel("log P(token | class) - mean log P(token | other classes)")
axes[1].set_title(f"The whole model, readable: top tokens for {cat}")
axes[1].grid(alpha=0.3, axis="x")
fig.tight_layout()
save_and_show(fig, "04_text_multinomial_nb.png")

# ---------------------------------------------------------------------------
# 5. The small-data claim, tested
# ---------------------------------------------------------------------------
# Ng & Jordan (2001): a generative model like Naive Bayes reaches its (higher)
# asymptotic error faster, so it wins when training data is scarce and loses once
# there is enough of it for the discriminative model to converge. That predicts a
# crossing. Below is whether one appears on this task.
print("\n" + "-" * 78)
print("5. Textbook claim: NB wins on small training sets, loses on large ones")
print("-" * 78)

train_sizes = [20, 50, 100, 200, 400, 800, 1600, len(train_txt.data)]
SEEDS = [0, 1, 2]
nb_curve, lr_curve = [], []
for n in train_sizes:
    nb_runs, lr_runs = [], []
    for seed in SEEDS:
        if n >= len(train_txt.data):
            idx = np.arange(len(train_txt.data))
        else:
            idx, _ = train_test_split(
                np.arange(len(train_txt.data)), train_size=n,
                stratify=train_txt.target, random_state=seed)
        Xs, ys = Xtr_txt[idx], train_txt.target[idx]
        nb_runs.append(accuracy_score(
            test_txt.target, MultinomialNB(alpha=0.05).fit(Xs, ys).predict(Xte_txt)))
        lr_runs.append(accuracy_score(
            test_txt.target,
            LogisticRegression(max_iter=2000, random_state=seed)
            .fit(Xs, ys).predict(Xte_txt)))
    nb_curve.append(np.mean(nb_runs))
    lr_curve.append(np.mean(lr_runs))
    print(f"  n_train={n:5d}   MultinomialNB {nb_curve[-1]:.4f}   "
          f"LogisticRegression {lr_curve[-1]:.4f}   "
          f"gap {nb_curve[-1] - lr_curve[-1]:+.4f}")

gaps = np.array(nb_curve) - np.array(lr_curve)

# A crossing means the sign flips from + to - or back. An exact tie (gap == 0.0)
# is not a crossing, and must not be counted as one: np.sign(0.0) is 0, so a
# naive sign comparison reports a tie as *two* sign changes and contradicts the
# "LR ahead at" line directly below it. Ties are dropped before pairing.
nonzero = [(n, g) for n, g in zip(train_sizes, gaps) if g != 0]
crossings = [(nonzero[i][0], nonzero[i + 1][0])
             for i in range(len(nonzero) - 1)
             if np.sign(nonzero[i][1]) != np.sign(nonzero[i + 1][1])]
print(f"\n  NB ahead at: {[n for n, g in zip(train_sizes, gaps) if g > 0]}")
print(f"  LR ahead at: {[n for n, g in zip(train_sizes, gaps) if g < 0]}")
print(f"  Exact ties at: {[n for n, g in zip(train_sizes, gaps) if g == 0]}")
print(f"  Crossings: {crossings if crossings else 'none -- the curves never cross'}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
axes[0].plot(train_sizes, nb_curve, "o-", lw=2, color="crimson",
             label="MultinomialNB")
axes[0].plot(train_sizes, lr_curve, "o-", lw=2, color="steelblue",
             label="LogisticRegression")
axes[0].set_xscale("log")
axes[0].set_xlabel("training documents (log scale)")
axes[0].set_ylabel(f"test accuracy (mean of {len(SEEDS)} subsamples)")
axes[0].set_title("Learning curves on 4-way newsgroups")
axes[0].legend()
axes[1].axhline(0, color="k", lw=1.5)
axes[1].plot(train_sizes, gaps, "o-", lw=2, color="darkorange")
axes[1].fill_between(np.array(train_sizes), 0, gaps, where=gaps > 0, alpha=0.25,
                     color="crimson", label="NB ahead")
axes[1].fill_between(np.array(train_sizes), 0, gaps, where=gaps < 0, alpha=0.25,
                     color="steelblue", label="LR ahead")
axes[1].set_xscale("log")
axes[1].set_xlabel("training documents (log scale)")
axes[1].set_ylabel("NB accuracy - LR accuracy")
axes[1].set_title("The gap, signed\nA crossing is the claim; this is the test")
axes[1].legend()
for ax in axes:
    ax.grid(alpha=0.3)
fig.tight_layout()
save_and_show(fig, "05_learning_curve_nb_vs_lr.png")

print("\n" + "=" * 78)
print("Figures written to plots/")
print("=" * 78)
