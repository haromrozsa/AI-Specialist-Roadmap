"""
Imbalanced classification: why accuracy lies, and what to look at instead.

A dataset where only 1% of rows are positive (fraud, disease, defects). The
script compares four approaches on the same stratified split:

1. A dummy model that always predicts "negative" -- the 99% accuracy trap.
2. Plain logistic regression.
3. Logistic regression with class_weight="balanced" -- each minority row
   counts ~99x more in the loss.
4. Logistic regression trained on SMOTE-oversampled data. SMOTE is written by
   hand below (imbalanced-learn is not needed): it creates synthetic minority
   points on the line between a minority row and one of its minority neighbours.

Key rule: resampling is applied to the TRAINING set only. The test set keeps
its real 1% prevalence, otherwise every metric is measured on a world that
does not exist.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

RANDOM_STATE = 42
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def smote(X, y, k=5, random_state=RANDOM_STATE):
    """Oversample the minority class (label 1) until both classes are equal size."""
    rng = np.random.RandomState(random_state)
    X_min = X[y == 1]
    n_new = (y == 0).sum() - len(X_min)

    # k+1 because each point's nearest neighbour is itself
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    _, neighbours = nn.kneighbors(X_min)

    base = rng.randint(0, len(X_min), n_new)               # random minority row
    partner = neighbours[base, rng.randint(1, k + 1, n_new)]  # one of its neighbours
    gap = rng.rand(n_new, 1)                               # position along the line
    X_new = X_min[base] + gap * (X_min[partner] - X_min[base])

    return np.vstack([X, X_new]), np.concatenate([y, np.ones(n_new, dtype=int)])


# --- Data: 20,000 rows, 1% positive -----------------------------------------
X, y = make_classification(n_samples=20000, n_features=10, n_informative=4,
                           weights=[0.99], flip_y=0, class_sep=1.5,
                           n_clusters_per_class=1,
                           random_state=RANDOM_STATE)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)

print(f"Train: {len(y_train)} rows, {y_train.sum()} positive ({y_train.mean():.2%})")
print(f"Test:  {len(y_test)} rows, {y_test.sum()} positive ({y_test.mean():.2%})")

X_smote, y_smote = smote(X_train, y_train)
print(f"After SMOTE (train only): {len(y_smote)} rows, {y_smote.mean():.0%} positive\n")

# --- Models -------------------------------------------------------------------
models = {
    "Always negative": (DummyClassifier(strategy="most_frequent"), X_train, y_train),
    "Plain LR": (LogisticRegression(max_iter=1000), X_train, y_train),
    "class_weight=balanced": (LogisticRegression(max_iter=1000, class_weight="balanced"),
                              X_train, y_train),
    "SMOTE + LR": (LogisticRegression(max_iter=1000), X_smote, y_smote),
}

print(f"{'Model':<24}{'Accuracy':>9}{'Precision':>10}{'Recall':>8}{'F1':>7}"
      f"{'ROC-AUC':>9}{'PR-AUC':>8}")
scores = {}
for name, (model, X_fit, y_fit) in models.items():
    model.fit(X_fit, y_fit)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    scores[name] = proba
    print(f"{name:<24}"
          f"{accuracy_score(y_test, pred):>9.4f}"
          f"{precision_score(y_test, pred, zero_division=0):>10.4f}"
          f"{recall_score(y_test, pred):>8.4f}"
          f"{f1_score(y_test, pred):>7.4f}"
          f"{roc_auc_score(y_test, proba):>9.4f}"
          f"{average_precision_score(y_test, proba):>8.4f}")

print("\nAccuracy/precision/recall/F1 use the default 0.5 threshold.")
print("ROC-AUC and PR-AUC are threshold-free (they rank the probabilities).")
print(f"A random model scores ROC-AUC 0.5 but PR-AUC only {y_test.mean():.4f} (the prevalence).")

# --- Plot: ROC vs PR curves -----------------------------------------------------
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5))
for name, proba in scores.items():
    if name == "Always negative":
        continue
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, proba):.3f})")
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ax_pr.plot(recall, precision,
               label=f"{name} (AP {average_precision_score(y_test, proba):.3f})")

ax_roc.plot([0, 1], [0, 1], "k--", label="Random (0.5)")
ax_roc.set(xlabel="False positive rate", ylabel="True positive rate (recall)",
           title="ROC curve -- looks excellent")
ax_pr.axhline(y_test.mean(), color="k", linestyle="--",
              label=f"Random ({y_test.mean():.3f})")
ax_pr.set(xlabel="Recall", ylabel="Precision",
          title="Precision-recall curve -- the honest view")
for ax in (ax_roc, ax_pr):
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
fig.suptitle("Same models, same test set, 1% positive class")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "roc_vs_pr.png"), dpi=120)
plt.show()
