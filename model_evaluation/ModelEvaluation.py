"""
Model evaluation deep dive: ROC/AUC, calibration, and threshold selection.

Three questions you should ask of any classifier, and they are independent:

1. Ranking -- does it score positives above negatives?      -> ROC-AUC
2. Calibration -- when it says 0.8, is it right 80% of the time? -> calibration
   curve and Brier score
3. Decision -- which threshold turns probabilities into actions? -> chosen on a
   validation set, never on the test set

The script fits three models on the same data. Logistic regression is
usually well calibrated; Gaussian Naive Bayes is famously overconfident
(it pushes probabilities to 0 and 1). The Naive Bayes model is then
recalibrated with CalibratedClassifierCV to show that calibration can be fixed
without changing the ranking much.

Splits: 60% train / 20% validation / 20% test. The validation set picks the
threshold, the test set only reports the result.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, f1_score, log_loss,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

RANDOM_STATE = 42
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Data: 10,000 rows, 20% positive, three-way split -----------------------
X, y = make_classification(n_samples=10000, n_features=20, n_informative=5,
                           n_redundant=5, weights=[0.8], flip_y=0.02,
                           random_state=RANDOM_STATE)
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=RANDOM_STATE)
print(f"Train {len(y_train)} / validation {len(y_val)} / test {len(y_test)} rows, "
      f"{y.mean():.0%} positive\n")

# --- 1 + 2. Ranking and calibration ------------------------------------------
models = {
    "Logistic regression": LogisticRegression(max_iter=1000),
    "Random forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "Naive Bayes": GaussianNB(),
    # isotonic recalibration, fitted with internal CV on the training set only
    "Naive Bayes + isotonic": CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5),
}

print(f"{'Model':<24}{'ROC-AUC':>9}{'Brier':>8}{'Log loss':>10}")
test_proba = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    test_proba[name] = proba
    print(f"{name:<24}{roc_auc_score(y_test, proba):>9.4f}"
          f"{brier_score_loss(y_test, proba):>8.4f}{log_loss(y_test, proba):>10.4f}")
print("ROC-AUC: higher is better (ranking). Brier / log loss: lower is better "
      "(probability quality).\n")

# --- 3. Threshold selection on the VALIDATION set -----------------------------
# Business assumption: a missed positive (FN) costs 5x a false alarm (FP).
COST_FN, COST_FP = 5, 1
lr = models["Logistic regression"]
val_proba = lr.predict_proba(X_val)[:, 1]

thresholds = np.linspace(0.01, 0.99, 99)
val_f1 = [f1_score(y_val, val_proba >= t) for t in thresholds]
val_cost = [COST_FN * ((val_proba < t) & (y_val == 1)).sum()
            + COST_FP * ((val_proba >= t) & (y_val == 0)).sum() for t in thresholds]
fpr, tpr, roc_thr = roc_curve(y_val, val_proba)

chosen = {
    "Default 0.5": 0.5,
    "Max F1": thresholds[np.argmax(val_f1)],
    "Youden's J (max TPR-FPR)": roc_thr[np.argmax(tpr - fpr)],
    f"Min cost (FN={COST_FN}x FP)": thresholds[np.argmin(val_cost)],
}

print("Logistic regression, threshold picked on validation, scored on test:")
print(f"{'Rule':<28}{'Threshold':>10}{'Precision':>10}{'Recall':>8}{'F1':>7}{'Cost':>7}")
lr_test = test_proba["Logistic regression"]
for rule, t in chosen.items():
    pred = lr_test >= t
    cost = COST_FN * (~pred & (y_test == 1)).sum() + COST_FP * (pred & (y_test == 0)).sum()
    print(f"{rule:<28}{t:>10.3f}{precision_score(y_test, pred):>10.4f}"
          f"{recall_score(y_test, pred):>8.4f}{f1_score(y_test, pred):>7.4f}{cost:>7d}")

# --- Plot: ROC, calibration, threshold sweep ----------------------------------
fig, (ax_roc, ax_cal, ax_thr) = plt.subplots(1, 3, figsize=(18, 5.5))

for name, proba in test_proba.items():
    f, t, _ = roc_curve(y_test, proba)
    ax_roc.plot(f, t, label=f"{name} (AUC {roc_auc_score(y_test, proba):.3f})")
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    ax_cal.plot(mean_pred, frac_pos, "o-",
                label=f"{name} (Brier {brier_score_loss(y_test, proba):.3f})")
ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
ax_roc.set(xlabel="False positive rate", ylabel="True positive rate",
           title="ROC (test) -- ranking quality")
ax_cal.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
ax_cal.set(xlabel="Mean predicted probability", ylabel="Fraction actually positive",
           title="Calibration (test) -- is 0.8 really 80%?")

precision, recall, pr_thr = precision_recall_curve(y_val, val_proba)
ax_thr.plot(pr_thr, precision[:-1], label="Precision")
ax_thr.plot(pr_thr, recall[:-1], label="Recall")
ax_thr.plot(thresholds, val_f1, label="F1")
for (rule, t), color in zip(chosen.items(), ["k", "C3", "C4", "C5"]):
    ax_thr.axvline(t, color=color, linestyle=":", label=f"{rule}: {t:.2f}")
ax_thr.set(xlabel="Threshold", ylabel="Score",
           title="Threshold sweep (validation, logistic regression)")

for ax in (ax_roc, ax_cal, ax_thr):
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "evaluation.png"), dpi=120)
plt.show()
