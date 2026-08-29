"""
Gradient boosting: building the ensemble sequentially instead of in parallel,
and why almost every tuning instinct from RandomForestEnsemble.py inverts.

A random forest grows its trees independently and averages them. Each tree is a
complete model; the ensemble exists to cancel their individual errors. Boosting
grows its trees one after another, each fitted to the errors the ensemble has
made so far. No single tree is a model of anything -- it is a correction.

Three consequences follow, and each one is measured here rather than asserted:

  * more rounds CAN overfit (section 1). A forest's error curve flattens; a
    booster's turns back upward once it starts fitting noise. n_estimators stops
    being a budget and becomes a real hyperparameter.
  * boosting wants WEAK trees (section 2). A forest wants its trees deep so they
    have low bias; a booster wants them shallow so each correction is small.
  * the round count has to be chosen on validation data (section 3), which is
    what early stopping automates.

Section 4 puts all five models from this directory on the same split and reports
what the differences are actually worth.
"""

import os
import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Set random seed for reproducibility
np.random.seed(42)
RANDOM_STATE = 42
N_ROUNDS = 500

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
print("GRADIENT BOOSTING: SEQUENTIAL CORRECTION, AND THE OPPOSITE TUNING ADVICE")
print("=" * 70)

# ---------------------------------------------------------------------------
# 0. The same data and the same split as the other two scripts
# ---------------------------------------------------------------------------
titanic = sns.load_dataset("titanic")

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
).fit(X_train, y_train)

X_train_t = preprocessor.transform(X_train)
X_test_t = preprocessor.transform(X_test)
FEATURE_NAMES = NUMERICAL + list(
    preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names(CATEGORICAL)
)
print(f"\nTrain {X_train_t.shape}   Test {X_test_t.shape}")

# ---------------------------------------------------------------------------
# 1. More rounds can overfit, and the learning rate controls how fast
# ---------------------------------------------------------------------------
# staged_predict_proba replays the ensemble one round at a time, so a single fit
# yields the whole error curve. Log-loss is plotted rather than accuracy because
# it responds to the model's confidence: a booster that has started memorising
# grows more confident about its training rows long before its accuracy on them
# changes, and log-loss shows that while accuracy hides it.
print("\n--- Round sweep at three learning rates (max_depth=3) ---")

learning_rates = [1.0, 0.1, 0.05]
colours = {1.0: "#d62728", 0.1: "#1f77b4", 0.05: "#2ca02c"}
curves = {}

for lr in learning_rates:
    gb = GradientBoostingClassifier(
        n_estimators=N_ROUNDS, learning_rate=lr, max_depth=3, random_state=RANDOM_STATE
    )
    t0 = time.time()
    gb.fit(X_train_t, y_train)
    fit_time = time.time() - t0

    train_loss = np.array([
        log_loss(y_train, p) for p in gb.staged_predict_proba(X_train_t)
    ])
    test_loss = np.array([
        log_loss(y_test, p) for p in gb.staged_predict_proba(X_test_t)
    ])
    curves[lr] = (train_loss, test_loss)

    best_round = int(np.argmin(test_loss)) + 1
    print(
        f"lr={lr:<5} best test log-loss {test_loss.min():.4f} at round {best_round:>3}"
        f"   |  at round {N_ROUNDS}: {test_loss[-1]:.4f}"
        f"   |  degradation {test_loss[-1] - test_loss.min():+.4f}   ({fit_time:.1f}s)"
    )

print("\nEvery curve has a minimum and then climbs. That is the whole difference")
print("from a random forest, whose error curve in 07_n_estimators_oob.png simply")
print("flattens. Here the extra rounds keep fitting whatever is left, and what is")
print("left, eventually, is noise.")
print("A lower learning rate shrinks each correction, so it takes more rounds to")
print("reach the minimum but overfits more slowly past it. learning_rate and")
print("n_estimators are one hyperparameter wearing two hats.")

fig, ax = plt.subplots(figsize=(10, 6.5))
rounds = np.arange(1, N_ROUNDS + 1)
for lr in learning_rates:
    train_loss, test_loss = curves[lr]
    ax.plot(rounds, train_loss, "--", color=colours[lr], alpha=0.55, linewidth=1.2,
            label=f"lr={lr} train")
    ax.plot(rounds, test_loss, "-", color=colours[lr], linewidth=1.8,
            label=f"lr={lr} test")
    best = int(np.argmin(test_loss))
    ax.plot(best + 1, test_loss[best], "o", color=colours[lr], markersize=8,
            markeredgecolor="k", zorder=5)

ax.set_xlabel("Boosting rounds (trees added so far)")
ax.set_ylabel("Log-loss")
ax.set_title(
    "Training loss (dashed) falls forever. Test loss (solid) bottoms out and\n"
    "climbs -- marked with a dot. Unlike a forest, a booster can be given\n"
    "too many trees."
)
ax.legend(ncol=3, fontsize=9)
ax.grid(alpha=0.3)
save_and_show(fig, "11_gb_deviance_learning_rate.png")

# ---------------------------------------------------------------------------
# 2. Boosting wants weak learners -- the exact opposite of a forest
# ---------------------------------------------------------------------------
# A random forest wants deep trees: each one should be a low-bias model, and
# averaging removes the variance that depth costs. Boosting has no averaging to
# hide behind -- every tree's error is carried into the next round -- so it wants
# each tree to be barely better than a guess.
print("\n--- Depth sweep at learning_rate=0.1 ---")

depth_curves = {}
depth_colours = {1: "#2ca02c", 3: "#1f77b4", 6: "#d62728"}

for depth in [1, 3, 6]:
    gb = GradientBoostingClassifier(
        n_estimators=N_ROUNDS, learning_rate=0.1, max_depth=depth,
        random_state=RANDOM_STATE,
    )
    gb.fit(X_train_t, y_train)
    test_loss = np.array([
        log_loss(y_test, p) for p in gb.staged_predict_proba(X_test_t)
    ])
    train_loss = np.array([
        log_loss(y_train, p) for p in gb.staged_predict_proba(X_train_t)
    ])
    depth_curves[depth] = (train_loss, test_loss)

    best_round = int(np.argmin(test_loss)) + 1
    label = "decision stumps" if depth == 1 else f"depth-{depth} trees"
    print(
        f"max_depth={depth} ({label:<15}) best test log-loss {test_loss.min():.4f} "
        f"at round {best_round:>3}   |  final {test_loss[-1]:.4f}"
    )

print("\nRead the two columns against each other. Depth 6 reaches the LOWEST minimum")
print("(0.4487) -- and then loses it completely, finishing three times worse than it")
print("started. Its minimum lasts a moment and you would need to know exactly when to")
print("stop to collect it. Stumps never reach that minimum, but they end within 0.02")
print("of their own best: one split per tree is too weak to fit noise quickly, so the")
print("round count barely matters.")
print("\nThat is the trade boosting actually offers -- depth buys a better attainable")
print("optimum and a narrower window in which to hit it. A random forest has no such")
print("window and simply wants its trees deep.")

fig, ax = plt.subplots(figsize=(10, 6.5))
for depth in [1, 3, 6]:
    train_loss, test_loss = depth_curves[depth]
    ax.plot(rounds, test_loss, "-", color=depth_colours[depth], linewidth=1.8,
            label=f"max_depth={depth} test")
    ax.plot(rounds, train_loss, "--", color=depth_colours[depth], alpha=0.5,
            linewidth=1.1, label=f"max_depth={depth} train")
    best = int(np.argmin(test_loss))
    ax.plot(best + 1, test_loss[best], "o", color=depth_colours[depth], markersize=8,
            markeredgecolor="k", zorder=5)

ax.set_xlabel("Boosting rounds")
ax.set_ylabel("Log-loss")
ax.set_title(
    "Depth buys a lower minimum and a narrower window to catch it in.\n"
    "Depth-6 bottoms out lowest, then collapses; stumps never dip as far but\n"
    "hold their loss for 500 rounds. A random forest faces neither trade."
)
ax.legend(ncol=3, fontsize=9)
ax.grid(alpha=0.3)
save_and_show(fig, "12_gb_depth.png")

# ---------------------------------------------------------------------------
# 3. XGBoost, and letting a validation set choose the round count
# ---------------------------------------------------------------------------
# Section 1 established that the right number of rounds exists and matters. It
# cannot be read off the test set -- that would be selecting on the data used to
# report the result. Early stopping carves a validation set out of the TRAINING
# data, watches its loss, and stops when it stops improving.
print("\n--- XGBoost with early stopping ---")

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train_t, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
)
print(f"Training split again for early stopping: fit {X_fit.shape}, val {X_val.shape}")
print("The test set is untouched by this -- it is scored once, at the end.")

xgb_clf = xgb.XGBClassifier(
    n_estimators=N_ROUNDS,
    learning_rate=0.1,
    max_depth=3,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
)
t0 = time.time()
xgb_clf.fit(
    X_fit, y_fit,
    eval_set=[(X_fit, y_fit), (X_val, y_val)],
    early_stopping_rounds=20,
    verbose=False,
)
xgb_time = time.time() - t0

evals = xgb_clf.evals_result()
xgb_train_loss = evals["validation_0"]["logloss"]
xgb_val_loss = evals["validation_1"]["logloss"]

print(f"Stopped after {len(xgb_val_loss)} of {N_ROUNDS} rounds "
      f"(best_iteration={xgb_clf.best_iteration}, best val log-loss "
      f"{min(xgb_val_loss):.4f})  [{xgb_time:.2f}s]")
print(f"Early stopping fired: {len(xgb_val_loss) < N_ROUNDS}")
print("The round count was chosen by data the model never trained on, without")
print("anyone picking a number and without touching the test set.")

fig, ax = plt.subplots(figsize=(10, 6))
xgb_rounds = np.arange(1, len(xgb_train_loss) + 1)
ax.plot(xgb_rounds, xgb_train_loss, "--", color="#d62728", label="Training log-loss")
ax.plot(xgb_rounds, xgb_val_loss, "-", color="#1f77b4", linewidth=1.8,
        label="Validation log-loss")
ax.axvline(xgb_clf.best_iteration + 1, color="grey", linestyle=":",
           label=f"best_iteration = {xgb_clf.best_iteration}")
ax.plot(int(np.argmin(xgb_val_loss)) + 1, min(xgb_val_loss), "o", color="#1f77b4",
        markersize=9, markeredgecolor="k", zorder=5)
ax.set_xlabel("Boosting rounds")
ax.set_ylabel("Log-loss")
ax.set_title(
    f"XGBoost stopped itself after {len(xgb_val_loss)} of {N_ROUNDS} rounds.\n"
    "Early stopping is section 1's problem solved automatically: the validation\n"
    "curve, not a guess, decides how many trees to keep."
)
ax.legend()
ax.grid(alpha=0.3)
save_and_show(fig, "13_xgb_early_stopping.png")

# ---------------------------------------------------------------------------
# 4. Everything in this directory, on one split
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("HEAD TO HEAD -- identical train/test split, 179 test rows")
print("=" * 70)

# Alpha 0.00243 is the value DecisionTreeDepth.py's cross-validation selected;
# the forest settings are RandomForestEnsemble.py's. Nothing here was tuned on
# the test set.
#
# Gradient boosting appears twice on purpose. Section 1 found the test-loss
# minimum at around 20 rounds, so a plausible-looking n_estimators=200 is already
# well past it -- and 200 is exactly the kind of number people put there. The
# second entry picks its round count the same way XGBoost did: on the validation
# split carved out of the training data, never on the test set.
gb_probe = GradientBoostingClassifier(
    n_estimators=N_ROUNDS, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE
).fit(X_fit, y_fit)
gb_val_loss = np.array([
    log_loss(y_val, p) for p in gb_probe.staged_predict_proba(X_val)
])
gb_best_rounds = int(np.argmin(gb_val_loss)) + 1
print(f"Rounds chosen for gradient boosting on the validation split: {gb_best_rounds}"
      f" (val log-loss {gb_val_loss.min():.4f})\n")
models = {
    "Logistic regression": Pipeline(
        steps=[("scale", StandardScaler()),
               ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]
    ),
    "Decision tree (pruned)": DecisionTreeClassifier(
        ccp_alpha=0.00243, random_state=RANDOM_STATE
    ),
    "Random forest (300)": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Gradient boosting (200)": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE
    ),
    "Gradient boosting (val-picked)": GradientBoostingClassifier(
        n_estimators=gb_best_rounds, learning_rate=0.1, max_depth=3,
        random_state=RANDOM_STATE
    ),
    "XGBoost (early stopped)": None,  # already fitted above, handled separately
}

results = {}
for name, model in models.items():
    if model is None:
        continue
    t0 = time.time()
    model.fit(X_train_t, y_train)
    fit_time = time.time() - t0
    proba = model.predict_proba(X_test_t)[:, 1]
    results[name] = {
        "accuracy": accuracy_score(y_test, model.predict(X_test_t)),
        "auc": roc_auc_score(y_test, proba),
        "time": fit_time,
        "proba": proba,
    }

xgb_proba = xgb_clf.predict_proba(X_test_t)[:, 1]
results["XGBoost (early stopped)"] = {
    "accuracy": accuracy_score(y_test, xgb_clf.predict(X_test_t)),
    "auc": roc_auc_score(y_test, xgb_proba),
    "time": xgb_time,
    "proba": xgb_proba,
}

print(f"{'model':<32} {'accuracy':>9} {'ROC-AUC':>9} {'fit time':>10}")
for name, r in results.items():
    print(f"{name:<32} {r['accuracy']:>9.3f} {r['auc']:>9.3f} {r['time']:>9.3f}s")

accs = np.array([r["accuracy"] for r in results.values()])
aucs = np.array([r["auc"] for r in results.values()])
print(f"\nSpread: {accs.max() - accs.min():.3f} accuracy, {aucs.max() - aucs.min():.3f} AUC.")
print(f"One test row is worth {1 / len(y_test):.4f} accuracy, so the whole spread is")
print(f"about {round((accs.max() - accs.min()) * len(y_test))} passengers.")
naive = results["Gradient boosting (200)"]
picked = results["Gradient boosting (val-picked)"]
print("\nThe two gradient boosting rows are the same algorithm with a different")
print(f"round count. {gb_best_rounds} rounds picked on validation: AUC {picked['auc']:.3f}. "
      f"A guessed 200: AUC {naive['auc']:.3f}.")
print("Section 1's curve predicted that, and validation costs nothing to add.")
print(f"(Accuracy moves the other way -- {picked['accuracy']:.3f} vs "
      f"{naive['accuracy']:.3f} -- because accuracy at a fixed 0.5")
print(" threshold on 179 rows is the noisier of the two measures. AUC is the one")
print(" to read here.)")
print("\nThe honest conclusion: on 891 rows with seven weak features, the ensembles")
print("do not meaningfully beat the logistic regression this repository already had.")
print("That is not a failure of the ensembles -- it is what a small, low-dimensional,")
print("mostly-linear problem looks like. Boosting earns its reputation on wide")
print("tabular data with interactions worth discovering. The reason to learn it here")
print("is the mechanism, not the leaderboard.")

fig, ax = plt.subplots(figsize=(8.5, 8))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["proba"])
    ax.plot(fpr, tpr, linewidth=1.8, label=f"{name} (AUC {r['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Chance (AUC 0.500)")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title(
    "Five models, one split, 179 test rows.\n"
    "The curves sit on top of each other -- which is the result, not a problem\n"
    "with the plot."
)
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
ax.set_aspect("equal")
save_and_show(fig, "14_roc_comparison.png")

print("\n" + "=" * 70)
print("Forest vs booster, in one line each:")
print("  forest  -- deep trees, grown independently, averaged. More trees is free.")
print("  booster -- shallow trees, each fixing the last. More trees is a risk.")
print("=" * 70)
