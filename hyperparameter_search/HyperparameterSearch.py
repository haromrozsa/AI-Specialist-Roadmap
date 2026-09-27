"""
Hyperparameter search: GridSearchCV vs RandomizedSearchCV vs Optuna, on the
same equal budget, all logged to MLflow.

Only one `GridSearchCV` call exists anywhere else in this repository
(trees_and_boosting/), and it was never compared against an alternative
search strategy. This script asks the obvious follow-up question: for the
same number of model fits, does *how* you search the space matter?

Model and data: a RandomForestClassifier on the same Titanic split
(test_size=0.2, stratify, random_state=42) used throughout the repo, so the
accuracy is comparable to logistic_regression/, trees_and_boosting/,
shap_explainability/ and knn_and_naive_bayes/.

The budget is fixed at N_TRIALS=18 parameter combinations, 5-fold CV each,
for every method -- 90 model fits per method, 270 total. That makes the
comparison about search *strategy*, not compute spent:

* GridSearchCV can only evaluate points on a fixed lattice: three choices
  of n_estimators, three of max_depth, two of min_samples_split -- 18
  points, a complete enumeration of a small grid.
* RandomizedSearchCV draws 18 points uniformly at random from the same
  ranges, now continuous where GridSearchCV was discrete.
* Optuna's default sampler (TPE) draws 18 points too, but each one is
  informed by the trials that came before it -- it is not searching blind.

Every one of the 3 x 18 = 54 evaluated combinations is logged to MLflow as
its own run (params + mean/std CV accuracy + fit time), tagged by method, so
`mlflow ui` can show all three searches side by side. A summary run per
method logs the winning pipeline.
"""

import os
import time

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import seaborn as sns
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)

np.random.seed(42)
RANDOM_STATE = 42
N_TRIALS = 18  # equal budget for every method
CV_FOLDS = 5

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_and_show(fig, filename):
    """Write the figure to plots/ and then display it (no-op headless)."""
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=120, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 1. Data: same Titanic split and preprocessing as trees_and_boosting/
# ---------------------------------------------------------------------------
print("=" * 70)
print("HYPERPARAMETER SEARCH: GRID vs RANDOM vs OPTUNA")
print("=" * 70)

titanic = sns.load_dataset("titanic")
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
NUMERICAL = ["age", "sibsp", "parch", "fare"]
CATEGORICAL = ["pclass", "sex", "embarked"]

X = titanic[FEATURES]
y = titanic["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape}   Test: {X_test.shape}   budget: {N_TRIALS} fits x {CV_FOLDS}-fold CV each")

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


def make_pipeline(**rf_kwargs):
    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE, **rf_kwargs)),
        ]
    )


cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

mlflow.set_experiment("titanic-hyperparameter-search")


def log_cv_result(method, params, mean_score, std_score, fit_time, run_name):
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("method", method)
        mlflow.log_params(params)
        mlflow.log_metric("mean_cv_accuracy", mean_score)
        mlflow.log_metric("std_cv_accuracy", std_score)
        mlflow.log_metric("fit_time_seconds", fit_time)


def evaluate_and_log_best(method, best_params, elapsed, best_cv_score):
    """Refit the winning params on the full training set, score on the held-out
    test set, and log one summary run per method with the final pipeline."""
    model = make_pipeline(**best_params)
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))
    with mlflow.start_run(run_name=f"{method}_best"):
        mlflow.set_tag("method", method)
        mlflow.set_tag("summary", "true")
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_accuracy", best_cv_score)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("search_wall_time_seconds", elapsed)
        mlflow.sklearn.log_model(model, "model")
    print(
        f"  best params: {best_params}\n"
        f"  best CV accuracy: {best_cv_score:.4f}   test accuracy: {test_acc:.4f}   "
        f"wall time: {elapsed:.2f}s"
    )
    return test_acc


results = {}  # method -> dict(best_so_far list, best_cv_score, test_acc, elapsed)

# ---------------------------------------------------------------------------
# 2. GridSearchCV: exhaustive over a fixed 3 x 3 x 2 = 18-point lattice
# ---------------------------------------------------------------------------
print("\n--- GridSearchCV (18-point lattice) ---")
grid_params = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [3, 6, None],
    "clf__min_samples_split": [2, 5],
}

start = time.time()
grid_search = GridSearchCV(
    make_pipeline(), grid_params, cv=cv, scoring="accuracy", n_jobs=-1
)
grid_search.fit(X_train, y_train)
grid_elapsed = time.time() - start

grid_best_so_far = []
best_seen = -np.inf
for i in range(len(grid_search.cv_results_["params"])):
    raw_params = grid_search.cv_results_["params"][i]
    params = {k.replace("clf__", ""): v for k, v in raw_params.items()}
    mean_score = grid_search.cv_results_["mean_test_score"][i]
    std_score = grid_search.cv_results_["std_test_score"][i]
    fit_time = grid_search.cv_results_["mean_fit_time"][i]
    log_cv_result("grid", params, mean_score, std_score, fit_time, f"grid_{i}")
    best_seen = max(best_seen, mean_score)
    grid_best_so_far.append(best_seen)

grid_best_params = {k.replace("clf__", ""): v for k, v in grid_search.best_params_.items()}
grid_test_acc = evaluate_and_log_best(
    "grid", grid_best_params, grid_elapsed, grid_search.best_score_
)
results["Grid"] = dict(
    best_so_far=grid_best_so_far,
    best_cv=grid_search.best_score_,
    test_acc=grid_test_acc,
    elapsed=grid_elapsed,
)

# ---------------------------------------------------------------------------
# 3. RandomizedSearchCV: 18 uniform draws from continuous/wider ranges
# ---------------------------------------------------------------------------
print("\n--- RandomizedSearchCV (18 random draws, wider ranges) ---")
random_distributions = {
    "clf__n_estimators": randint(50, 350),
    "clf__max_depth": [3, 6, 10, None],
    "clf__min_samples_split": randint(2, 10),
}

start = time.time()
random_search = RandomizedSearchCV(
    make_pipeline(),
    random_distributions,
    n_iter=N_TRIALS,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
random_search.fit(X_train, y_train)
random_elapsed = time.time() - start

random_best_so_far = []
best_seen = -np.inf
for i in range(len(random_search.cv_results_["params"])):
    raw_params = random_search.cv_results_["params"][i]
    params = {k.replace("clf__", ""): v for k, v in raw_params.items()}
    mean_score = random_search.cv_results_["mean_test_score"][i]
    std_score = random_search.cv_results_["std_test_score"][i]
    fit_time = random_search.cv_results_["mean_fit_time"][i]
    log_cv_result("random", params, mean_score, std_score, fit_time, f"random_{i}")
    best_seen = max(best_seen, mean_score)
    random_best_so_far.append(best_seen)

random_best_params = {k.replace("clf__", ""): v for k, v in random_search.best_params_.items()}
random_test_acc = evaluate_and_log_best(
    "random", random_best_params, random_elapsed, random_search.best_score_
)
results["Random"] = dict(
    best_so_far=random_best_so_far,
    best_cv=random_search.best_score_,
    test_acc=random_test_acc,
    elapsed=random_elapsed,
)

# ---------------------------------------------------------------------------
# 4. Optuna: 18 trials from the same wide ranges, but each one informed by
#    the trials that came before it (TPE sampler, the library default)
# ---------------------------------------------------------------------------
print("\n--- Optuna (18 TPE-guided trials, same wide ranges) ---")


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 350),
        "max_depth": trial.suggest_categorical("max_depth", [3, 6, 10, None]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
    }
    t0 = time.time()
    model = make_pipeline(**params)
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    fit_time = (time.time() - t0) / CV_FOLDS
    log_cv_result("optuna", params, scores.mean(), scores.std(), fit_time, f"optuna_{trial.number}")
    return scores.mean()


start = time.time()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=N_TRIALS)
optuna_elapsed = time.time() - start

optuna_best_so_far = []
best_seen = -np.inf
for trial in study.trials:
    best_seen = max(best_seen, trial.value)
    optuna_best_so_far.append(best_seen)

optuna_test_acc = evaluate_and_log_best(
    "optuna", study.best_params, optuna_elapsed, study.best_value
)
results["Optuna"] = dict(
    best_so_far=optuna_best_so_far,
    best_cv=study.best_value,
    test_acc=optuna_test_acc,
    elapsed=optuna_elapsed,
)

# ---------------------------------------------------------------------------
# 5. Compare: does the search strategy matter at equal budget?
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY (equal budget: 18 fits x 5-fold CV = 90 model fits each)")
print("=" * 70)
print(f"{'Method':<10}{'Best CV acc':>14}{'Test acc':>12}{'Wall time (s)':>16}")
for method, r in results.items():
    print(f"{method:<10}{r['best_cv']:>14.4f}{r['test_acc']:>12.4f}{r['elapsed']:>16.2f}")

fig, ax = plt.subplots(figsize=(7, 5))
methods = list(results.keys())
best_cv_scores = [results[m]["best_cv"] for m in methods]
test_scores = [results[m]["test_acc"] for m in methods]
x = np.arange(len(methods))
width = 0.35
ax.bar(x - width / 2, best_cv_scores, width, label="Best CV accuracy", color="#4C72B0")
ax.bar(x + width / 2, test_scores, width, label="Held-out test accuracy", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylabel("Accuracy")
ax.set_ylim(0.75, 0.90)
ax.set_title("Same budget (18 fits x 5-fold CV), three search strategies")
ax.legend()
for i, (cv_s, test_s) in enumerate(zip(best_cv_scores, test_scores)):
    ax.annotate(f"{cv_s:.3f}", (x[i] - width / 2, cv_s), ha="center", va="bottom", fontsize=9)
    ax.annotate(f"{test_s:.3f}", (x[i] + width / 2, test_s), ha="center", va="bottom", fontsize=9)
fig.tight_layout()
save_and_show(fig, "01_best_score_comparison.png")

fig, ax = plt.subplots(figsize=(8, 5))
trial_numbers = np.arange(1, N_TRIALS + 1)
ax.plot(trial_numbers, results["Grid"]["best_so_far"], marker="o", label="Grid (fixed lattice)")
ax.plot(trial_numbers, results["Random"]["best_so_far"], marker="s", label="Random (uniform draws)")
ax.plot(trial_numbers, results["Optuna"]["best_so_far"], marker="^", label="Optuna (TPE-guided)")
ax.set_xlabel("Trial number")
ax.set_ylabel("Best CV accuracy found so far")
ax.set_title("Convergence: best score seen after each of the 18 trials")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
save_and_show(fig, "02_convergence.png")

print("\nAll runs logged to MLflow experiment 'titanic-hyperparameter-search'.")
print("View with: mlflow ui   (run from this directory)")
