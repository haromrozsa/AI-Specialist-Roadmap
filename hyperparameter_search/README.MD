# ML Fundamentals – Hyperparameter Search: Grid vs Random vs Optuna

# Session Summary: Same Budget, Three Strategies, One MLflow Experiment

One script, two figures, one RandomForestClassifier on the Titanic split
already used across `logistic_regression/`, `trees_and_boosting/`,
`shap_explainability/` and `knn_and_naive_bayes/` (`test_size=0.2`,
stratified, `random_state=42`).

The backlog gap: exactly one `GridSearchCV` call exists anywhere in this
repository (`trees_and_boosting/`), and it was never compared against an
alternative search strategy, nor logged anywhere. This session asks the
obvious follow-up: **for the same number of model fits, does the search
strategy matter?**

All three methods get an identical budget — 18 parameter combinations,
5-fold cross-validation each (90 model fits, 270 total):

- **GridSearchCV** exhaustively enumerates a fixed 3 × 3 × 2 lattice
  (`n_estimators`, `max_depth`, `min_samples_split`) — it can only ever land
  exactly on those 18 points.
- **RandomizedSearchCV** draws 18 points uniformly at random from wider,
  partly continuous ranges — it can land between the grid's points, but each
  draw ignores everything learned from the previous ones.
- **Optuna** draws 18 points from the same wide ranges using its default TPE
  sampler, which *is* informed by earlier trials.

**Result on this run:** all three landed within 0.003 of each other on
best cross-validated accuracy (Grid 0.8328, Random 0.8314, Optuna 0.8301) and
identically at 0.7933 on the untouched test set. At only 18 trials on a
7-feature, 712-row training set the search space is small enough that none of
the three strategies is meaningfully handicapped — the difference between them
shows up in *how fast* they approach their best score (see the convergence
plot) and in wall-clock time, not in the final answer. That is itself the
lesson: smarter search buys the most when the space is larger or each
evaluation is expensive, neither of which is true here.

Every one of the 54 evaluated combinations (18 × 3 methods) is logged to
MLflow as its own run — params, mean/std CV accuracy, fit time — tagged by
`method`, plus one summary run per method with the winning pipeline attached
as an MLflow model. `mlflow ui` shows all three searches side by side.

## Running

```bash
cd hyperparameter_search
python HyperparameterSearch.py
mlflow ui   # then open http://localhost:5000
```

Requires `scikit-learn`, `numpy`, `pandas`, `seaborn`, `matplotlib`, `mlflow`
and `optuna`. Pulls the Titanic dataset over the network on first run via
`seaborn.load_dataset` (offline-capable afterwards from the local cache).
Runtime is under a minute; MLflow writes to a local `./mlruns` folder created
next to the script.

## Plots

- `01_best_score_comparison.png` — best CV accuracy and held-out test
  accuracy per method, side by side.
- `02_convergence.png` — best score found so far after each of the 18
  trials, for all three methods on the same axes.
