# ML Fundamentals – Decision Trees, Random Forests & Gradient Boosting

# Session Summary: One Tree Overfits, A Hundred Trees Average, And Boosting Inverts Every Rule

Three scripts covering the tree family end to end, built around a dataset this
repository already models with logistic regression — so every accuracy figure has
something to sit beside instead of floating on its own.

The through-line is that the standard advice about these algorithms ("random forests
decorrelate the trees, so subsampling features helps", "boosting beats bagging",
"look at the feature importances") is made of claims that can be measured. Several of
them do not survive being measured on this dataset, and those are written up as they
came out rather than smoothed over. Where a textbook effect fails to appear, the
script measures the *mechanism* separately to show it is real, and says plainly why
the effect did not follow.

The gap this closes: `RandomForestClassifier` appears as a black box in three MLOps
projects here (`mlflow-experiment-tracking/`, `sagemaker-pipeline-demo/`,
`airflow-ml-pipeline/`) and the root README advertises a Random Forest result, but
nothing in the repository had ever explained what a decision tree is. There were zero
hits for `DecisionTreeClassifier`, `GradientBoosting`, or `xgboost`.

## Running

Each script is standalone and writes its figures to `plots/`.

```bash
cd trees_and_boosting
python DecisionTreeDepth.py
python RandomForestEnsemble.py
python GradientBoostingComparison.py
```

Requires `scikit-learn`, `xgboost`, `pandas`, `seaborn` and `matplotlib`. Titanic is
fetched by `seaborn.load_dataset('titanic')`, which needs network access on first use
and is cached afterwards.

Run headless with `MPLBACKEND=Agg`; every figure is saved before `plt.show()`, so a
headless run produces the identical artifacts.

## What I Did During the Session

**`DecisionTreeDepth.py`** — one tree, and how it fails.

- Drew the decision boundary on `make_moons` at `max_depth` 1, 3, 5 and unlimited, to
  show that every boundary a tree produces is a staircase of axis-aligned cuts, and
  that an unconstrained tree carves private boxes around individual noisy points
- Swept `max_depth` 1→15 on Titanic, tracking training, test and 5-fold
  cross-validated accuracy together
- Rendered the depth-3 tree with `plot_tree` and read the root split off it
- Ran cost-complexity pruning (`ccp_alpha`) as the principled alternative to guessing
  a depth
- Compared `gini` against `entropy`, and refit the repository's existing logistic
  regression on the identical split as a baseline

**`RandomForestEnsemble.py`** — why averaging works.

- Put the single tree's boundary beside a 100-tree forest's, shading `predict_proba`
  rather than hard labels so the averaging is visible
- Measured the variance directly: 20 bootstrap resamples, one tree and one forest
  each, comparing both accuracy spread and **pairwise prediction disagreement**
- Swept `n_estimators` 1→300 with `oob_score=True`, plotting out-of-bag error against
  test error
- Swept `max_features` 1→9 to test the decorrelation argument, then measured
  tree-to-tree disagreement inside single forests when the accuracy result came back
  flat
- Compared impurity-based `feature_importances_` against `permutation_importance` on
  the test set
- Deliberately leaked the target (`alive`) back into the features to record what
  leakage looks like from the outside

**`GradientBoostingComparison.py`** — sequential correction.

- Used `staged_predict_proba` to plot train/test log-loss across 500 rounds at three
  learning rates
- Repeated it across `max_depth` 1, 3 and 6 to test "boosting wants weak learners"
- Fitted XGBoost with `early_stopping_rounds` against a validation split carved out
  of the **training** data
- Put all six models on one split with accuracy, ROC-AUC, fit time and overlaid ROC
  curves — including gradient boosting twice, once with a guessed round count and
  once with a validated one

## Results

All three scripts use `train_test_split(test_size=0.2, stratify=y, random_state=42)`
on the 891-row Titanic set: **712 training rows, 179 test rows**, 38.4% survived.
Features are `pclass, sex, age, sibsp, parch, fare, embarked` — the same seven the
existing `logistic_regression/TitanicDatasetLogisticRegression.py` uses — which become
9 columns after one-hot encoding.

**One test row is worth 0.0056 accuracy.** Differences below about 0.02 between
models on this split are noise, and are treated as such below.

### A single tree (`DecisionTreeDepth.py`)

Depth sweep, 5-fold CV on the training set:

| max_depth | leaves | train | test | CV mean | CV std |
|---|---|---|---|---|---|
| 1 | 2 | 0.789 | 0.777 | 0.789 | 0.015 |
| 3 | 8 | 0.833 | 0.793 | 0.809 | 0.032 |
| **4** | **15** | **0.844** | **0.799** | **0.817** | **0.033** |
| 6 | 34 | 0.883 | 0.804 | 0.803 | 0.037 |
| 9 | 69 | 0.920 | 0.810 | 0.787 | 0.054 |
| 15 | 138 | 0.975 | 0.810 | 0.750 | 0.044 |

The three curves do not tell the same story, and the difference is the point:

- **training accuracy** climbs monotonically, 0.789 → 0.975, and keeps climbing. A
  deeper tree can always cut the training data finer, so this curve can never
  diagnose anything on its own.
- **CV accuracy** peaks at depth 4 (0.817) and decays to 0.750. This is the
  overfitting curve. The train-minus-CV gap widens from 0.000 to +0.225.
- **test accuracy** wanders between 0.760 and 0.810 with no clear peak — not a
  contradiction, just 179 rows. The entire visible range is about eight passengers.

That last row is the practical reason cross-validation exists. The depth-15 tree
scores 0.810 on this test split — as good as anything else in the table — while
having memorised the training set and posting the worst CV score of any depth. A
179-row held-out set is not powerful enough to catch it.

Root split: `sex_male <= 0.5`, gini 0.4728. A tree picks the split that drops impurity
most, so the root is the single most informative question available about a passenger.

Cost-complexity pruning found 69 candidate alphas; CV picked α = 0.00243, cutting
**153 leaves down to 22** at depth 9, for 0.879 train / **0.810 test**.

`gini` and `entropy` were indistinguishable at the selected depth (test 0.799 both, CV
0.817 vs 0.818, identical leaf count).

Baselines on the identical test split:

| Model | Test accuracy |
|---|---|
| Predict the majority class | 0.615 |
| Logistic regression | 0.804 |
| Tree, CV-selected depth 4 | 0.799 |
| Tree, cost-complexity pruned | 0.810 |
| Tree, depth 15 | 0.810 (train 0.975, CV 0.750) |

### The forest (`RandomForestEnsemble.py`)

**Variance, 20 bootstrap resamples:**

| | mean | std | min | max | range |
|---|---|---|---|---|---|
| 1 unlimited tree | 0.757 | 0.0280 | 0.709 | 0.821 | 0.112 |
| 100-tree forest | 0.790 | 0.0211 | 0.743 | 0.827 | 0.084 |

Accuracy std shrank only 1.3×, which *understates* the effect — a 179-row test set has
sampling noise of its own that inflates both figures equally. Comparing the models to
each other removes that floor:

- two single trees disagree on **19.2%** of test rows
- two forests disagree on **12.9%** (1.5× less)

That is what variance reduction means concretely: which rows you happened to train on
stops determining what you predict.

**`n_estimators` sweep with out-of-bag error:**

| n | OOB error | test error | train error |
|---|---|---|---|
| 1 | 0.338 | 0.335 | 0.110 |
| 10 | 0.230 | 0.207 | 0.024 |
| 30 | 0.185 | 0.196 | 0.017 |
| 100 | 0.201 | 0.179 | 0.017 |
| 300 | 0.195 | 0.179 | 0.017 |

Neither curve turns upward. Adding trees to a forest is an averaging budget, not a
capacity knob — it cannot overfit. OOB error stayed within 0.016 of test error at
n=300, which makes it a usable free validation estimate: each tree sees a bootstrap
sample, so ~37% of rows are out-of-bag for it and can be scored honestly with no split.

**`max_features` — the result that did not go as advertised.**

The textbook claim is that restricting features per split decorrelates the trees and
that this is what makes a *random* forest better than plain bagged trees. On this
dataset it produced nothing:

| max_features | OOB | test | |
|---|---|---|---|
| 1 | 0.802 | 0.804 | |
| 3 | 0.806 | 0.821 | ← `sqrt(9)`, sklearn's default |
| 5 | 0.816 | 0.816 | best OOB |
| 9 | 0.812 | 0.827 | plain bagging, no subsampling |

Total spread across the whole sweep: **0.014 OOB accuracy**. Plain bagging matched the
`sqrt` default and posted the best test score.

Rather than drop the section, the script asks the trees directly — pulling the
individual estimators out of a forest and measuring how often two of them predict
different labels:

- `max_features=3`: trees disagree on **22.2%** of test rows
- `max_features=9`: trees disagree on **19.3%**

The decorrelation is real and measurable. It just buys no accuracy here, and the
reason is the dataset rather than the theory: 9 columns, 5 of them one-hot dummies
with two or three usable splits between them. There is barely a redundant feature for
subsampling to route around. The technique pays off on wide, correlated tabular data —
hundreds of columns measuring overlapping things — which is where random forests
earned their reputation and is not this.

**Feature importance, measured two ways** (300 trees, OOB 0.805, test 0.821):

| feature | impurity | permutation | perm std |
|---|---|---|---|
| sex_male | 0.2667 | **0.1898** | 0.0284 |
| pclass_3 | 0.0661 | 0.0687 | 0.0148 |
| age | 0.2512 | 0.0663 | 0.0186 |
| fare | **0.2777** | 0.0609 | 0.0217 |
| parch | 0.0382 | 0.0223 | 0.0093 |
| sibsp | 0.0475 | 0.0020 | 0.0102 |
| pclass_2 | 0.0184 | −0.0020 | 0.0070 |

The two rankings disagree at the top:

- impurity: `fare`, `sex_male`, `age`, `pclass_3`
- permutation: `sex_male`, `pclass_3`, `age`, `fare`

`fare` is ranked most important by the measure everybody quotes and *fourth* by the
measure that tests whether the model actually needs it. Impurity importance is
computed on the training data and rewards features that had many chances to be
chosen — a continuous feature offers hundreds of candidate thresholds, a binary dummy
offers one. That is a bias toward high-cardinality features, not evidence of
usefulness.

**The leakage demonstration:**

| | test accuracy |
|---|---|
| Correct feature set | 0.8212 |
| With `alive` added | **1.0000** (OOB also 1.0000) |

`alive` took 0.7605 of the impurity importance; every other feature collapsed
(`sex_male` 0.1141, `fare` 0.0466, the rest below 0.03).

The signature worth recognising: a perfect or near-perfect score, one feature holding
essentially all the importance, everything else near zero. Cross-validation does not
catch it. OOB does not catch it. A held-out test set does not catch it — the leaked
column is in the test set too. The only thing that catches it is looking at your
columns and asking where each one would come from at prediction time. `alive` is only
knowable after the outcome being predicted.

### Boosting (`GradientBoostingComparison.py`)

**Round sweep, `max_depth=3`, 500 rounds:**

| learning_rate | best test log-loss | at round | at round 500 | degradation |
|---|---|---|---|---|
| 1.0 | 0.4652 | 14 | 1.6507 | **+1.1855** |
| 0.1 | 0.4541 | 22 | 0.6319 | +0.1778 |
| 0.05 | 0.4569 | 57 | 0.5673 | +0.1104 |

Every curve has a minimum and then climbs — the whole difference from the forest,
whose error curve simply flattens. A lower learning rate shrinks each correction, so
it takes more rounds to reach the minimum but degrades more slowly past it.
`learning_rate` and `n_estimators` are one hyperparameter wearing two hats.

**Depth sweep, `learning_rate=0.1`:**

| max_depth | best test log-loss | at round | at round 500 |
|---|---|---|---|
| 1 (stumps) | 0.4846 | 87 | 0.5001 |
| 3 | 0.4541 | 22 | 0.6319 |
| 6 | **0.4487** | 25 | 1.3410 |

This did not come out the way "boosting wants weak learners" is usually told. Depth 6
reached the *lowest* minimum — and then lost it completely, finishing three times
worse than it started. Its minimum lasts a moment, and collecting it requires knowing
exactly when to stop. Stumps never dip as far, but they end within 0.02 of their own
best: one split per tree is too weak to fit noise quickly, so the round count barely
matters.

The real trade is that depth buys a lower attainable optimum and a narrower window in
which to hit it. A random forest faces neither.

**XGBoost with early stopping:** the training set was split again (569 fit / 143
validation), leaving the test set untouched until final scoring. It stopped after
**62 of 500 rounds**, `best_iteration=41`, best validation log-loss 0.3909, in 0.05s.

**Head to head, identical split:**

| model | accuracy | ROC-AUC | fit time |
|---|---|---|---|
| Logistic regression | 0.804 | **0.842** | 0.008s |
| Decision tree (pruned) | 0.810 | 0.824 | 0.003s |
| Random forest (300) | **0.821** | 0.833 | 0.681s |
| Gradient boosting (200 rounds, guessed) | 0.804 | 0.806 | 0.188s |
| Gradient boosting (63 rounds, validated) | 0.788 | 0.830 | 0.092s |
| XGBoost (early stopped, 41) | 0.804 | 0.832 | 0.054s |

The two gradient boosting rows are the same algorithm with a different round count.
A guessed 200 — exactly the kind of round number people type — is already far past the
minimum the first plot located, and it costs 0.024 AUC. Validation picked 63 and cost
nothing to add. (Accuracy moves the other way, 0.788 vs 0.804, because accuracy at a
fixed 0.5 threshold on 179 rows is much noisier than AUC. AUC is the measure to read.)

Total spread across all six models: 0.034 accuracy — about **six passengers** — and
0.037 AUC. The honest conclusion is that on 891 rows with seven weak features, the
ensembles do not meaningfully beat the logistic regression this repository already
had, and the fastest model has the best AUC. That is not a failure of the ensembles;
it is what a small, low-dimensional, mostly-linear problem looks like. Boosting earns
its reputation on wide tabular data with interactions worth discovering. The reason to
learn it here is the mechanism, not the leaderboard.

### Plots produced

| File | Shows |
|---|---|
| `01_moons_depth_grid.png` | Tree boundaries at depth 1/3/5/unlimited — axis-aligned staircases, noise islands |
| `02_depth_vs_accuracy.png` | Train / test / CV accuracy vs `max_depth`, with the ±1σ CV band |
| `03_tree_depth3.png` | The depth-3 tree rendered with gini, sample counts, class distributions |
| `04_cost_complexity_pruning.png` | Accuracy and surviving leaf count vs `ccp_alpha` |
| `05_bagging_boundary.png` | One unlimited tree vs a 100-tree forest, shading `predict_proba` |
| `06_variance_reduction.png` | Test-accuracy spread over 20 bootstrap resamples, tree vs forest |
| `07_n_estimators_oob.png` | OOB / test / train error vs number of trees — flat, never upward |
| `08_max_features.png` | The decorrelation sweep that produced no accuracy effect |
| `09_importance_impurity_vs_permutation.png` | The two importance measures disagreeing at the top |
| `10_leakage_importance.png` | Correct feature set beside the leaked one — one bar eats the chart |
| `11_gb_deviance_learning_rate.png` | Train/test log-loss over 500 rounds at three learning rates |
| `12_gb_depth.png` | The same, over `max_depth` 1/3/6 |
| `13_xgb_early_stopping.png` | XGBoost's validation curve and where it stopped itself |
| `14_roc_comparison.png` | All six models' ROC curves on one axis |

## What I Learned

**A tree is a staircase.** Every split is a threshold on one feature, so every boundary
is axis-aligned. Anything a tree appears to draw smoothly is that staircase at finer
resolution — which is why a diagonal boundary costs a tree so many nodes, and why the
unlimited tree in `01` ends up with boxes around individual points.

**Training accuracy is not a diagnostic.** It rose monotonically to 0.975 while CV
accuracy fell to 0.750. Any curve that can only go up cannot tell you when to stop.

**A small test set cannot select hyperparameters.** The clearest lesson of the session.
Test accuracy was flat across depths 1–15 while CV showed a clean peak-and-decay,
because 179 rows put the noise floor above the effect. Selecting the depth that scored
best on the test set would have been fitting the test set by hand — and it would have
chosen depth 9, one of the worst by CV.

**Variance is better measured between models than through accuracy.** The accuracy
spread said 1.3×; prediction disagreement said 1.5× and is not contaminated by the test
set's own sampling noise. When an effect looks smaller than expected, check whether
the measurement has a floor under it.

**Bagging and boosting fail in opposite directions.** More trees in a forest is free —
the error curve flattens and stays flat. More rounds in a booster is a real risk — at
`learning_rate=1.0` the test loss more than tripled past its minimum. `n_estimators`
is a budget in one and a hyperparameter in the other, and the same word hides that.

**Forests want deep trees; boosters want shallow ones.** A forest relies on averaging
to cancel the variance depth costs. A booster has no averaging to hide behind, because
every tree's error is carried into the next round. Depth 6 boosted to the best minimum
and then collapsed; stumps held their loss for 500 rounds.

**Early stopping is not a convenience.** Once the round count matters and cannot be
read off the test set, something has to choose it from held-out data. XGBoost stopping
itself at 41 rounds and the validated gradient boosting at 63 rounds are the same idea;
the guessed 200 cost 0.024 AUC for nothing.

**The feature importance everybody quotes is the biased one.** `fare` ranked first by
impurity and fourth by permutation. Impurity importance is computed on the training
data and rewards features with many candidate split points. Permutation importance on
the test set answers the question people think they are asking.

**Leakage has a recognisable shape.** Perfect accuracy, one feature holding nearly all
the importance, everything else at zero — and no validation scheme catches it, because
the leaked column is in every split. Only reading your columns catches it.

**When a textbook effect does not appear, measure the mechanism separately.**
`max_features` moved accuracy by 0.014 across its whole range, which reads as "the
theory is wrong". Measuring tree-to-tree disagreement showed the decorrelation
happening exactly as described (22.2% vs 19.3%) — it simply had nothing to buy on 9
columns. The effect was absent; the mechanism was not.

**Ensembles are not automatically better.** Six models, six-passenger spread, and the
0.008-second logistic regression posted the best AUC. Worth knowing before reaching
for a forest by reflex.

## What Was Deliberately Left Out

| Left out | Why / where it belongs |
|---|---|
| SHAP values, summary and force plots | Backlog #3 — the natural sequel to the importance section here |
| GridSearchCV / RandomSearch / Optuna | Backlog #6. Depth and alpha were selected by hand-rolled CV loops on purpose, so the selection is visible rather than hidden in a search object |
| `Pipeline` + `ColumnTransformer` deep-dive | Backlog #7. Both are used here and kept leak-free, but leakage prevention as a topic belongs there |
| `class_weight`, SMOTE, precision-recall curves | Backlog #8. Titanic's 38/62 split makes this tempting, but it is a topic, not an aside |
| Calibration curves, threshold selection | Backlog #9. The 0.5 threshold is used throughout without comment except where it distorts a comparison |
| LightGBM | Not installed in this environment |
| CatBoost | Installed, but a third boosting library demonstrates no new concept |
| AdaBoost | Historically first, but gradient boosting is what is actually used; the reweighting story would have cost a section without changing the conclusions |
| Regression trees | Everything here is classification. `DecisionTreeRegressor` splits on variance instead of gini and is otherwise the same algorithm |
