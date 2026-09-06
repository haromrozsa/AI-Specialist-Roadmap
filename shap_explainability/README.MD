# ML Fundamentals – SHAP Explainability (TreeSHAP)

# Session Summary: A Feature Importance Cannot Explain A Prediction

One script, one class, four plots. It fits a random forest on the Titanic split that
`trees_and_boosting/` already uses — same features, same 80/20 stratified split at
`random_state=42` — and then explains it with TreeSHAP instead of ranking its features.

The gap this closes: `trees_and_boosting/RandomForestEnsemble.py` ends with impurity
("gain") and permutation importances, and both answer one narrow question — *across
the whole dataset, how much does this feature matter?* Neither can say anything about
a single passenger, and gain importance additionally carries no sign and no unit. It
is a by-product of how the trees were grown, not a decomposition of any prediction
the model actually made. There were zero hits for `shap` anywhere in the repository.

SHAP answers the other question. For one prediction it splits the distance between the
model's average output and this prediction's output into one number per feature, and
those numbers sum *exactly* to that distance. The global ranking then falls out for
free by averaging |SHAP| across rows, so the local and the global view come from one
computation rather than two unrelated ones.

## Running

```bash
cd shap_explainability
python ShapExplainability.py
```

Requires `shap`, `scikit-learn`, `pandas`, `seaborn` and `matplotlib`. Titanic is
fetched by `seaborn.load_dataset('titanic')`, which needs network access on first use
and is cached afterwards.

Run headless with `MPLBACKEND=Agg`; every figure is saved before `plt.show()`, so a
headless run produces the identical artifacts in `plots/`.

**Version note.** This was written against **shap 0.37**, which predates the modern
`shap.Explainer` / `Explanation` API. It uses `TreeExplainer`, `summary_plot`,
`force_plot` and `dependence_plot`, and a scikit-learn classifier's `shap_values()`
comes back as a **list `[class_0, class_1]`** rather than an `Explanation` object —
everything here indexes `[1]` (survived) and takes `explainer.expected_value[1]` as
the base value. On shap ≥ 0.41 the same four plots are reachable through
`shap.Explainer(model)(X)` and the modern `shap.plots.*` functions.

## What I Did During the Session

Everything lives in one class, `TitanicShapExplainer`, with a method per plot.

- **`fit()`** — loads Titanic, splits it, encodes it, fits a 300-tree forest at
  `max_depth=6`, and builds the `TreeExplainer`
- **`_encode()`** — imputes and one-hot encodes into a *named* `DataFrame`, with the
  imputer and encoder fitted on the training rows only
- **`_check_additivity()`** — verifies numerically that `base + sum(SHAP)` reproduces
  `predict_proba`, rather than asserting it
- **`plot_global_bar()`** — mean |SHAP| per feature: the global ranking
- **`plot_beeswarm()`** — the same ranking with the sign and per-passenger spread the
  bar chart averages away
- **`plot_force_single()`** — the local explanation, for the most confidently-predicted
  survivor in the test set
- **`plot_dependence()`** — how `age`'s effect changes with its value, auto-coloured
  by whatever interacts with it most

**One design decision worth stating.** The preprocessing is done by hand instead of
inside a `ColumnTransformer`. TreeSHAP explains the columns the model was actually
fitted on, so a pipeline that hands the forest an unnamed array produces plots labelled
`x0…x8`. Encoding into a named `DataFrame` is what makes `sex_male` and `pclass_3`
readable in every figure. The `fit`/`transform` split is preserved by hand for the same
reason it exists in a pipeline: the test rows must not reach the imputer's medians or
the encoder's categories.

## Results

Model being explained: test accuracy **0.8156**, test ROC-AUC **0.8414**, on 712 train
and 179 test rows with 9 encoded features.

### Additivity is exact, not approximate

```
base value (mean predicted P(survived) over the training rows): 0.385412
max |base + sum(SHAP) - predict_proba| over 179 test rows:      1.776e-15
```

That is floating-point noise. TreeSHAP walks the trees to compute exact Shapley values
in polynomial time — this is not the sampling approximation KernelSHAP would need for a
model it cannot see inside. The number is the reason SHAP is trusted where gain
importance is not: the explanation *reconstructs* the prediction.

### Global ranking (`01_shap_global_bar.png`)

| Feature | mean \|SHAP\| |
|---|---|
| `sex_male` | 0.1981 |
| `pclass_3` | 0.0707 |
| `fare` | 0.0621 |
| `age` | 0.0318 |
| `embarked_S` | 0.0256 |
| `sibsp` | 0.0165 |
| `parch` | 0.0085 |
| `pclass_2` | 0.0083 |
| `embarked_Q` | 0.0049 |

The ordering is unsurprising — sex dominates, as it does in every model of this dataset
in this repository. What is new is the **unit**: 0.1981 means *`sex_male` moves the
predicted survival probability by about 20 points on average*. Gain importance produces
a number with no unit at all (summed impurity decrease) and no scale on which two
models could be compared.

### The sign the bar chart destroys (`02_shap_beeswarm.png`)

```
mean SHAP for sex_male among men   : -0.1541
mean SHAP for sex_male among women : +0.2833
```

Same feature, opposite directions, and the bar chart shows one bar. The beeswarm keeps
one dot per passenger per feature, so the two clusters separate visibly. Note also that
the two magnitudes differ: the minority class (women, ~35% of the data) is pushed
further from the 0.385 base rate than the majority class is, which is a property of
where the base rate sits, not of the model treating them differently.

### One passenger (`03_shap_force_single.png`)

Test row 375, the most confidently-predicted survivor:
`pclass=1, sex=female, age=NaN→28.5 (train median), sibsp=1, parch=0, fare=82.1708, embarked=C`
— actually survived, predicted **P(survived) = 0.9781**.

```
  base value                 +0.3854
  sex_male                   +0.2994
  fare                       +0.1446
  pclass_3                   +0.0991
  embarked_S                 +0.0331
  sibsp                      +0.0114
  parch                      +0.0061
  pclass_2                   +0.0058
  age                        -0.0048
  embarked_Q                 -0.0021
  = prediction               +0.9781
```

Read that arithmetic: the model does not start from zero. It starts from the average
passenger's 0.3854 and this passenger's features carry it to 0.9781.

**A one-hot artefact worth understanding, not hiding.** `pclass_3 = 0.0` contributes
**+0.0991** and `embarked_S = 0.0` contributes **+0.0331**. SHAP attributes to the
*encoded column*, so what these say is "*not* being in third class helped" and "*not*
having embarked at Southampton helped". That is correct, but it means a one-hot model's
explanation is spread across the dummy columns and the plot reads slightly oddly:
a feature at value `0.0` can be the third-largest positive contribution. Native
categorical handling (CatBoost, or LightGBM's categorical mode) avoids the split;
`pclass` and `embarked` would then each get one number.

### The shape a single number cannot carry (`04_shap_dependence_age.png`)

| Age bucket | n | mean SHAP |
|---|---|---|
| [0, 10) | 12 | **+0.1659** |
| [10, 20) | 23 | +0.0213 |
| [20, 40) | 114 | −0.0147 |
| [40, 55) | 18 | −0.0217 |
| [55, 100) | 12 | −0.0752 |

`age` ranks fourth globally at 0.0318, which reads as "mildly useful". The buckets say
something quite different: nearly all of its contribution sits in the under-10 band,
and across the 114 passengers aged 20–40 — 64% of the test set — the effect is close to
nothing. The scatter shows the same thing: a steep positive spike below 10, then a flat
mildly-negative band, then a decline after 55. "Women and children first" is visible in
the plot at the point where it stops applying, somewhere around age 10.

The dependence plot auto-selected `sex_male` as the colouring, i.e. as the strongest
interaction with age — which is the mechanism behind that same phrase.

### Plots produced

| File | What it shows |
|---|---|
| `01_shap_global_bar.png` | mean \|SHAP\| per feature — the global ranking, in probability units |
| `02_shap_beeswarm.png` | one dot per passenger per feature, keeping sign and spread |
| `03_shap_force_single.png` | the local explanation for test row 375 |
| `04_shap_dependence_age.png` | SHAP for `age` against `age`, coloured by `sex_male` |

## What I Learned

- **Additivity is the whole argument.** Every other explainability method produces a
  plausible-looking ranking. SHAP produces numbers that add up to the prediction, and
  the 1.776e-15 reconstruction error is what separates "an explanation" from "a
  ranking that correlates with the explanation".
- **Global importance is a lossy summary of local explanations, and the losses matter.**
  Averaging |SHAP| throws away the sign (the beeswarm recovers it) and the shape (the
  dependence plot recovers it). Both losses would have hidden something real here.
- **"Importance" and "effect" are different questions.** `age` has a small average
  importance and a very large effect on twelve children. A model card citing only the
  ranking would understate what the model does to that group.
- **One-hot encoding fragments the explanation.** Not a defect in SHAP — a consequence
  of what the model was fitted on, and a concrete argument for native categorical
  support when explainability is a requirement.
- **TreeSHAP is cheap and exact only because the model is a tree ensemble.** For a
  model SHAP cannot see inside, KernelSHAP samples coalitions, costs far more, and
  returns an approximation. The exactness demonstrated here does not transfer.

## What Was Deliberately Left Out

- **KernelSHAP / `DeepExplainer`.** The model-agnostic and neural-network paths. The
  learning here is the Shapley decomposition itself, which TreeSHAP shows exactly and
  in a second.
- **Interaction values (`shap_interaction_values`).** The dependence plot's colouring
  already surfaces the age × sex interaction; the full n×n×features tensor is a
  separate topic.
- **The `force_plot` JavaScript view.** It requires a notebook and `initjs()`, and
  produces nothing a repository can commit. `matplotlib=True` gives the same
  decomposition as a static PNG.
- **A comparison against the impurity and permutation importances in
  `trees_and_boosting/plots/09_importance_impurity_vs_permutation.png`.** Deliberately
  kept as a separate question; the two directories now hold both halves of it.
- **Tests, type hints, logging.** Teaching-demo standard, per `CLAUDE.md`.
