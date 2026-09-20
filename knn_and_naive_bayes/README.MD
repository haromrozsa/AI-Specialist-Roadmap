# ML Fundamentals – k-Nearest Neighbours & Naive Bayes

# Session Summary: The Lazy Learner Isn't, And The Baseline Won

Two scripts, twelve figures. These are the two models everyone reaches for as a
*baseline* — the thing you run first so you have a number to beat — and this
session took both of them seriously enough to find out where that reputation is
earned and where it is wrong.

Two results came out against expectation, and both are written up below as they
came out rather than smoothed over:

- **sklearn's k-NN is not a lazy learner by default.** "k-NN does no work at
  training time" is a statement about the algorithm, not about the library.
  `algorithm="auto"` picked a kd_tree, whose `fit()` time grew **554×** between
  n=500 and n=128,000 while its `predict()` time grew only 2.6× over the same
  256× increase in data.
- **The textbook small-data crossing never happened.** Naive Bayes is supposed to
  beat logistic regression when data is scarce and lose once there is enough of
  it. MultinomialNB led at *every* training size tested, from 20 documents to the
  full 2,317, and finished ahead on the complete task — while fitting **105×
  faster**.

The gap this closes: the backlog listed k-NN and Naive Bayes as the last open
item of the classical-ML foundation (task #5), with both absent from the
repository. Regression, trees, forests, boosting, SVMs, KMeans and PCA were all
covered; the two standard baselines were not.

## Running

```bash
cd knn_and_naive_bayes
python KNNClassification.py
python NaiveBayesClassification.py
```

Requires `scikit-learn`, `numpy`, `pandas`, `seaborn` and `matplotlib`. Both
scripts pull their data over the network on first run — `seaborn.load_dataset`
for Titanic, `fetch_20newsgroups` for the text task — and are offline-capable
afterwards from the local cache.

Run headless with `MPLBACKEND=Agg`; every figure is saved before `plt.show()`, so
a headless run produces the identical artifacts.

**A note on runtime.** `KNNClassification.py` is slow, and deliberately so in one
place: every decision-boundary panel is a full k-NN query per pixel. The mesh
resolution is set to 160 rather than 300 for exactly this reason — see the
`plot_boundary` docstring.

## Why two different datasets

`trees_and_boosting/` and `shap_explainability/` both reused the Titanic split so
their numbers were comparable. `svm_kernels/` deliberately broke from that and
used 2-D synthetic data, because an SVM's whole idea is the shape of its
boundary. This session needed both, so it does both:

- **k-NN geometry** — `make_moons`, 2-D, because "which training points are near"
  is a statement about a picture.
- **Titanic** — the same seven features, the same `test_size=0.2`, the same
  `stratify=y`, the same `random_state=42` as
  `logistic_regression/TitanicDatasetLogisticRegression.py` and everything built
  on it. Both models run on it, so both land in the repository's running
  comparison table instead of being lone figures.
- **20 newsgroups** — for Naive Bayes only, because the tabular result alone
  would have libelled the model. Text is where it is actually the right answer.

## What I Did During the Session

**k-NN**

- **Drew k as a smoothing knob**, at k = 1, 5, 15, 50, and made the point that it
  runs *backwards*: small k is the complex model
- **Swept k = 1..100** with train, test and 5-fold CV accuracy on one axis, to
  show that the training curve starts at 1.000 for free and is therefore useless
  for selection
- **Broke the feature scale on purpose** (one feature × 1000) and measured the
  cost, then measured what putting `StandardScaler` back recovers
- **Compared `uniform` against `distance` weighting**, and showed *structurally*
  why distance weighting pins training accuracy at 1.000 for every k: a training
  point is its own nearest neighbour at distance 0, which is infinite weight
- **Timed `fit` against `predict`** across five training-set sizes and *both*
  neighbour algorithms — which is where the "lazy learner" claim fell over
- **Measured the curse of dimensionality** as two curves rather than one: the
  accuracy collapse, and the distance concentration underneath it that causes it
- **Tuned k, weighting and the distance metric** on the Titanic split with a
  5-fold `GridSearchCV`, then ran the same hyperparameters without a scaler

**Naive Bayes**

- **Ran GaussianNB on the Titanic split** against a replica of the logistic
  regression already recorded in `trees_and_boosting/README.MD` — `StandardScaler`
  included, so the baseline reproduces its 0.804 exactly rather than landing one
  test row away
- **Broke the independence assumption by hand**, copying one feature up to 17
  times and tracking accuracy, AUC, log-loss and mean confidence separately,
  because they do not degrade together
- **Checked the calibration** with a reliability diagram and a histogram of the
  predicted probabilities
- **Ran MultinomialNB, ComplementNB and logistic regression on 20 newsgroups**,
  with headers, footers and quotes stripped, timing the fits
- **Read the model out loud** — NB's entire parameter set is one log-probability
  per (class, token), so the most distinctive tokens per class come straight out
  of `feature_log_prob_`
- **Tested the Ng & Jordan small-data claim** with learning curves over eight
  training sizes, averaged across three subsamples each

Every preprocessing step in both scripts lives inside a `Pipeline`, so imputation,
encoding and scaling are refitted from each fold's training rows during
cross-validation rather than leaking the fold's own distribution into itself.

## Results — k-NN

All numbers from the committed run: `make_moons`, n=300, noise=0.25, 210 train /
90 test.

### k is a smoothing knob that runs backwards

| k | train | test |
|---|---|---|
| 1 | **1.0000** | 0.9222 |
| 5 | 0.9286 | 0.9000 |
| 15 | 0.9190 | 0.8889 |
| 50 | 0.9190 | 0.9000 |

The k=1 training accuracy is 1.0000 and always will be — the nearest neighbour of
a training point is itself, at distance zero. **A training score is not evidence
about a k-NN model**, which is the practical reason the sweep below reports
cross-validation.

**k sweep, k = 1..100:** CV picked **k=18** (CV 0.9333, test 0.9000). At the
extremes, k=1 gives train 1.0000 / test 0.9222 / CV 0.8905, and k=100 — nearly
half the 210-row training set voting on every prediction — gives train 0.8238 /
test 0.8222 / CV 0.8048.

Worth noticing: the CV-selected k=18 scores 0.9000 on test, while k=1 scores
0.9222. On 90 test rows one row is worth 0.0111, so that is a two-row difference
and not evidence that k=1 is better. The CV curve is the one to trust.

### Scaling is not optional for a distance model

k=15 throughout, one feature multiplied by 1000:

| setup | test accuracy |
|---|---|
| Comparable scales, no scaler | 0.8889 |
| Feature 2 × 1000, no scaler | **0.7889** |
| Feature 2 × 1000, `StandardScaler` | 0.9000 |

Ten points of accuracy, recovered in full by one pipeline step. The middle row is
not a *worse* boundary — it is a boundary in a different space. When one axis is
1000× larger, the other contributes essentially nothing to the Euclidean
distance, so the model reads horizontal stripes through data whose structure is
two interleaved crescents.

### Distance weighting buys almost nothing here

| weighting | best test | at k | training accuracy across all k |
|---|---|---|---|
| `uniform` | **0.9444** | 6 | varies |
| `distance` | 0.9333 | 4 | **1.0000 – 1.0000** |

Mean test difference across k = 1..60: **+0.0031** in favour of `distance` — well
inside the noise on 90 test rows.

The training-accuracy column is the structural point rather than a measurement
artefact. With `weights="distance"` a training point sits at distance 0 from
itself, sklearn treats that as infinite weight, and the training score is
therefore pinned at 1.000 *for every k*. It is a second, independent reason not
to select k on training accuracy.

### The lazy learner isn't — the headline result

1000 query rows against a growing training set, both neighbour algorithms:

| n_train | kd_tree fit | kd_tree predict | brute fit | brute predict |
|---|---|---|---|---|
| 500 | 1.42 ms | 175 ms | 0.89 ms | 316 ms |
| 2,000 | 2.83 ms | 322 ms | 1.61 ms | 592 ms |
| 8,000 | 18.06 ms | 494 ms | 2.15 ms | 1,074 ms |
| 32,000 | 168 ms | 406 ms | 14.19 ms | 3,507 ms |
| 128,000 | **788 ms** | 455 ms | 47.05 ms | **12,360 ms** |

Across a **256× increase in training data**:

- **kd_tree `fit` grew 554×** — superlinear, because it is building a spatial index
- **kd_tree `predict` grew 2.59×** — the index doing its job
- **brute `predict` grew 39.1×** — the linear scan the textbook describes
- **brute `fit` stayed cheap** — 47 ms at 128,000 rows, against 12.4 seconds to
  predict against them. It is not literally flat (it grew 53×, since storing and
  validating the array still scales with n), but it is 17× cheaper than building
  the kd_tree and 260× cheaper than one batch of predictions. *This* is the lazy
  learner the description means

So the familiar sentence is describing `algorithm="brute"`. sklearn's default is
`algorithm="auto"`, which on low-dimensional data picks kd_tree and moves the cost
straight back to training time. Both rows are correct; only one of them is what
runs when you type `KNeighborsClassifier()`.

**Caveat on the absolute milliseconds.** On the machine these were measured on, a
`predict()` call carries roughly 0.3–0.5 ms of fixed per-query-row overhead, so
the kd_tree predict column is sitting on a floor rather than showing the index's
true asymptotic cost. The *shapes* transfer; the milliseconds do not.

### The curse of dimensionality, and the mechanism under it

Two informative features, unchanged throughout. Everything added is pure noise:

| noise dims | test accuracy | mean(nearest / farthest) |
|---|---|---|
| 0 | 0.9000 | 0.036 |
| 2 | 0.8667 | 0.122 |
| 5 | 0.8222 | 0.230 |
| 10 | 0.7778 | 0.346 |
| 20 | 0.8000 | 0.481 |
| 50 | 0.7556 | 0.607 |
| 100 | **0.6778** | 0.714 |
| 200 | 0.7111 | **0.790** |

The right-hand column is the part worth keeping. It is the mean ratio of each test
point's *nearest* training point to its *farthest* one. At 2 dimensions the
nearest neighbour is 28× closer than the farthest; at 200 it is 1.27× closer.
"Nearest" stops carrying information, and the model built entirely on that word
degrades with it.

The accuracy column is noisy — it goes *up* from 100 to 200 dimensions, and up
from 10 to 20. On 90 test rows those are three- and two-row wobbles. The distance
ratio, which is measured over all 18,900 test-train pairs, is monotone and is the
cleaner evidence for the same claim.

### Titanic, tuned

| | value |
|---|---|
| Best params | `k=21`, Manhattan (`p=1`), `uniform` |
| CV accuracy (train split) | 0.8217 |
| **Test accuracy** | **0.7821** |
| Test ROC-AUC | 0.8272 |
| Same hyperparameters, no scaler | **0.6872** |

**The scaler is worth +0.0950 on real data** — 17 of 179 test rows. This is the
synthetic section-3 result reproduced on a dataset nobody constructed to make the
point: `fare` runs 0–512 while `sibsp` runs 0–8, so without standardisation the
neighbour set is chosen almost entirely by ticket price.

The top of the grid is flat — the best five combinations span 0.8147 to 0.8217 CV
accuracy — and every one of them uses `uniform` weighting. Manhattan edged
Euclidean at the top (0.8217 vs 0.8203), by less than one CV row.

## Results — Naive Bayes

### Titanic, against the number already on record

| model | accuracy | ROC-AUC | log-loss | Brier |
|---|---|---|---|---|
| GaussianNB | 0.7877 | 0.8065 | 0.7007 | 0.1749 |
| Logistic regression | **0.8045** | **0.8422** | **0.4629** | **0.1448** |

The logistic regression reproduces the 0.804 / 0.842 recorded in
`trees_and_boosting/README.MD` exactly, which confirms the split is identical and
the comparison is honest. GaussianNB loses on every metric, and loses by much more
on the probability metrics (log-loss 1.5×) than on accuracy (three test rows).

Part of that is a mis-specification worth naming: GaussianNB fits a mean and a
variance per feature, which is simply the wrong distribution for the one-hot
columns — a 0/1 indicator is Bernoulli. sklearn has `BernoulliNB` and
`CategoricalNB`, but no single estimator handles a mixed frame, so the common
practical move is to push everything through GaussianNB and accept the lie.

### What "naive" actually costs

Copying `fare` N extra times, so the same evidence is counted N+1 times:

| copies of `fare` | NB accuracy | NB ROC-AUC | NB log-loss | mean confidence | rows with p > 0.99 | LR accuracy |
|---|---|---|---|---|---|---|
| 0 | 0.7877 | 0.8065 | 0.7007 | 0.8726 | 20 / 179 | 0.8045 |
| 1 | 0.7709 | 0.8026 | 0.9240 | 0.9026 | 26 / 179 | 0.8045 |
| 2 | 0.7709 | 0.7984 | 1.1099 | 0.9130 | 72 / 179 | 0.8045 |
| 4 | 0.6760 | 0.7942 | 1.5735 | 0.9324 | 105 / 179 | 0.8045 |
| 8 | 0.6536 | 0.7885 | 2.6401 | 0.9894 | 144 / 179 | 0.8045 |
| 16 | **0.6480** | 0.7817 | **4.7211** | **0.9965** | **174 / 179** | **0.8045** |

Three different things happen at three different rates, and the usual one-line
summary of Naive Bayes only captures one of them:

- **ROC-AUC barely moves** (0.8065 → 0.7817). It depends only on the *ordering* of
  the predictions, and raising every posterior to a power is close to
  order-preserving. This is the half of the folklore that holds.
- **Accuracy does not get off lightly** (0.7877 → 0.6480, a 14-point fall).
  Accuracy reads that ordering through a fixed 0.5 threshold, and the saturating
  probabilities drag rows across it. I expected this to hold up and it did not.
- **Log-loss explodes** (0.70 → 4.72) and by 16 copies **174 of 179 predictions
  are more confident than 99%**. This is the real damage.

Logistic regression, given exactly the same duplicated columns, does not move at
all — 0.8045 accuracy and 0.4629 log-loss on every row of the table. It splits the
coefficient between the copies; Naive Bayes multiplies the evidence.

Nobody duplicates a column by accident, but correlation is the everyday version of
the same thing, and Titanic has it: `sibsp`/`parch` correlate at 0.392.

### The probabilities are not probabilities

| model | Brier | predictions outside [0.05, 0.95] |
|---|---|---|
| GaussianNB | 0.1749 | **42.5%** |
| Logistic regression | 0.1448 | 4.5% |

Even with no duplication at all, GaussianNB puts **more than two in five**
predictions into the top or bottom 5% of the probability range, against one in
twenty for logistic regression. Use it to rank or to classify; calibrate it before
anyone reads a number off it as a likelihood.

### Text: where Naive Bayes is actually the right answer

4-way newsgroup classification (`rec.autos`, `sci.space`, `talk.politics.guns`,
`comp.graphics`), headers/footers/quotes stripped, 2,317 train / 1,543 test,
12,837 TF-IDF features at 0.44% density:

| model | test accuracy | fit time | predict time |
|---|---|---|---|
| **MultinomialNB** | **0.8924** | **45.6 ms** | 13.9 ms |
| ComplementNB | 0.8769 | 15.7 ms | 19.3 ms |
| Logistic regression | 0.8691 | 4,804 ms | 7.6 ms |

**MultinomialNB wins on accuracy and fits 105× faster.** On this problem it is not
a baseline to beat — it is the better model, and it is the one that would still be
retrainable on every commit.

It is also the most readable model in this repository. Its entire parameter set is
one log-probability per (class, token), so the distinctive vocabulary falls
straight out of `feature_log_prob_`:

| class | top tokens |
|---|---|
| `comp.graphics` | 3d, graphics, vga, animation, tiff, polygon, vesa, pov, format, viewer |
| `rec.autos` | toyota, honda, car, ford, tires, cars, wagon, dealer, convertible, sho |
| `sci.space` | moon, orbit, shuttle, lunar, spacecraft, mars, orbital, billion, mary, ssto |
| `talk.politics.guns` | fbi, firearms, guns, amendment, nra, assault, firearm, jmd, handheld, handgun |

Mostly clean topical vocabulary, and two leaks worth admitting rather than
cropping: `jmd` and `mary` are not topic words. They are almost certainly initials
and a name from prolific posters, surviving `remove=("headers","footers","quotes")`
because they appear in the message bodies. Stripping headers removes the *obvious*
author signal, not all of it.

### The textbook claim that did not reproduce

Ng & Jordan (2001): a generative model like Naive Bayes reaches its (higher)
asymptotic error faster, so it should win when data is scarce and lose once there
is enough of it. That predicts the two learning curves crossing.

Mean test accuracy over three subsamples per size:

| training docs | MultinomialNB | Logistic regression | gap |
|---|---|---|---|
| 20 | 0.5273 | 0.4448 | **+0.0825** |
| 50 | 0.6353 | 0.5463 | **+0.0890** |
| 100 | 0.7166 | 0.6647 | +0.0518 |
| 200 | 0.7680 | 0.7680 | 0.0000 |
| 400 | 0.8257 | 0.8038 | +0.0218 |
| 800 | 0.8540 | 0.8345 | +0.0194 |
| 1,600 | 0.8812 | 0.8622 | +0.0190 |
| 2,317 (all) | 0.8924 | 0.8691 | +0.0233 |

**No crossing.** Logistic regression is never ahead at any training size, and the
script reports it as such: `LR ahead at: []`, `Crossings: none`.

What *does* reproduce is the first half of the mechanism. The gap is largest where
data is scarcest (+0.089 at n=50) and shrinks by roughly four-fifths as data
arrives (+0.019 by n=1,600). That is exactly the shape the argument predicts — the
discriminative model closing on the generative one as n grows. It simply never
finishes the job here, because this task tops out at 2,317 documents and NB starts
too far ahead.

The honest reading is that the claim is about asymptotics and this dataset does not
reach them. A larger corpus is where the crossing would have to be looked for, and
I have not shown it does not exist — only that it is not in this range.

One detail: at n=200 the two models tie to four decimal places. A tie is not a
crossing, and the first version of the crossing detector called it two of them,
because `np.sign(0.0)` is `0` and differs from both `+1` and `-1`. That printed
"sign changes" directly above "LR ahead at: []", contradicting itself. Ties are now
dropped before pairing.

## Figures

**k-NN** — `01_k_boundary_grid` (k = 1/5/15/50 boundaries) · `02_k_sweep`
(train/test/CV over k = 1..100) · `03_scaling_matters` (one feature × 1000, with
and without a scaler) · `04_weighting` (uniform vs distance) ·
`05_lazy_learner_timing` (fit and predict, kd_tree vs brute) ·
`06_curse_of_dimensionality` (accuracy collapse and distance concentration) ·
`07_titanic_comparison` (the CV grid, and k-NN against the repo's other models)

**Naive Bayes** — `01_titanic_gaussian_nb` (ROC and confusion matrix) ·
`02_independence_broken` (what survives, what breaks, and why) · `03_calibration`
(reliability diagram and probability histogram) · `04_text_multinomial_nb`
(accuracy vs fit time, and top tokens) · `05_learning_curve_nb_vs_lr` (the claim
that did not reproduce)

## Where both models now sit in this repository

Same Titanic split, same seven features, same seed, across every model this
repository has built on it:

| model | test accuracy | where |
|---|---|---|
| Random forest (300) | **0.821** | `trees_and_boosting/` |
| Decision tree (pruned) | 0.810 | `trees_and_boosting/` |
| Logistic regression | 0.804 | `logistic_regression/` |
| **GaussianNB** | **0.788** | here |
| **k-NN (tuned, scaled)** | **0.782** | here |
| k-NN (tuned, unscaled) | 0.687 | here |

Both of this session's models land at the bottom, and that is the useful result
rather than a disappointing one. It is what a baseline is *for*: on 712 training
rows of mixed categorical and continuous features, the two cheapest models get
within 4 points of the ensemble, which is how you find out whether the ensemble is
earning its complexity. The text task is the counterweight — there, the same
"baseline" beat logistic regression outright.

## What I'd do next

- **Calibrate the GaussianNB** with `CalibratedClassifierCV` and re-measure the
  Brier score, closing the loop on the calibration section rather than only
  diagnosing it
- **Push the learning curve past 2,317 documents** on a larger corpus, to find the
  crossing this dataset could not reach
- **Try `BernoulliNB` on the one-hot Titanic columns** and a mixed-model split, to
  measure what the Gaussian mis-specification actually costs rather than just
  naming it
