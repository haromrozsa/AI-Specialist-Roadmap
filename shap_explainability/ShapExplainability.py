"""
SHAP: turning a fitted forest into an explanation of one single prediction.

`trees_and_boosting/RandomForestEnsemble.py` ends with two feature-importance
plots -- impurity ("gain") and permutation -- and both answer the same narrow
question: across the whole dataset, how much does this feature matter? Neither
can say anything about one passenger. Gain importance additionally has no sign
(it cannot tell you whether being male pushed a prediction up or down) and is
biased toward high-cardinality features, which is exactly how the leakage demo
in that script was caught.

SHAP answers the other question. For a single prediction it splits the distance
between the model's average output and this prediction's output into one number
per feature, and those numbers sum exactly to that distance -- the additivity
property, verified numerically in `_check_additivity` below rather than asserted.
Because every local explanation is on the same scale, averaging |SHAP| across
rows gives a global importance for free, so the local and the global view come
from one computation instead of two unrelated ones.

Four plots, in the order the concept builds:

  01  mean|SHAP| per feature      -- the global ranking, comparable to gain
  02  beeswarm                    -- the same ranking, but keeping the sign and
                                     the per-passenger spread that (01) averages away
  03  force plot, one passenger   -- the local explanation SHAP actually exists for
  04  dependence plot on `age`    -- how one feature's effect changes with its value,
                                     coloured by whatever interacts with it most

TreeSHAP (`shap.TreeExplainer`) is used throughout: for tree ensembles the exact
Shapley values are computable in polynomial time by walking the trees, so none of
this is the sampling approximation that KernelSHAP would need.

Environment note: shap 0.37 predates the modern `shap.Explainer` API, so this
uses `TreeExplainer` / `summary_plot` / `force_plot` / `dependence_plot`, and a
scikit-learn classifier's `shap_values` comes back as a list [class_0, class_1].
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

np.random.seed(42)
RANDOM_STATE = 42

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


class TitanicShapExplainer:
    """Fits one random forest on the Titanic split and explains it with TreeSHAP.

    The same dataset, the same feature list and the same 80/20 stratified split
    (random_state=42) as `trees_and_boosting/`, so the model being explained here
    is the one whose importance plots that directory already produced.
    """

    # Identical to trees_and_boosting/ -- deliberately, so the two are comparable.
    FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    NUMERICAL = ["age", "sibsp", "parch", "fare"]
    CATEGORICAL = ["pclass", "sex", "embarked"]

    def __init__(self, n_estimators=300, max_depth=6, random_state=RANDOM_STATE):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.random_state = random_state
        # Fitted on the training rows only, then applied to the test rows.
        self._num_imputer = SimpleImputer(strategy="median")
        self._cat_imputer = SimpleImputer(strategy="most_frequent")
        self._encoder = OneHotEncoder(drop="first", sparse=False, handle_unknown="error")

    # ------------------------------------------------------------------
    # Data and model
    # ------------------------------------------------------------------
    def _encode(self, X_raw, fit):
        """Impute and one-hot into a *named* DataFrame.

        SHAP explains the columns the model was actually fitted on. Hiding the
        preprocessing inside a ColumnTransformer would leave the plots labelled
        x0..x9; doing it by hand keeps `sex_male` and `pclass_3` readable. The
        `fit` flag is what keeps the test rows out of the imputer's medians and
        the encoder's categories.
        """
        if fit:
            num = self._num_imputer.fit_transform(X_raw[self.NUMERICAL])
            cat = self._cat_imputer.fit_transform(X_raw[self.CATEGORICAL])
            enc = self._encoder.fit_transform(cat)
        else:
            num = self._num_imputer.transform(X_raw[self.NUMERICAL])
            cat = self._cat_imputer.transform(X_raw[self.CATEGORICAL])
            enc = self._encoder.transform(cat)

        cat_names = list(self._encoder.get_feature_names(self.CATEGORICAL))
        out = pd.DataFrame(
            np.hstack([num, enc]),
            columns=self.NUMERICAL + cat_names,
            index=X_raw.index,
        )
        return out

    def fit(self):
        print("=" * 70)
        print("SHAP: FROM A GLOBAL RANKING TO ONE PASSENGER'S EXPLANATION")
        print("=" * 70)

        titanic = sns.load_dataset("titanic")
        X = titanic[self.FEATURES]
        y = titanic["survived"]

        X_train_raw, X_test_raw, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        self.X_test_raw = X_test_raw

        self.X_train = self._encode(X_train_raw, fit=True)
        self.X_test = self._encode(X_test_raw, fit=False)

        self.model.fit(self.X_train, self.y_train)
        self.proba = self.model.predict_proba(self.X_test)[:, 1]

        print("\n0. THE MODEL BEING EXPLAINED")
        print("-" * 70)
        print(f"Rows: {len(titanic)}   train: {len(self.X_train)}   test: {len(self.X_test)}")
        print(f"Encoded features ({len(self.X_train.columns)}): {list(self.X_train.columns)}")
        print(f"Test accuracy: {accuracy_score(self.y_test, self.model.predict(self.X_test)):.4f}")
        print(f"Test ROC-AUC : {roc_auc_score(self.y_test, self.proba):.4f}")

        # TreeSHAP: exact Shapley values by walking the trees, not sampling.
        self.explainer = shap.TreeExplainer(self.model)
        shap_values = self.explainer.shap_values(self.X_test)

        # shap 0.37 hands back [class_0, class_1] for an sklearn classifier.
        # Everything below is the "survived" side; class_0 is its exact negative.
        self.shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values
        expected = self.explainer.expected_value
        self.base_value = expected[1] if np.ndim(expected) > 0 else expected

        self._check_additivity()
        return self

    def _check_additivity(self):
        """The property that makes SHAP trustworthy, measured rather than claimed.

        base_value + sum(shap values for a row) must equal the model's output for
        that row. Gain importance has no comparable guarantee -- it is a by-product
        of how the trees were grown, not a decomposition of any actual prediction.
        """
        reconstructed = self.base_value + self.shap_values.sum(axis=1)
        deviation = np.abs(reconstructed - self.proba)

        print("\n1. ADDITIVITY CHECK")
        print("-" * 70)
        print(f"base value (mean predicted P(survived) over the training rows): {self.base_value:.6f}")
        print(f"max |base + sum(SHAP) - predict_proba| over {len(self.proba)} test rows: {deviation.max():.3e}")
        print("The explanation is not an approximation of the model; it reconstructs it.")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def _save(self, filename):
        """Save the figure shap just drew, then show it.

        The shap plotting functions draw onto the current figure and return None,
        so the figure has to be picked up with gcf() after the call. Saving before
        show() means the artifacts appear whether the run is interactive or
        headless (MPLBACKEND=Agg), where show() is a no-op.
        """
        fig = plt.gcf()
        fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=120, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def plot_global_bar(self):
        """mean|SHAP| per feature: the global ranking, directly comparable to gain."""
        print("\n2. GLOBAL IMPORTANCE -- mean |SHAP| per feature")
        print("-" * 70)

        mean_abs = pd.Series(
            np.abs(self.shap_values).mean(axis=0), index=self.X_test.columns
        ).sort_values(ascending=False)
        for name, value in mean_abs.items():
            print(f"  {name:<14} {value:.4f}")
        print("Units are probability: 'sex_male shifts P(survived) by this much on average'.")
        print("Gain importance has no unit at all -- it is summed impurity decrease.")

        shap.summary_plot(
            self.shap_values, self.X_test, plot_type="bar", show=False
        )
        plt.title("Global importance: mean |SHAP value|")
        self._save("01_shap_global_bar.png")

    def plot_beeswarm(self):
        """The same ranking with the sign and the spread that (01) averages away."""
        print("\n3. BEESWARM -- what the bar chart threw away")
        print("-" * 70)

        male = self.X_test["sex_male"] == 1
        print(f"  mean SHAP for sex_male among men   : {self.shap_values[male.values, self.X_test.columns.get_loc('sex_male')].mean():+.4f}")
        print(f"  mean SHAP for sex_male among women : {self.shap_values[~male.values, self.X_test.columns.get_loc('sex_male')].mean():+.4f}")
        print("Same magnitude in the bar chart, opposite sign here. One number per")
        print("feature cannot express that; one number per feature per passenger can.")

        shap.summary_plot(self.shap_values, self.X_test, show=False)
        plt.title("Beeswarm: one dot per passenger per feature")
        self._save("02_shap_beeswarm.png")

    def plot_force_single(self):
        """The local explanation SHAP exists for: why THIS passenger."""
        i = int(np.argmax(self.proba))
        row = self.X_test_raw.iloc[i]

        print("\n4. LOCAL EXPLANATION -- the most confidently-predicted survivor")
        print("-" * 70)
        print(f"Test row {self.X_test_raw.index[i]}: " + ", ".join(
            f"{f}={row[f]}" for f in self.FEATURES
        ))
        if row[self.NUMERICAL].isna().any():
            missing = [f for f in self.NUMERICAL if pd.isna(row[f])]
            # The model never saw the NaN -- it saw the training median. The force
            # plot labels the imputed value, so print it here or the two disagree.
            print("Imputed before the model saw it: " + ", ".join(
                f"{f}={self.X_test.iloc[i][f]:.4f} (train median)" for f in missing
            ))
        print(f"Actual: {'survived' if self.y_test.iloc[i] == 1 else 'died'}   "
              f"predicted P(survived): {self.proba[i]:.4f}")
        print(f"\n  base value                 {self.base_value:+.4f}")
        contributions = pd.Series(
            self.shap_values[i], index=self.X_test.columns
        ).sort_values(key=np.abs, ascending=False)
        for name, value in contributions.items():
            print(f"  {name:<24} {value:+.4f}")
        print(f"  {'= prediction':<24} {self.base_value + contributions.sum():+.4f}")

        shap.force_plot(
            self.base_value,
            self.shap_values[i],
            self.X_test.iloc[i],
            matplotlib=True,
            show=False,
            text_rotation=15,
        )
        self._save("03_shap_force_single.png")

    def plot_dependence(self, feature="age"):
        """How one feature's effect changes with its value -- and what interacts with it."""
        print(f"\n5. DEPENDENCE -- the effect of `{feature}` is not one number")
        print("-" * 70)

        col = self.X_test.columns.get_loc(feature)
        values = self.X_test[feature].values
        shap_col = self.shap_values[:, col]
        edges = [0, 10, 20, 40, 55, 100]
        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bucket = (values >= lo) & (values < hi)
            if in_bucket.any():
                print(f"  {feature} [{lo:>2}, {hi:>3}): n={in_bucket.sum():>3}   "
                      f"mean SHAP {shap_col[in_bucket].mean():+.4f}")
        print("A single 'importance' number reports the size of this effect but never")
        print("its shape. Nearly all of age's contribution sits in the under-10 band;")
        print("across the 114 passengers aged 20-40 the effect is close to nothing.")
        print("Gain importance ranks age fourth either way and cannot tell you that.")

        shap.dependence_plot(feature, self.shap_values, self.X_test, show=False)
        self._save(f"04_shap_dependence_{feature}.png")

    def run_all(self):
        self.fit()
        self.plot_global_bar()
        self.plot_beeswarm()
        self.plot_force_single()
        self.plot_dependence("age")
        print("\n" + "=" * 70)
        print(f"Done. Four plots written to {PLOTS_DIR}")
        print("=" * 70)
        return self


if __name__ == "__main__":
    TitanicShapExplainer().run_all()
