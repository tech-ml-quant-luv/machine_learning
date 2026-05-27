Here's your cleaned-up, properly formatted version:

---

## Feature Engineering

- **Semi-supervised Learning / Cluster-based Feature Engineering**: Clustering algorithm outputs can be used as inputs to supervised learning models.
- **Stratified Sampling**: Use stratified shuffling from sklearn to preserve the ratio of any target category in the original dataset.
- **EDA — Data Artifacts / Price Caps**: Plots may reveal horizontal lines at values like $450,000, $350,000, and $280,000 — these are data collection quirks. Consider removing the corresponding districts to prevent the model from learning to reproduce them.
- **Log Transform for Skewed Distributions**: Attributes with tail-heavy distributions should be transformed (e.g., by computing their logarithm).
- **Derived / Ratio Features**: Raw counts are less meaningful than ratios. Useful combinations include:
  - Rooms per household
  - Bedrooms per room
  - Population per household
- **Feature-Target Correlation via Domain-driven Engineering**: Derived features built with domain knowledge and sound mathematics are likely to correlate better with the target variable.

---

## Transformation Library

- Gradually build a library of reusable transformation functions for future projects.
- Apply these functions in your live system to transform new data before feeding it to algorithms.
- This makes it easy to systematically try various transformation combinations and find what works best.

---

## Handling Missing Data

Three options:

1. Remove the corresponding districts entirely.
2. Drop the attribute.
3. Impute with a value (zero, mean, median, etc.).

> **⚠️ Important — Data Leakage**: If you choose option 3, compute the median **only on the training set** and use it to fill missing values in both the training and test sets. Save this computed median — you'll need it for test set evaluation and when the system goes live.

---

## Scikit-Learn Design Principles

Scikit-Learn's API is built around five core principles:

**1. Consistency** — All objects share a simple, uniform interface across three roles:

- **Estimators**: Any object that estimates parameters from data using `fit()`. Hyperparameters are set via constructor arguments; learned parameters are stored as instance variables.
- **Transformers**: Estimators that also transform data via `transform()`. Also expose `fit_transform()` as a convenience method (sometimes optimized to run faster than calling both separately).
- **Predictors**: Estimators that make predictions via `predict()`. Also expose `score()` to evaluate prediction quality on a test set.

**2. Inspection** — All hyperparameters are accessible as public instance variables (e.g., `imputer.strategy`). Learned parameters use an underscore suffix (e.g., `imputer.statistics_`).

**3. Nonproliferation of classes** — Datasets are represented as NumPy arrays or SciPy sparse matrices, not custom classes. Hyperparameters are plain Python strings or numbers.

**4. Composition** — Building blocks are reused and composed. For example, a `Pipeline` chains an arbitrary sequence of transformers followed by a final estimator.

**5. Sensible defaults** — Most parameters have reasonable defaults, making it easy to spin up a baseline system quickly.

---

## Custom Sklearn Transformer

Subclass `BaseEstimator` and `TransformerMixin`, implementing `fit()` and `transform()`. Gating optional steps behind a hyperparameter (e.g., `add_bedrooms_per_room`) lets you treat data preparation choices as tunable parameters — the more you automate, the more combinations grid search can explore automatically.

```python
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household      = X[:, rooms_ix]      / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
```

---
Min-max scaling (many people call this normalization) is the simplest: values are shif‐
ted and rescaled so that they end up ranging from 0 to 1. We do this by subtracting
the min value and dividing by the max minus the min. Scikit-Learn provides a trans‐
former called MinMaxScaler for this. It has a feature_range hyperparameter that lets
you change the range if, for some reason, you don’t want 0–1.
Standardization is different: first it subtracts the mean value (so standardized values
always have a zero mean), and then it divides by the standard deviation so that the
resulting distribution has unit variance. Unlike min-max scaling, standardization
does not bound values to a specific range, which may be a problem for some algo‐
rithms (e.g., neural networks often expect an input value ranging from 0 to 1). How‐
ever, standardization is much less affected by outliers. For example, suppose a district
had a median income equal to 100 (by mistake). Min-max scaling would then crush
all the other values from 0–15 down to 0–0.15, whereas standardization would not be
much affected. Scikit-Learn provides a transformer called StandardScaler for
standardization.

#### Important! 
As with all the transformations, it is important to fit the scalers to
the training data only, not to the full dataset (including the test set).
Only then can you use them to transform the training set and the
test set (and new data).


Note that the OneHotEncoder returns a sparse matrix, while the num_pipeline returns
a dense matrix. When there is such a mix of sparse and dense matrices, the Colum
nTransformer estimates the density of the final matrix (i.e., the ratio of nonzero
cells), and it returns a sparse matrix if the density is lower than a given threshold (by
default, sparse_threshold=0.3). In this example, it returns a dense matrix. And
that’s it! We have a preprocessing pipeline that takes the full housing data and applies
the appropriate transformations to each column

#### Cross Validation
- The output of a cross_val_score method is an array containing the 10 evaluation scores:

Possible solutions for overfitting are
to simplify the model, constrain it (i.e., regularize it), or get a lot more training data.
Before you dive much deeper into Random Forests, however, you should try out
many other models from various categories of Machine Learning algorithms (e.g.,
several Support Vector Machines with different kernels, and possibly a neural net‐
work), without spending too much time tweaking the hyperparameters. The goal is to
shortlist a few (two to five) promising models.

When you have no idea what value a hyperparameter should have,
a simple approach is to try out consecutive powers of 10 (or a
smaller number if you want a more fine-grained search, as shown
in this example with the n_estimators hyperparameter).

If GridSearchCV is initialized with refit=True (which is the
default), then once it finds the best estimator using crossvalidation, it retrains it on the whole training set. This is usually a
good idea, since feeding it more data will likely improve its
performance.

Don’t forget that you can treat some of the data preparation steps as
hyperparameters. For example, the grid search will automatically
find out whether or not to add a feature you were not sure about
(e.g., using the add_bedrooms_per_room hyperparameter of your
CombinedAttributesAdder transformer). It may similarly be used
to automatically find the best way to handle outliers, missing fea‐
tures, feature selection, and more.

With this information, you may want to try dropping some of the less useful features
(e.g., apparently only one ocean_proximity category is really useful, so you could try
dropping the others).

You should also look at the specific errors that your system makes, then try to under‐
stand why it makes them and what could fix the problem (adding extra features or
getting rid of uninformative ones, cleaning up outliers, etc.).

In some cases, such a point estimate of the generalization error will not be quite
enough to convince you to launch: what if it is just 0.1% better than the model cur‐
rently in production? You might want to have an idea of how precise this estimate is.
For this, you can compute a 95% confidence interval for the generalization error using
scipy.stats.t.interval()

If you did a lot of hyperparameter tuning, the performance will usually be slightly
worse than what you measured using cross-validation (because your system ends up
fine-tuned to perform well on the validation data and will likely not perform as well
on unknown datasets). It is not the case in this example, but when this happens you
must resist the temptation to tweak the hyperparameters to make the numbers look
good on the test set; the improvements would be unlikely to generalize to new data.

