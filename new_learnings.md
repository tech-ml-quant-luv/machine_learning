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
resulting distribution has unit variance. 