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


Here's the cleaned-up version:

---

## Cross Validation

- The output of `cross_val_score` is an array containing the N evaluation scores (one per fold).

---

## Dealing with Overfitting

Possible solutions for overfitting:
- Simplify the model
- Constrain it (i.e., regularize it)
- Get a lot more training data

Before diving deeper into Random Forests, try out many other models from various ML categories (e.g., several SVMs with different kernels, possibly a neural network) — without spending too much time tweaking hyperparameters. The goal is to **shortlist 2–5 promising models**.

---

## Hyperparameter Search Tips

- When you have no idea what value a hyperparameter should have, a simple approach is to try consecutive powers of 10 (or a smaller number for a more fine-grained search, e.g., `n_estimators`).
- If `GridSearchCV` is initialized with `refit=True` (the default), once it finds the best estimator via cross-validation, it retrains it on the whole training set — usually a good idea since more data tends to improve performance.
- Data preparation steps can be treated as hyperparameters. For example, grid search can automatically determine whether to add a feature you were unsure about (e.g., `add_bedrooms_per_room` in `CombinedAttributesAdder`). It can similarly find the best way to handle outliers, missing features, and feature selection.

---

## Model Analysis & Error Inspection

- Look at feature importances — if only one category of a feature is useful (e.g., one `ocean_proximity` category), consider dropping the others.
- Examine specific errors the system makes: understand why they occur and what could fix them (adding features, removing uninformative ones, cleaning outliers, etc.).

---

## Evaluating Generalization Error

- A point estimate of generalization error may not be enough to justify launching. If it's only 0.1% better than the current production model, compute a **95% confidence interval** using `scipy.stats.t.interval()`.
- If you did a lot of hyperparameter tuning, performance on the test set will often be slightly worse than on the validation set — the model has been fine-tuned to the validation data and may not generalize as well.

> **⚠️ Important**: Resist the temptation to tweak hyperparameters to make test set numbers look good. Those improvements are unlikely to generalize to new data.

---

## Model Deployment

**Option 1 — Web Service (REST API)**
- Wrap the model in a dedicated web service your application queries via REST API.
- Load the model at server startup, not on every request.
- Benefits: easier version upgrades without interrupting the main app, simple horizontal scaling (load-balance across multiple web service instances), and your web app can be written in any language.

**Option 2 — Cloud Deployment (e.g., Google Cloud AI Platform)**
- Save model with `joblib`, upload to Google Cloud Storage, create a new model version on AI Platform pointing to the GCS file.
- Gives you a managed web service with automatic load balancing and scaling.
- Takes JSON input (e.g., district data) and returns JSON predictions.

---

## Monitoring in Production

- Write monitoring code to check live performance at regular intervals and trigger alerts on drops.
- Watch for both **steep drops** (likely broken infrastructure) and **gentle decay** (model rot — the world changes, last year's data may no longer reflect today's reality).

**Inferring performance from downstream metrics** (when possible):
- Example: in a recommender system, monitor the number of recommended products sold per day. A drop implicates the model.

**Human raters** (when automated metrics aren't enough):
- Send a sample of model outputs (especially low-confidence ones) to human raters for evaluation.
- Raters can be domain experts, crowdsourcing workers (e.g., Amazon Mechanical Turk), or even end users via surveys or repurposed captchas.

---

## Automating the ML Pipeline

Automate as much as possible:

- Collect and label fresh data regularly.
- Script model training and hyperparameter tuning to run automatically (e.g., daily or weekly).
- Script evaluation: compare the new model against the previous one on the updated test set; deploy only if performance has not decreased. Investigate if it has.

---

## Input Data Quality Monitoring

Monitor model inputs, not just outputs. Triggers for alerts:
- More inputs missing a feature than usual
- Mean or standard deviation of a feature drifts too far from the training set distribution
- A categorical feature starts containing new, unseen categories

---

## Backups & Rollback

- Keep backups of every model version — enables quick rollback and easy comparison of new vs. old models.
- Keep backups of every dataset version — allows rollback if fresh data turns out to be corrupted or full of outliers, and lets you evaluate any model against any historical dataset.

---

## Test Set Segmentation

Consider creating multiple subsets of the test set to evaluate performance on specific slices:
- Most recent data only
- Specific input types (e.g., inland districts vs. coastal districts)

This gives deeper insight into model strengths and weaknesses.

---

## Key Takeaway

ML involves a lot of infrastructure. The first project takes significant time and effort — but once the infrastructure is in place, going from idea to production becomes much faster.

> It is preferable to be comfortable with the overall process and know 3–4 algorithms well, rather than spending all your time exploring advanced algorithms.

---
