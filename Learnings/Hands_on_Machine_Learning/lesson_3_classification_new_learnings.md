# Lesson 3 — Classification: Key Learnings

---

## 1. MNIST Dataset

In this chapter we will be using the MNIST dataset, which is a set of 70,000 small
images of digits handwritten by high school students and employees of the US Census
Bureau. Each image is labeled with the digit it represents. This set has been studied
so much that it is often called the "hello world" of Machine Learning: whenever
people come up with a new classification algorithm they are curious to see how it will
perform on MNIST, and anyone who learns Machine Learning tackles this dataset
sooner or later.

---

## 2. Scikit-Learn Dataset Structure

Datasets loaded by Scikit-Learn generally have a similar dictionary structure, including the following:

- A `DESCR` key describing the dataset
- A `data` key containing an array with one row per instance and one column per feature
- A `target` key containing an array with the labels

---

## 3. Key Concepts

### 3.1 SGDClassifier & Randomness

The SGDClassifier relies on randomness during training (hence the name "stochastic").
If you want reproducible results, you should set the `random_state` parameter.

---

### 3.2 Cross-Validation

**StratifiedKFold (Custom Cross-Validation):**
The `StratifiedKFold` class performs stratified sampling (as explained in Chapter 2)
to produce folds that contain a representative ratio of each class. At each iteration the
code creates a clone of the classifier, trains that clone on the training folds, and makes
predictions on the test fold. Then it counts the number of correct predictions and
outputs the ratio of correct predictions.

**`cross_val_predict()`:**
Just like the `cross_val_score()` function, `cross_val_predict()` performs K-fold
cross-validation, but instead of returning the evaluation scores, it returns the
predictions made on each test fold. This means that you get a clean prediction for each
instance in the training set ("clean" meaning that the prediction is made by a model
that never saw the data during training).

---

### 3.3 Accuracy & Skewed Datasets

> **! Important:** When the accuracy is really high, do check for data imbalance and leakage.

Accuracy is generally not the preferred performance measure for classifiers, especially
when you are dealing with skewed datasets (i.e., when some classes are much more
frequent than others).

---

### 3.4 Confusion Matrix

A much better way to evaluate the performance of a classifier is to look at the
confusion matrix. The general idea is to count the number of times instances of class A
are classified as class B. For example, to know the number of times the classifier
confused images of 5s with 3s, you would look in the fifth row and third column of the
confusion matrix.

---

### 3.5 Precision–Recall Tradeoff

> **! Important:** If someone says, "Let's reach 99% precision," you should ask, "At what recall?"

**Decision Function & Threshold:**

| Question | Answer |
|---|---|
| Does `decision_function` give accuracy scores? | No, it gives a raw confidence score (signed distance from the decision boundary). |
| What is the meaning of threshold here? | The cutoff value — if score >= threshold, predict positive; else predict negative. Default is 0. |
| Does it give a confidence interval? | No, it gives a single confidence score — a number, not a statistical range. |
| When do we adjust the threshold? | When accuracy alone isn't enough — e.g., cancer detection (catch all positives) or spam filter (avoid false alarms). |
| Are accuracy threshold and decision function threshold the same? | Yes, same threshold — changing it affects all metrics (accuracy, precision, recall), not just one. |
| Does accuracy threshold lie between 0 and 1? | Only when using `predict_proba()` (probabilities); `decision_function` scores have no fixed range. |

---

### 3.6 SGDClassifier Probabilities

| Question | Answer |
|---|---|
| Does SGDClassifier have probabilities? | Not by default; use `loss="log_loss"` or wrap with `CalibratedClassifierCV` to get 0–1 probabilities. |
| Why doesn't SGDClassifier give probabilities by default? | Its default hinge loss (SVM-style) only produces a decision score mathematically — it doesn't model probabilities, so extra calibration is needed to convert to 0–1. |

---

### 3.7 ROC Curve vs. PR Curve

> **! Important:** ROC curve plots the recall vs. false positive rate at different probability threshold levels.

Since the ROC curve is so similar to the precision/recall (PR) curve, you may wonder
how to decide which one to use. As a rule of thumb:

- **Prefer the PR curve** whenever the positive class is rare, or when you care more about false positives than false negatives.
- **Use the ROC curve** otherwise.

For example, looking at a ROC curve (and the ROC AUC score), you may think that the
classifier is really good. But this is mostly because there are few positives (5s)
compared to the negatives (non-5s). In contrast, the PR curve makes it clear that the
classifier has room for improvement (the curve could be closer to the top-left corner).

The ROC curve is about the tradeoff between TPR (recall) and FPR — not recall values alone.
FPR is a recall-like metric but for the negative class specifically: *"Of all actual negatives, what fraction did we incorrectly call positive?"*

**The way to think about it:**

| Curve | Tradeoff | Measures |
|---|---|---|
| **ROC** | How well you catch positives (TPR) vs. how often you falsely alarm on negatives (FPR) | Model's ability to discriminate between classes |
| **PR** | How precise you are when you fire (Precision) vs. how many true positives you actually catch (Recall) | Usefulness of positive predictions |

**Key structural difference:** FPR involves true negatives (TN) in the denominator, while Precision ignores TN entirely. That's why PR curves are preferred when the negative class is huge — ROC gets "credit" for the massive TN pool, which inflates how good the model looks.


---

### 3.8 AUC (Area Under the Curve)

AUC is just the area under whichever curve we're talking about — it's not specific to ROC.

| Metric | Range | Interpretation |
|---|---|---|
| **ROC-AUC** | 0.5 (random) → 1.0 (perfect) | Probability that the model ranks a random positive instance higher than a random negative one |
| **PR-AUC** | Baseline positive rate (random) → 1.0 (perfect) | Generally a harder, more informative metric on imbalanced data |


---

### 3.9 Multiclass Classification

Whereas binary classifiers distinguish between two classes, multiclass classifiers (also
called multinomial classifiers) can distinguish between more than two classes.

- **Natively multiclass:** SGD classifiers, Random Forest classifiers, Naive Bayes
- **Strictly binary (need a strategy):** Logistic Regression, Support Vector Machine classifiers

**OvR — One-vs-the-Rest (also called One-vs-All):**
Train one binary classifier per class. To classify an image, get the decision score from
each classifier and select the class with the highest score. For 10 digits → 10 classifiers.

**OvO — One-vs-One:**
Train a binary classifier for every pair of classes. For N classes, you need N × (N – 1) / 2
classifiers. For MNIST (10 classes) → 45 classifiers. To classify, run through all classifiers
and pick the class that wins the most duels. The main advantage: each classifier is only
trained on the two classes it distinguishes, so the training sets are small.

**When to use which:**

| Strategy | Prefer when |
|---|---|
| **OvO** | Algorithm scales poorly with training set size (e.g., SVMs) — many small sets is faster |
| **OvR** | Most other binary classification algorithms — fewer classifiers, simpler |

> *(Book reference: Page 100, Chapter 3)*

Scikit-Learn detects when you try to use a binary classification algorithm for a multiclass
classification task, and it automatically runs OvR or OvO, depending on the algorithm.

**Example — SVC for multiclass:**

```python
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
svm_clf.predict([some_digit])
```

This trains the SVC on the original target classes 0–9 (`y_train`), instead of the
5-versus-the-rest target classes (`y_train_5`). Under the hood, Scikit-Learn used the
**OvO strategy**: it trained 45 binary classifiers, got their decision scores for the image,
and selected the class that won the most duels.

If you call `decision_function()`, it returns **10 scores per instance** (one per class)
instead of just 1:


> **! Important:** When a classifier is trained, it stores the list of target classes in its
> `classes_` attribute, ordered by value. The index of each class in the `classes_` array
> conveniently matches the class itself (e.g., the class at index 5 happens to be class 5),
> but in general you won’t be so lucky.

SGD classifiers can directly classify instances into multiple classes. The
`decision_function()` method now returns one value per class.

> **! Important:** Rate of change is often a very good feature — either compared to itself over time, or compared to other metrics.

---

### 3.10 Error Analysis — Confusion Matrix

Analyzing the confusion matrix often gives you insights into ways to improve your
classifier. For example, if efforts should be spent on reducing false 8s, you could:

- Gather more training data for digits that look like 8s (but are not), so the classifier learns to distinguish them from real 8s
- Engineer new features — e.g., count the number of closed loops (8 has two, 6 has one, 5 has none)
- Preprocess the images (e.g., using Scikit-Image, Pillow, or OpenCV) to make patterns such as closed loops stand out more

Analyzing individual errors can also be a good way to gain insights on what your
classifier is doing and why it is failing, but it is more difficult and time-consuming.
For example, plotting examples of 3s and 5s (using `plot_digits()`, which wraps
Matplotlib’s `imshow()`).


---

### 3.11 Multilabel Classification

Until now each instance has always been assigned to just one class. In some cases you
may want your classifier to output multiple classes for each instance. Consider a
face-recognition classifier: if it recognizes several people in the same picture, it should
attach one tag per person. Such a system that outputs multiple binary tags is called a
**multilabel classification system**.

Example: a classifier trained on Alice, Bob, and Charlie shown a picture of Alice and
Charlie should output `[1, 0, 1]` (Alice yes, Bob no, Charlie yes).

**Example — KNeighborsClassifier with multilabel targets:**

```python
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
```

This creates a `y_multilabel` array with two target labels per digit image:
- First label: whether the digit is large (7, 8, or 9)
- Second label: whether the digit is odd

`KNeighborsClassifier` supports multilabel classification (not all classifiers do).
Prediction outputs two labels:

```python
>>> knn_clf.predict([some_digit])
array([[False, True]])
# Digit 5: not large (False), odd (True) ✓
```

**Evaluating a multilabel classifier:**

One approach is to compute the F1 score per label, then average them:

```python
>>> y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
>>> f1_score(y_multilabel, y_train_knn_pred, average=”macro”)
0.976410265560605
```

`average=”macro”` assumes all labels are equally important. If some labels have more
instances (e.g., more pictures of Alice than Bob), give each label a weight equal to its
**support** (number of instances with that label) by using `average=”weighted”` instead.

> *(Book reference: Page 106, Chapter 3 — Scikit-Learn offers additional averaging options and multilabel metrics; see the documentation for more details.)*


---

### 3.12 Multioutput Classification

The last type of classification task is **multioutput–multiclass classification** (or simply
multioutput classification). It is a generalization of multilabel classification where each
label can be multiclass (i.e., it can have more than two possible values).

**Example — image denoiser:**
Takes a noisy digit image as input and outputs a clean digit image as an array of pixel
intensities. The output is multilabel (one label per pixel) and each label is multiclass
(pixel intensity ranges 0–255). This makes it a multioutput classification system.

> The line between classification and regression is sometimes blurry. Predicting pixel
> intensity is arguably more akin to regression than classification. Multioutput systems
> are also not limited to classification — a system can output both class labels and
> value labels per instance.

**Creating noisy training/test sets:**

```python
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise

noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise

y_train_mod = X_train   # targets are the original clean images
y_test_mod = X_test
```

**Training the classifier and cleaning an image:**

```python
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)
```

> *(Book reference: Page 107, Chapter 3 — You can use `shift()` from
> `scipy.ndimage.interpolation` to shift images, e.g. `shift(image, [2, 1], cval=0)`
> shifts two pixels down and one pixel to the right.)*

---

## 4. Chapter Summary

After completing this chapter, you should know how to:

- Select good metrics for classification tasks
- Pick the appropriate precision/recall tradeoff
- Compare classifiers using ROC and PR curves
- Build classification systems for binary, multiclass, multilabel, and multioutput tasks