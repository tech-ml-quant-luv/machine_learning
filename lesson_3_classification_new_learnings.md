In this chapter we will be using the MNIST dataset, which is a set of 70,000 small
images of digits handwritten by high school students and employees of the US Cen‐
sus Bureau. Each image is labeled with the digit it represents. This set has been stud‐
ied so much that it is often called the “hello world” of Machine Learning: whenever
people come up with a new classification algorithm they are curious to see how it will
perform on MNIST, and anyone who learns Machine Learning tackles this dataset
sooner or later

Datasets loaded by Scikit-Learn generally have a similar dictionary structure, includ‐
ing the following:
• A DESCR key describing the dataset
• A data key containing an array with one row per instance and one column per
feature
• A target key containing an array with the labels

The SGDClassifier relies on randomness during training (hence
the name “stochastic”). If you want reproducible results, you
should set the random_state parameter.

The custom crossvalidation function - The StratifiedKFold class performs stratified sampling (as explained in Chapter 2)
to produce folds that contain a representative ratio of each class. At each iteration the
code creates a clone of the classifier, trains that clone on the training folds, and makes
predictions on the test fold. Then it counts the number of correct predictions and
outputs the ratio of correct predictions

!Important: When the accuracy is really high, do check the data imbalance, leakage
Accuracy is generally not the preferred performance measure
for classifiers, especially when you are dealing with skewed datasets (i.e., when some
classes are much more frequent than others)

A much better way to evaluate the performance of a classifier is to look at the confu‐
sion matrix. The general idea is to count the number of times instances of class A are
classified as class B. For example, to know the number of times the classifier confused
images of 5s with 3s, you would look in the fifth row and third column of the confu‐
sion matrix.

Just like the cross_val_score() function, cross_val_predict() performs K-fold
cross-validation, but instead of returning the evaluation scores, it returns the predic‐
tions made on each test fold. This means that you get a clean prediction for each
instance in the training set (“clean” meaning that the prediction is made by a model
that never saw the data during training).


Q&A Summary

Does decision_function give accuracy scores?
No, it gives a raw confidence score (signed distance from the decision boundary).

What is the meaning of threshold here?
The cutoff value — if score >= threshold, predict positive; else predict negative. Default is 0.

Does it give a confidence interval?
No, it gives a single confidence score — a number, not a statistical range.

When do we adjust the threshold?
When accuracy alone isn't enough — e.g., cancer detection (catch all positives) or spam filter (avoid false alarms).

Are accuracy threshold and decision function threshold the same?
Yes, same threshold — changing it affects all metrics (accuracy, precision, recall), not just one.

Does accuracy threshold lie between 0 and 1?
Only when using predict_proba() (probabilities); decision_function scores have no fixed range.

Does SGDClassifier have probabilities?
Not by default; use loss="log_loss" or wrap with CalibratedClassifierCV to get 0–1 probabilities.

Why doesn't SGDClassifier give probabilities by default?
Its default hinge loss (SVM-style) only produces a decision score mathematically — it doesn't model probabilities, so extra calibration is needed to convert to 0–1.

!important If someone says, “Let’s reach 99% precision,” you should ask, “At
what recall?”