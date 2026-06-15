##### Intuition - 
Suppose you pose a complex question to thousands of random people, then aggregate
their answers. In many cases you will find that this aggregated answer is better than
an expert’s answer. This is called the wisdom of the crowd. Similarly, if you aggregate
the predictions of a group of predictors (such as classifiers or regressors), you will
often get better predictions than with the best individual predictor. A group of pre‐
dictors is called an ensemble; thus, this technique is called Ensemble Learning, and an
Ensemble Learning algorithm is called an Ensemble method.

##### Random Forest - 

As an example of an Ensemble method, you can train a group of Decision Tree classi‐
fiers, each on a different random subset of the training set. To make predictions, you
obtain the predictions of all the individual trees, then predict the class that gets the
most votes (see the last exercise in Chapter 6). Such an ensemble of Decision Trees is
called a Random Forest, and despite its simplicity, this is one of the most powerful
Machine Learning algorithms available today.

##### Voting Classifiers -

Suppose you have trained a few classifiers, each one achieving about 80% accuracy.
You may have a Logistic Regression classifier, an SVM classifier, a Random Forest
classifier, a K-Nearest Neighbors classifier, and perhaps a few more.
A very simple way to create an even better classifier is to aggregate the predictions of
each classifier and predict the class that gets the most votes. This majority-vote classi‐
fier is called a hard voting classifier

##### Classifier Diversification - 

suppose you build an ensemble containing 1,000 classifiers that are individ‐
ually correct only 51% of the time (barely better than random guessing). If you pre‐
dict the majority voted class, you can hope for up to 75% accuracy! However, this is
only true if all classifiers are perfectly independent, making uncorrelated errors,
which is clearly not the case because they are trained on the same data. They are likely
to make the same types of errors, so there will be many majority votes for the wrong
class, reducing the ensemble’s accuracy.
Ensemble methods work best when the predictors are as independ‐
ent from one another as possible. One way to get diverse classifiers
is to train them using very different algorithms. This increases the
chance that they will make very different types of errors, improving
the ensemble’s accuracy.

- If all classifiers are able to estimate class probabilities (i.e., they all have a pre
dict_proba() method), then you can tell Scikit-Learn to predict the class with the
highest class probability, averaged over all the individual classifiers. This is called soft
voting. It often achieves higher performance than hard voting because it gives more
weight to highly confident votes. All you need to do is replace voting="hard" with
voting="soft" and ensure that all classifiers can estimate class probabilities. This is
not the case for the SVC class by default, so you need to set its probability hyper‐
parameter to True (this will make the SVC class use cross-validation to estimate class
probabilities, slowing down training, and it will add a predict_proba() method). If
you modify the preceding code to use soft voting, you will find that the voting classi‐
fier achieves over 91.2% accuracy!

