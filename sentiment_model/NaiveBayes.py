import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm


class MultinomialNaiveBayes:
    """Multinomial Naive Bayes from scratch.

    The model expects non-negative bag-of-words style features. It can run on
    TF-IDF because TF-IDF values are non-negative, but count features are closer
    to the classic MultinomialNB assumption.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        y = np.asarray(y)
        n_samples = X.shape[0]

        class_log_prior = []
        feature_log_prob = []

        for c in self.classes:
            X_c = X[y == c]
            class_log_prior.append(np.log(X_c.shape[0] / n_samples))

            class_count = np.asarray(X_c.sum(axis=0)).ravel() + self.alpha
            smoothed_total = class_count.sum()
            log_prob = np.log(class_count / smoothed_total)
            feature_log_prob.append(log_prob)

        self.class_log_prior_ = np.asarray(class_log_prior)
        self.feature_log_prob_ = np.vstack(feature_log_prob)
        self.class_log_prior_ = np.asarray(class_log_prior).reshape(1, -1)
        return self

    def _joint_log_likelihood(self, X):
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes[np.asarray(jll.argmax(axis=1)).ravel()]

    def predict_proba(self, X):
        jll = np.asarray(self._joint_log_likelihood(X))
        log_prob_x = logsumexp(jll, axis=1, keepdims=True)
        return np.exp(jll - log_prob_x)


class ComplementNaiveBayes(MultinomialNaiveBayes):
    """Complement Naive Bayes from scratch.

    This variant estimates each class using statistics from all *other* classes,
    which is often more robust for imbalanced text classification.
    """

    def fit(self, X, y):
        self.classes = np.unique(y)
        y = np.asarray(y)
        n_samples = X.shape[0]
        total_count = np.asarray(X.sum(axis=0)).ravel()

        class_log_prior = []
        feature_log_prob = []

        for c in self.classes:
            X_c = X[y == c]
            class_log_prior.append(np.log(X_c.shape[0] / n_samples))

            class_count = np.asarray(X_c.sum(axis=0)).ravel()
            complement_count = total_count - class_count + self.alpha
            complement_total = complement_count.sum()
            # Negative log-complement probabilities; larger score means better class.
            feature_log_prob.append(-np.log(complement_count / complement_total))

        self.class_log_prior_ = np.asarray(class_log_prior)
        self.feature_log_prob_ = np.vstack(feature_log_prob)
        self.class_log_prior_ = np.asarray(class_log_prior).reshape(1, -1)
        return self
