import numpy as np
from tqdm import tqdm


class Logisticregression:
    """Binary Logistic Regression implemented from scratch with NumPy.

    Fixes compared with the first version:
    - adds an intercept/bias term without regularizing it;
    - clips sigmoid/log values for numerical stability;
    - optionally supports class_weight="balanced" for imbalanced sentiment labels;
    - keeps the original public API: fit, predict, predict_proba.
    """

    def __init__(
        self,
        lr,
        max_iter,
        verbose=False,
        lam=0.01,
        tol=1e-6,
        fit_intercept=True,
        class_weight=None,
    ):
        self.max_iter = max_iter
        self.verbose = verbose
        self.lr = lr
        self.tol = tol
        self.lam = lam
        self.fit_intercept = fit_intercept
        self.class_weight = class_weight
        self.eps = 1e-15

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _linear_score(self, X):
        score = X @ self.theta
        if self.fit_intercept:
            score = score + self.intercept_
        return score

    def _sample_weight(self, y):
        y = np.asarray(y).reshape(-1)
        if self.class_weight != "balanced":
            return np.ones_like(y, dtype=float).reshape(-1, 1)

        classes, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        n_classes = len(classes)
        weights = {c: n_samples / (n_classes * cnt) for c, cnt in zip(classes, counts)}
        return np.array([weights[val] for val in y], dtype=float).reshape(-1, 1)

    def loss_cal(self, X, y):
        y = np.asarray(y).reshape(-1, 1)
        sample_weight = self._sample_weight(y)
        weight_sum = sample_weight.sum()

        y_hat = np.clip(self.sigmoid(self._linear_score(X)), self.eps, 1 - self.eps)
        data_loss = -np.sum(sample_weight * (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))) / weight_sum
        reg_loss = (self.lam / (2 * X.shape[0])) * float(np.sum(self.theta ** 2))
        return data_loss + reg_loss

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros((n, 1))
        self.intercept_ = 0.0
        loss_his = []
        y = np.asarray(y).reshape(-1, 1)
        sample_weight = self._sample_weight(y)
        weight_sum = sample_weight.sum()

        iterator = range(self.max_iter)
        if self.verbose:
            iterator = tqdm(iterator)

        for i in iterator:
            y_hat = self.sigmoid(self._linear_score(X))
            error = sample_weight * (y_hat - y)

            gradient = (X.T @ error) / weight_sum + (self.lam / m) * self.theta
            self.theta = self.theta - self.lr * gradient

            if self.fit_intercept:
                intercept_gradient = float(error.sum() / weight_sum)
                self.intercept_ = self.intercept_ - self.lr * intercept_gradient

            loss = self.loss_cal(X, y)
            loss_his.append(loss)

            if self.verbose:
                iterator.set_description(f"Iter {i} | Loss: {loss:.6f}")

            if i > 0 and abs(loss - loss_his[i - 1]) < self.tol:
                if self.verbose:
                    print("EARLY STOPPING")
                break
        return self

    def predict(self, X):
        y_hat = self.sigmoid(self._linear_score(X))
        return (y_hat >= 0.5).astype(int)

    def predict_proba(self, X):
        y_hat = self.sigmoid(self._linear_score(X))
        return y_hat
