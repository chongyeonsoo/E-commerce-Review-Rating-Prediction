import numpy as np
from tqdm import tqdm
class Logisticregression:
    def __init__(self, lr, max_iter, verbose = False, lam = 0.01, tol = 1e-6):
        self.max_iter = max_iter
        self.verbose = verbose
        self.lr = lr
        self.tol = tol
        self.lam = lam
    def sigmoid(self, X):
        return 1/(1 + np.exp(-(X)))
    def loss_cal(self, X, y):
        m = X.shape[0]
        y_hat = self.sigmoid(X@self.theta)
        loss = -1/m*np.sum(y*np.log(y_hat) + (1 - y)*np.log(1 - y_hat))
        return loss
    def fit(self, X, y):
        m,n = X.shape
        self.theta = np.zeros((n,1))
        loss_his = []
        y = np.array(y).reshape(-1, 1)
        bar = tqdm(range(self.max_iter))
        for i in bar:
            y_hat = self.sigmoid(X@self.theta)
            loss = self.loss_cal(X, y)
            loss_his.append(loss)
            gradient = (1/m)*(X.T@(y_hat - y)) + (self.lam/m)*self.theta
            self.theta = self.theta - self.lr*gradient
            bar.set_description(
                f"Iter {i} | Loss: {loss:.6f}"
            )
            if i > 0 and (abs(loss - loss_his[i - 1]) < self.tol):
                print("EARLY STOPPING")
                break
    def predict(self, X):
        y_hat = self.sigmoid(X@self.theta)
        return (y_hat >= 0.5).astype('int')
    def predict_proba(self, X):
        y_hat = self.sigmoid(X@self.theta)
        return y_hat