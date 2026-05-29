import numpy as np
from tqdm import tqdm
class MultinomialNaiveBayes:
    def __init__(self, alpha):
        self.alpha = alpha
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.pro_class = {}
        self.sample_pro = {}
        y = np.array(y)
        for c in self.classes:
            X_c = X[y == c]
            self.pro_class[c] = X_c.shape[0]/X.shape[0]
            class_count = np.array(X_c.sum(axis = 0)).flatten()
            class_count += self.alpha
            total = class_count.sum()
            self.sample_pro[c] = class_count/total
        return self
    def predict(self, X):
        result = []
        for i in tqdm(range(X.shape[0]), desc="Training"):
            bestscore = -np.inf
            best_class = None
            for c in self.classes:
                score = np.log(self.pro_class[c])
                index = X[i].indices
                data = X[i].data
                score = score +  np.sum(data*np.log(self.sample_pro[c][index]))
                if score > bestscore:
                    bestscore = score
                    best_class = c
            result.append(best_class)      
        return result
    def predict_proba(self, X):
        result = []
        for i in range(X.shape[0]):
            scores = []
            for c in self.classes:
                score = np.log(self.pro_class[c])
                index = X[i].indices
                data = X[i].data
                score = score +  np.sum(data*np.log(self.sample_pro[c][index]))
                scores.append(score)
            scores = np.array(scores)
            scores_exp = np.exp(scores - max(scores))
            proba = (scores_exp/ scores_exp.sum())
            result.append(proba)
        return np.array(result)
        
