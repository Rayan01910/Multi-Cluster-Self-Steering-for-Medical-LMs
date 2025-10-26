import numpy as np
from sklearn.linear_model import LogisticRegression

class ConfidenceCalibrator:
    def __init__(self):
        # liblinear handles small dense feature sets well
        self.lr = LogisticRegression(max_iter=1000, solver="liblinear")

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.lr.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # return probability of class 1 (correct)
        return self.lr.predict_proba(X)[:, 1]
