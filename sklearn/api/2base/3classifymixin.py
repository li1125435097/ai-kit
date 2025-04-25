import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
# Mixin classes should always be on the left-hand side for a correct MRO
class MyEstimator(ClassifierMixin, BaseEstimator):
    def __init__(self, *, param=1):
        self.param = param
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
    def predict(self, X):
        return np.full(shape=X.shape[0], fill_value=self.param)
estimator = MyEstimator(param=1)
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([1, 0, 1])
print(estimator.fit(X, y).predict(X))
print(estimator.score(X, y))