import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
class MyClusterer(ClusterMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.labels_ = np.ones(shape=(len(X),), dtype=np.int64)
        return self
X = [[1, 2], [2, 3], [3, 4]]
print(MyClusterer().fit_predict(X))