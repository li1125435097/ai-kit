import numpy as np
from sklearn.base import ClassNamePrefixFeaturesOutMixin, BaseEstimator
class MyEstimator(ClassNamePrefixFeaturesOutMixin, BaseEstimator):
    def fit(self, X, y=None):
        self._n_features_out = X.shape[1]
        return self
X = np.array([[1, 2], [3, 4]])
out = MyEstimator().fit(X).get_feature_names_out()
print(out)  # ['myestimator__feature_0', 'myestimator__feature_1']
print(X.shape[1])