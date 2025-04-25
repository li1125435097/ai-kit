from sklearn.base import DensityMixin
class MyEstimator(DensityMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
estimator = MyEstimator()
print(hasattr(estimator, "score"))



from sklearn.base import MetaEstimatorMixin
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
class MyEstimator(MetaEstimatorMixin):
    def __init__(self, *, estimator=None):
        self.estimator = estimator
    def fit(self, X, y=None):
        if self.estimator is None:
            self.estimator_ = LogisticRegression()
        else:
            self.estimator_ = self.estimator
        return self
X, y = load_iris(return_X_y=True)
estimator = MyEstimator().fit(X, y)
print(estimator.estimator_)



import numpy as np
from sklearn.base import OneToOneFeatureMixin, BaseEstimator
class MyEstimator(OneToOneFeatureMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self
X = np.array([[1, 2], [3, 4]])
print(MyEstimator().fit(X).get_feature_names_out())