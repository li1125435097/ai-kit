import numpy as np
from sklearn.base import BaseEstimator,BiclusterMixin
# 定义一个自定义的估计器类，继承自BaseEstimator
class MyEstimator(BaseEstimator):
    # 初始化函数，接受一个参数param
    def __init__(self, *, param=1):
        self.param = param
    # 训练函数，接受两个参数X和y，X为特征矩阵，y为标签向量
    def fit(self, X, y=None):
        # 设置is_fitted_属性为True，表示已经训练过
        self.is_fitted_ = True
        return self
    # 预测函数，接受一个参数X，X为特征矩阵
    def predict(self, X):
        # 返回一个全为param的数组，数组长度为X的行数
        return np.full(shape=X.shape[0], fill_value=self.param)
# 创建一个MyEstimator对象，参数为2
estimator = MyEstimator(param=2)
# 获取估计器的参数
estimator.get_params()
# 创建一个特征矩阵X和标签向量y
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([1, 0, 1])
# 打印训练后的预测结果
print(estimator.fit(X, y).predict(X))
# 修改估计器的参数为3，并重新训练和预测
print(estimator.set_params(param=3).fit(X, y).predict(X))


class DummyBiClustering(BiclusterMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.rows_ = np.ones(shape=(1, X.shape[0]), dtype=bool)
        self.columns_ = np.ones(shape=(1, X.shape[1]), dtype=bool)
        return self
X = np.array([[1, 1], [2, 1], [1, 0],
              [4, 7], [3, 5], [3, 6]])
bicluster = DummyBiClustering().fit(X)
print(bicluster.get_indices(0))