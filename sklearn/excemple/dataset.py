import sklearn.datasets as datasets

# 加载鸢尾花数据集
X, y = datasets.load_iris(return_X_y=True);print(X.shape, y.shape)
print(X[:1], y[:10])
# 加载糖尿病数据集
X, y = datasets.load_diabetes(return_X_y=True);print(X.shape, y.shape)
print(X[:1], y[:10])
# 加载手写数字数据集
X, y = datasets.load_digits(return_X_y=True);print(X.shape, y.shape)
print(X[:1], y[:10])
# 加载体能训练数据集
X, y = datasets.load_linnerud(return_X_y=True);print(X.shape, y.shape)
# print(X[:1], y[:10])
# 加载葡萄酒数据集
X, y = datasets.load_wine(return_X_y=True);print(X.shape, y.shape)
# print(X[:1], y[:10])
# 加载乳腺癌数据集
X, y = datasets.load_breast_cancer(return_X_y=True);print(X.shape, y.shape)
# print(X[:1], y[:10])
