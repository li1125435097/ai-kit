from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np

# 加载糖尿病数据集
X, y = load_diabetes(return_X_y=True)
print(X.shape, y.shape)
print(type(X))

# 只使用一个特征
# X = X[:, [2]]  

# 将数据集分为训练集和测试集，测试集占总数据的20%，不进行洗牌
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, shuffle=False)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.linear_model import LinearRegression

# 创建线性回归模型，并使用训练集进行拟合
regressor = LinearRegression(positive=True).fit(X_train, y_train)
print(regressor.coef_, regressor.intercept_)

from sklearn.metrics import mean_squared_error, r2_score

# 使用测试集进行预测
y_pred = regressor.predict(X_test)
# 输出均方误差和决定系数
print(f"Mean squared error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"Coefficient of determination: {r2_score(y_test, y_pred):.2f}")
print(f"Coefficient of determination2: {np.average((y_test-y_pred)**2):.2f}")