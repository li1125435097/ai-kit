from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 获取数据集
X,y = datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练
reg = DecisionTreeRegressor(max_depth=5).fit(X_train, y_train)
y_pred = reg.predict(X_test)

# 模型评估
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('score: %.2f' % r2_score(y_test, y_pred))