from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# 构造数据
# ['gender', 'age', 'height', 'MonthlyConsumption','weight']
data = np.array([
    [0, 20, 150, 1500, 80], 
    [0, 25, 160, 2000, 90],
    [0, 30, 170, 2500, 100],
    [1, 20, 170, 1500, 130],
    [1, 25, 180, 2000, 140],
    [1, 30, 185, 2500, 160]
])

# 分离特征和标签
# X为特征值，取前四列，y为标签值，取最后一列
X = data[:,:4]
y = data[:,4]
print(X, y)

# 选择线性回归模型
model = LinearRegression()
# 模型训练
model.fit(X, y)

# 预测35岁，身高175，月消费3000男女的体重
X_new = np.array([[0, 35, 175, 3000], [1, 35, 175, 3000]])
y_predict = model.predict(X_new)

# 打印预测结果  [120,210]
print(y_predict)

# 打印模型系数
print(model.coef_)  #斜率 [ 9.00000000e+01  5.99940006e-04 -2.00000000e+00  5.99940006e-02]
print(model.intercept_)  #截距 289.9970002999698

#y = 289.9970002999698 + 90*X1 + 0.0006*X2 - 2*X3 + 0.006*X4