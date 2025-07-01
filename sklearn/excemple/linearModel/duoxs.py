from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

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

# 生成2次多项式训练数据
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
print(X_poly.shape) # (6, 15)

# 模型训练
model = LinearRegression()
model.fit(X_poly, y)

# 查看模型参数
print(model.coef_)  #系数 [ 8.01658341e-14  4.09432366e-05  9.26648414e-05  2.64626852e-04 9.26648414e-03  4.09432366e-05  5.72074220e-04  5.72720164e-03  3.67358036e-02  9.26606140e-04  1.30080231e-03  4.63281933e-02  9.43459168e-03 -2.23319511e-03 -4.22744688e-04]
print(model.intercept_)#截距 -86.69398530179599

# 预测
print(model.predict(poly.fit_transform([[0, 35, 175, 3000],[1, 35, 175, 3000]]))) # [126.5225629 237.7523386]

# 特征自变量
print(poly.get_feature_names_out()) # 自变量 ['1' 'x0' 'x1' 'x2' 'x3' 'x0^2' 'x0 x1' 'x0 x2' 'x0 x3' 'x1^2' 'x1 x2' 'x1 x3' 'x2^2' 'x2 x3' 'x3^2']

# 组合系数和自变量
y = str(model.intercept_) + '+ '
for i in range(len(model.coef_)):
    y += f'{round(model.coef_[i],4)} * {poly.get_feature_names_out()[i].replace(" ","*")} + '

# 输出多项式函数  y = -86.69398530179599+ 0.0 * 1 + 0.0 * x0 + 0.0001 * x1 + 0.0003 * x2 + 0.0093 * x3 + 0.0 * x0^2 + 0.0006 * x0*x1 + 0.0057 * x0*x2 + 0.0367 * x0*x3 + 0.0009 * x1^2 + 0.0013 * x1*x2 + 0.0463 * x1*x3 + 0.0094 * x2^2 + -0.0022 * x2*x3 + -0.0004 * x3^2
print('y =',y[:-3])