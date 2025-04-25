import numpy as np
from sklearn.linear_model import SGDRegressor,LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#输入训练集数据
x=np.array([[100],[113],[90],[89],[60],[70],[50],[45],[55],[78]])	#房屋面积
y=np.array([[301],[324],[285],[296],[200],[260],[300],[120],[180],[245]])	#售价

#建立模型，训练模型
model=SGDRegressor(loss='huber',max_iter=5000,random_state=42)	#建立基于梯度下降法的线性回归模型
model.fit(x,y.ravel())				#开始训练模型

#预测
ypg = model.predict(x)

#求解线性回归方程参数
print("w=",model.coef_,"b=",model.intercept_)

model2 = LinearRegression().fit(x, y.ravel())
ypl = model2.predict(x)

# 比较均方误差
print("梯度下降均方误差：",np.mean((ypg-y)**2))
print("线性模型均方误差：",np.mean((ypl-y)**2))

# 比较score
print("梯度下降score：",model.score(x,y.ravel()))
print("线性模型score：",model2.score(x,y.ravel()))
