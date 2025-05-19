# 高斯回归
from sklearn.gaussian_process import GaussianProcessClassifier,GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
X = np.linspace(0, 5, 100)[:, np.newaxis]
y = np.sin(X).ravel()

# 添加噪声
y[::5] += 3 * (0.5 - np.random.rand(20))

# 训练模型
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gp.fit(X, y)

# 预测
X_test = np.linspace(0, 5, 300)[:, np.newaxis]
y_pred, sigma = gp.predict(X_test, return_std=True)

# 绘图
plt.plot(X, y, 'r.', markersize=10, label='观测点')

plt.plot(X_test, y_pred, 'b-', label='预测值')
plt.fill(np.concatenate([X_test, X_test[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                         (y_pred + 1.9600 * sigma)[::-1]]), alpha=.5,
         fc='b', ec='None', label='95% 置信区间')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.show()
