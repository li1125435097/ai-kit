import numpy as np

# 创建一个随机数生成器
rng = np.random.RandomState(0)
# 设置样本数、特征数和有效特征数
n_samples, n_features, n_informative = 50, 100, 10
# 生成时间步长
time_step = np.linspace(-2, 2, n_samples)
# 生成频率
freqs = 2 * np.pi * np.sort(rng.rand(n_features)) / 0.01
# 初始化X矩阵
X = np.zeros((n_samples, n_features))

# 生成X矩阵
for i in range(n_features):
    X[:, i] = np.sin(freqs[i] * time_step)

# 生成索引
idx = np.arange(n_features)
# 生成真实系数
true_coef = (-1) ** idx * np.exp(-idx / 10)
# 稀疏化系数
true_coef[n_informative:] = 0  
# 生成y
y = np.dot(X, true_coef)

# 添加噪声
for i in range(n_features):
    X[:, i] = np.sin(freqs[i] * time_step + 2 * (rng.random_sample() - 0.5))
    X[:, i] += 0.2 * rng.normal(0, 1, n_samples)

y += 0.2 * rng.normal(0, 1, n_samples)

# 划分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 计时
from time import time

# 使用Lasso回归
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error

t0 = time()
lasso = Lasso(alpha=0.14).fit(X_train, y_train)
print(f"Lasso fit done in {(time() - t0):.3f}s")

# 预测
y_pred_lasso = lasso.predict(X_test)
# 计算R2
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(f"Lasso r^2 on test data : {r2_score_lasso:.3f}")
# 计算均方误差
print(f"Lasso mean squared error on test data : {mean_squared_error(y_test, y_pred_lasso):.3f}")
# 打印系数和截距
print(lasso.coef_,lasso.intercept_)
