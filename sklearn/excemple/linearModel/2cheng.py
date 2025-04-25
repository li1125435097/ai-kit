from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# 加载糖尿病数据集
X, y = load_diabetes(return_X_y=True)
# 只使用一个特征
X = X[:, [2]]  
# 将数据集分为训练集和测试集，测试集占总数据的20%，不进行洗牌
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, shuffle=False)


from sklearn.linear_model import LinearRegression

# 创建线性回归模型，并使用训练集进行拟合
regressor = LinearRegression(positive=True).fit(X_train, y_train)


from sklearn.metrics import mean_squared_error, r2_score

# 使用测试集进行预测
y_pred = regressor.predict(X_test)

# 输出均方误差和决定系数
print(f"Mean squared error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"Coefficient of determination: {r2_score(y_test, y_pred):.2f}")


import matplotlib.pyplot as plt

# 创建一个包含两个子图的图形
fig, ax = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)

# 绘制训练集数据点和模型预测结果
ax[0].scatter(X_train, y_train, label="Train data points")
ax[0].plot(
    X_train,
    regressor.predict(X_train),
    linewidth=3,
    color="tab:orange",
    label="Model predictions",
)
ax[0].set(xlabel="Feature", ylabel="Target", title="Train set")
ax[0].legend()

# 绘制测试集数据点和模型预测结果
ax[1].scatter(X_test, y_test, label="Test data points")
ax[1].plot(X_test, y_pred, linewidth=3, color="tab:orange", label="Model predictions")
ax[1].set(xlabel="Feature", ylabel="Target", title="Test set")
ax[1].legend()

# 设置图形标题
fig.suptitle("Linear Regression")

# 显示图形
plt.show()
