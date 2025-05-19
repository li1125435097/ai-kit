import numpy as np


# 定义一个函数，输入参数x，返回sin(2πx)
def func(x):
    return np.sin(2 * np.pi * x)


# 定义训练集和测试集的大小
size = 25
# 创建一个随机数生成器，种子为1234
rng = np.random.RandomState(1234)
# 生成训练集x_train，范围为0.0到1.0，大小为size
x_train = rng.uniform(0.0, 1.0, size)
# 生成训练集y_train，为func(x_train)加上服从正态分布的随机数，标准差为0.1，大小为size
y_train = func(x_train) + rng.normal(scale=0.1, size=size)
# 生成测试集x_test，范围为0.0到1.0，大小为100
x_test = np.linspace(0.0, 1.0, 100)

# 导入贝叶斯岭回归模型
from sklearn.linear_model import BayesianRidge

# 定义多项式的阶数
n_order = 3
# 将训练集x_train转换为多项式形式
X_train = np.vander(x_train, n_order + 1, increasing=True)
# 将测试集x_test转换为多项式形式
X_test = np.vander(x_test, n_order + 1, increasing=True)
# 创建贝叶斯岭回归模型，设置容忍度为1e-6，不拟合截距，计算得分
reg = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)

# 导入绘图库
import matplotlib.pyplot as plt

# 创建一个1行2列的子图
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
# 遍历子图
for i, ax in enumerate(axes):
    # 贝叶斯岭回归，不同初始值对
    if i == 0:
        init = [1 / np.var(y_train), 1.0]  # 默认值
    elif i == 1:
        init = [1.0, 1e-3]
        reg.set_params(alpha_init=init[0], lambda_init=init[1])
    # 拟合模型
    reg.fit(X_train, y_train)
    # 预测测试集，返回预测值和标准差
    ymean, ystd = reg.predict(X_test, return_std=True)

    # 绘制sin(2πx)曲线
    ax.plot(x_test, func(x_test), color="blue", label="sin($2\\pi x$)")
    # 绘制观察点
    ax.scatter(x_train, y_train, s=50, alpha=0.5, label="observation")
    # 绘制预测均值
    ax.plot(x_test, ymean, color="red", label="predict mean")
    # 绘制预测标准差
    ax.fill_between(
        x_test, ymean - ystd, ymean + ystd, color="pink", alpha=0.5, label="predict std"
    )
    # 设置y轴范围
    ax.set_ylim(-1.3, 1.3)
    # 添加图例
    ax.legend()
    # 设置标题
    title = "$\\alpha$_init$={:.2f},\\ \\lambda$_init$={}$".format(init[0], init[1])
    if i == 0:
        title += " (Default)"
    ax.set_title(title, fontsize=12)
    # 添加文本
    text = "$\\alpha={:.1f}$\n$\\lambda={:.3f}$\n$L={:.1f}$".format(
        reg.alpha_, reg.lambda_, reg.scores_[-1]
    )
    ax.text(0.05, -1.0, text, fontsize=12)

# 调整布局
plt.tight_layout()
# 显示图像
plt.show()
