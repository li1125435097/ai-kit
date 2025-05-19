# 导入所需的库
from sklearn.datasets import load_iris  # 导入鸢尾花数据集
from sklearn.model_selection import train_test_split  # 导入数据集分割工具
from sklearn.neighbors import KNeighborsClassifier  # 导入K近邻分类器
from sklearn.pipeline import Pipeline  # 导入管道工具
from sklearn.preprocessing import StandardScaler  # 导入标准化处理器
import matplotlib.pyplot as plt  # 导入绘图库
from sklearn.inspection import DecisionBoundaryDisplay  # 导入决策边界显示工具

# 加载鸢尾花数据集，并将数据转换为DataFrame格式
iris = load_iris(as_frame=True)
# 选择花萼长度和花萼宽度作为特征
X = iris.data[["sepal length (cm)", "sepal width (cm)"]]
# 选择目标变量（花的种类）
y = iris.target
# 将数据集分割为训练集和测试集，保持类别比例，随机种子设为0
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# 创建一个管道，包含标准化处理器和K近邻分类器
clf = Pipeline(
    steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=11))]
)

# 创建两个子图，用于绘制不同权重参数下的决策边界
_, axs = plt.subplots(ncols=2, figsize=(12, 5))

# 遍历不同的权重参数（"uniform"和"distance"）
for ax, weights in zip(axs, ("uniform", "distance")):
    print(weights)
    # 设置K近邻分类器的权重参数，并拟合训练数据
    clf.set_params(knn__weights=weights).fit(X_train, y_train)
    # 使用拟合好的模型绘制决策边界
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X_test,
        response_method="predict",
        plot_method="pcolormesh",
        xlabel=iris.feature_names[0],  # 设置x轴标签
        ylabel=iris.feature_names[1],  # 设置y轴标签
        shading="auto",  # 自动着色
        alpha=0.5,  # 设置透明度
        ax=ax,  # 指定绘图轴
    )
    # 在决策边界图上绘制测试数据的散点图
    scatter = disp.ax_.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors="k")
    # 添加图例
    disp.ax_.legend(
        scatter.legend_elements()[0],
        iris.target_names,
        loc="lower left",
        title="Classes",
    )
    # 设置子图标题
    _ = disp.ax_.set_title(
        f"3-Class classification\n(k={clf[-1].n_neighbors}, weights={weights!r})"
    )

# 显示绘图
plt.show()
