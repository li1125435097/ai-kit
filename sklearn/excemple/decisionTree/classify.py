from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# 获取训练数据
X,y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 输出评分
print('score: ',model.score(X_test, y_test))

# 画图
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, s=100)
plt.scatter(X_test[:,0], X_test[:,1], c=model.predict(X_test),marker='x',s=160)
plt.show()
