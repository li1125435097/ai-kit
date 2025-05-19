import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split

X,y = datasets.load_iris(return_X_y=True)
X = X[:,:2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel).fit(X_train, y_train)
print(X_train.shape, y_train.shape)

y_pred = gpc.predict(X_test)
print('accuracy: ',gpc.score(X_test, y_test))

# plt.figure(figsize=(8, 6))  
plt.subplot(121)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.Paired, s=100)
plt.subplot(122)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Paired, s=180, marker='x', linewidths=3)
plt.show()