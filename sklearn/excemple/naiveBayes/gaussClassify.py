from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

print("Accuracy: ", gnb.score(X_test, y_test))
# Accuracy:  0.9666666666666667

plt.subplot(121)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
plt.subplot(122)
plt.scatter(X_test[:, 0], X_test[:, 1], c=gnb.predict(X_test))
plt.show()