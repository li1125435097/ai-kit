from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, learning_rate_init=0.02, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
plt.scatter(X_test[:,2], y_test)
plt.scatter(X_test[:,2], clf.predict(X_test), marker='x')
plt.show()

