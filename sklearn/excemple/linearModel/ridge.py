from sklearn import linear_model
import random
intercept = round(random.uniform(0, 1), 1)
reg = linear_model.Ridge(alpha=intercept)
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
result = reg.predict([[3, 3]])
print(result)
print(intercept)
print(reg.coef_)
print(reg.intercept_)