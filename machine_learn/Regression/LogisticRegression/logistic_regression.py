import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import matplotlib.pyplot as plt
from time import time

iris = datasets.load_iris()
print(list(iris.keys()))
# print(iris['DESCR'])
print(iris['feature_names'])
# 之后可用多个维度训练
print(iris['data'][-1])
# 最后一条作为测试数据
# X = iris['data'][:, 3:]
X = iris['data'][:-1]
# print(X)

y = iris['target'][:-1]
# print(y)


log_reg = LogisticRegression(multi_class='ovr', solver='sag',max_iter=10000)
log_reg.fit(X, y)
# 0到3 按1000分段
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
# print(X_new)

# y_proba = log_reg.predict_proba(X_new)
# print(y_proba)

# y_hat =log_reg.predict(X_new)
# print(y_hat)


# plt.plot(X_new, y_proba[:, 2], 'g-', label='Iris-Virginica')
# plt.plot(X_new, y_proba[:, 1], 'r-', label='Iris-Versicolour')
# plt.plot(X_new, y_proba[:, 0], 'b--', label='Iris-Setosa')
# plt.show()

# print(log_reg.predict([[1.7], [1.5]]))
print(log_reg.predict([[5.9, 3.,  5.1 ,1.8]]))
