# -*- coding: UTF-8 -*-

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from time import time



iris = datasets.load_iris()
# print(list(iris.keys()))
# print(iris['DESCR'])
# print(iris['feature_names'])
#之后可用多个维度训练
# X = iris['data'][:, 3:]
X = iris['data']
# print(X)

# print(iris['target'])
y = iris['target']
# y = (iris['target'] == 2).astype(np.int)
# print(y)


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
#
#
start = time()
param_grid = {"tol": [1e-4, 1e-3, 1e-2],
              "C": [0.4, 0.6, 0.8,1.0,1.2],
              'multi_class':['ovr','multinomial'],
              # 'max_iter':[1000,10000,5000],
              }
# log_reg = LogisticRegression(multi_class='ovr', solver='sag')
log_reg = LogisticRegression(C=0.8, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=10000, multi_class='ovr',
          n_jobs=None, penalty='l2', random_state=None, solver='sag',
          tol=0.0001, verbose=0, warm_start=False)


grid_search = GridSearchCV(log_reg, param_grid=param_grid, cv=3)
grid_search.fit(X, y)
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)


# 最佳超参数模型
print(grid_search.best_estimator_)
# X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
# print(X_new)

# y_proba = grid_sroba(X_new)
# y_hat = grid_search.predict(X_new)
# # print(y_proba)
# # print(y_hat)
#
# plt.plot(X_new, y_proba[:, 2], 'g-', label='Iris-Virginica')
# plt.plot(X_new, y_proba[:, 1], 'r-', label='Iris-Versicolour')
# plt.plot(X_new, y_proba[:, 0], 'b--', label='Iris-Setosa')
# plt.show()
#
# print(log_reg.predict([[1.7], [1.5]]))earch.predict_p


