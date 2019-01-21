from   sklearn.linear_model import  ElasticNet
from sklearn.linear_model import SGDRegressor
import numpy as np

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)


vec=np.array(1).reshape(1,-1)
elastic_net = ElasticNet(alpha=0.0001, l1_ratio=0.15)
elastic_net.fit(X, y)
print(elastic_net.predict(vec))

sgd_reg = SGDRegressor(penalty='elasticnet', max_iter=1000)
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict(vec))