import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

plt.plot(X, y, 'b.')
# plt.show()

d = {1: 'g-', 2: 'r+', 10: 'y*'}

for i in d:
    print(i)
    # 多项式
    poly_features = PolynomialFeatures(degree=i, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    # print(X[0])
    # print(X_poly[0])
    # print(X_poly[:, 0])

    lin_reg = LinearRegression(fit_intercept=True)
    lin_reg.fit(X_poly, y)
    print(lin_reg.intercept_, lin_reg.coef_)

    y_predict = lin_reg.predict(X_poly)
    plt.plot(X_poly[:, 0], y_predict, d[i])

plt.show()
