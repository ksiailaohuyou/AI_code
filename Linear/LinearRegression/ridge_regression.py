import   numpy   as np
from   sklearn .linear_model import  Ridge
from   sklearn .linear_model import  SGDRegressor


X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)


ridge_reg = Ridge(alpha=1,solver='sag')
ridge_reg.fit(X,y)
# 跟进numpy库官网的介绍，这里的-1被理解为unspecified value，意思是未指定为给定的。如果我只需要特定的行数，列数多少我无所谓，我只需要指定行数，那么列数直接用-1代替就行了，计算机帮我们算赢有多少列，反之亦然。
vec= np.array([[1,]]).reshape(1,-1)

print(ridge_reg.predict(vec))
print(ridge_reg.intercept_)
print(ridge_reg.coef_)

sgd_reg=SGDRegressor(penalty='l2',max_iter=1000)
sgd_reg.fit(X,y.ravel())
print(sgd_reg.predict(vec))
print("W0=", sgd_reg.intercept_)
print("W1=", sgd_reg.coef_)
