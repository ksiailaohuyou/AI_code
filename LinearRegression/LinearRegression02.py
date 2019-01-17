import   numpy  as np
from   sklearn.linear_model import   LinearRegression


# 这里相当于是随机X维度X1，rand是随机均匀分布
X=2 * np.random.rand(100,1)


# 人为的设置真实的Y一列，np.random.randn(100, 1)是设置error，randn是标准正太分布
y=4+3*X  +np.random.randn(100,1)


lin_reg=LinearRegression()

lin_reg.fit(X,y)


print(lin_reg.intercept_,lin_reg.coef_)
print(lin_reg.intercept_,lin_reg.coef_)


X_new =np.array([[0],[2]])
print(lin_reg.predict(X_new))



