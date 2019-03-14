import   pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
import numpy  as np
from sklearn.linear_model import LinearRegression, SGDRegressor,  Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.model_selection import  GridSearchCV
from sklearn.tree import DecisionTreeClassifier,export_graphviz

from   sklearn.ensemble  import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures

from sklearn.decomposition import PCA



def  test1():
    train_data = pd.read_csv('./train.csv')

    X_data = train_data.iloc[:, 2:]
    y_data = train_data['Score']

    dict = DictVectorizer(sparse=False)
    X_data_dict = dict.fit_transform(X_data.to_dict(orient="records"))


    # 特征缩减到百分之九十
    pca = PCA(n_components=0.95)
    data = pca.fit_transform(X_data_dict)

    std_x = StandardScaler()
    X_data_dict = std_x.fit_transform(data)

    X_data_dict_poly = X_data_dict
    x_train_poly, x_test_poly, y_train_poly, y_test_poly = train_test_split(X_data_dict_poly, y_data, test_size=0.25)
    param = {"penalty": ['l2', 'elasticnet'], "max_iter": [30000, 20000, 15000]}
    reg = SGDRegressor()
    # 网格搜索与交叉验证
    gc = GridSearchCV(reg, param_grid=param, cv=3)
    #     print(x_train_poly, y_train_poly)
    gc.fit(x_train_poly, y_train_poly)
    y_sgd_predict = gc.predict(x_test_poly)
    print("梯度下降的均方误差：", mean_squared_error(y_test_poly, y_sgd_predict))
    # print("准确率：", gc.score(x_test_poly, y_test_poly))

    print("查看选择的参数模型：", gc.best_params_)
    print("选择最好的模型是：", gc.best_estimator_)

    #预测
    test_data = pd.read_csv('./test.csv')
    X_test_data = test_data.iloc[:, 2:]

    X_test_data_dict = dict.transform(X_test_data.to_dict(orient="records"))
    X_test_data_dict = pca.transform(X_test_data_dict)
    X_test_data_dict = std_x.transform(X_test_data_dict)
    X_test_data_dict_poly = X_test_data_dict
    y_gc_predict = gc.predict(X_test_data_dict_poly)
    submission_data = pd.read_csv('./submission.csv')
    submission_data['Id'] = test_data['Id']
    submission_data['Score'] = np.rint(y_gc_predict).astype(np.int)
    submission_data.to_csv('submission.csv', header=True)

if __name__=='__main__':
    test1()