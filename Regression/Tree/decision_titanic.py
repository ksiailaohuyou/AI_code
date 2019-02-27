import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import  GridSearchCV
from sklearn.tree import DecisionTreeClassifier,export_graphviz

from   sklearn.ensemble  import RandomForestClassifier

def decision():
    """
    决策树对泰坦尼克号进行预测生死
    :return: None
    """
    # 获取数据  也可以是网络数据
    titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

    # 处理数据，找出特征值和目标值
    x = titan[['pclass', 'age', 'sex']]

    y = titan['survived']

    print(x)
    # 缺失值处理
    x['age'].fillna(x['age'].mean(), inplace=True)

    # 分割数据集到训练集合测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 进行处理（特征工程）特征-》类别-》one_hot编码
    dict = DictVectorizer(sparse=False)

    x_train = dict.fit_transform(x_train.to_dict(orient="records"))

    print(dict.get_feature_names())

    x_test = dict.transform(x_test.to_dict(orient="records"))

    # print(x_train)
    # 用决策树进行预测
    dec = DecisionTreeClassifier()
    #
    dec.fit(x_train, y_train)
    #
    # # 预测准确率
    print("预测的准确率：", dec.score(x_test, y_test))
    #
    # # 导出决策树的结构
    export_graphviz(dec, out_file="./tree.dot", feature_names=['年龄', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', '女性', '男性'])

    # # 随机森林进行预测 （超参数调优）
    # rf = RandomForestClassifier()
    #
    # param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    #
    # # 网格搜索与交叉验证
    # gc = GridSearchCV(rf, param_grid=param, cv=2)
    #
    # gc.fit(x_train, y_train)
    #
    # print("准确率：", gc.score(x_test, y_test))
    #
    # print("查看选择的参数模型：", gc.best_params_)



def  rf():
    """
        随机对泰坦尼克号进行预测生死
        :return: None
        """
    # 获取数据
    titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

    # 处理数据，找出特征值和目标值
    x = titan[['pclass', 'age', 'sex']]

    y = titan['survived']

    print(x)
    # 缺失值处理
    x['age'].fillna(x['age'].mean(), inplace=True)

    # 分割数据集到训练集合测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 进行处理（特征工程）特征-》类别-》one_hot编码
    dict = DictVectorizer(sparse=False)

    x_train = dict.fit_transform(x_train.to_dict(orient="records"))

    print(dict.get_feature_names())

    x_test = dict.transform(x_test.to_dict(orient="records"))


    # 随机森林进行预测 （超参数调优）
    rf = RandomForestClassifier()

    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}

    # 网格搜索与交叉验证
    gc = GridSearchCV(rf, param_grid=param, cv=2)

    gc.fit(x_train, y_train)

    print("准确率：", gc.score(x_test, y_test))

    print("查看选择的参数模型：", gc.best_params_)

if __name__ == '__main__':
    decision()
