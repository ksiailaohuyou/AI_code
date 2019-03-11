from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer
import numpy  as np

from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA


def dicvec():
    """
    字典数据抽取
    :return: None
    """
    dict = DictVectorizer(sparse=False)

    # 调用转换
    data = dict.fit_transform(
        [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}])

    print(dict.get_feature_names())
    print(dict.inverse_transform(data))
    print(data)
    # print(data.toarray())


def countvec():
    """
    对文本进行特征值化
    :return: None
    """
    cv = CountVectorizer()

    data = cv.fit_transform(["人生 苦短，我 喜欢 python", "人生漫长，不用 python"])

    print(cv.get_feature_names())

    print(data.toarray())

    return None


def cutword():
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")

    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")

    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")

    # print(next(con1))
    # 转换成列表
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)

    # 吧列表转换成字符串
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)

    return c1, c2, c3


def hanzivec():
    """
    中文特征值化
    :return: None
    """
    c1, c2, c3 = cutword()

    print(c1, c2, c3)

    cv = CountVectorizer()

    data = cv.fit_transform([c1, c2, c3])

    print(cv.get_feature_names())
    print(cv.inverse_transform(data))

    print(data.toarray())

    return None


def tfidfvec():
    """
    中文特征值化
    :return: None
    """
    c1, c2, c3 = cutword()

    print(c1, c2, c3)

    tf = TfidfVectorizer()

    data = tf.fit_transform([c1, c2, c3])

    print(tf.get_feature_names())

    print(data.toarray())

    return None


def mm():
    """
    归一化
    :return:
    """
    mm = MinMaxScaler(feature_range=(0, 2))
    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
    print(data)


def stand():
    """
    标准化
    通过对原始数据进行变换把数据变换到均值为0,标准差为1范围内
    :return:
    """
    std = StandardScaler()
    data = std.fit_transform([[1., -1., 3.], [2., 4., 2.], [4., 6., -1.]])

    print(data)

    return None


def im():
    """
    缺失值处理
    :return:
    """
    # axis=0   为列   1为行
    im = Imputer(missing_values='NaN', strategy='mean', axis=0)

    data = im.fit_transform([[1, 2], [np.nan, 3], [7, 6]])

    print(data)


def var():
    """
    特征选择-删除低方差的特征
    :return:
    """

    var = VarianceThreshold(threshold=0)
    data = var.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    print(data)


def pca():
    """
    主成分分析进行特征降维
    :return:
    """
    # 特征缩减到百分之九十
    pca = PCA(n_components=0.9)
    data = pca.fit_transform([[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]])

    print(data)


if __name__ == '__main__':
    dicvec()
    # mm()
    # pca()
    # var()
    # im()
    # countvec()
    # cutword()
    # hanzivec()
    # tfidfvec()
