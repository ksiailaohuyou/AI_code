from sklearn.datasets import fetch_20newsgroups

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfVectorizer
from   sklearn.metrics import   classification_report

def naviebayes():
    """
    朴素贝叶斯进行文本分类
    :return: None
    """

    news = fetch_20newsgroups(data_home='../../datas/20Newsgroups', subset='all',download_if_missing=False)

    # 进行数据分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)

    # 对数据集进行特征抽取
    tf = TfidfVectorizer()

    # 以训练集当中的词的列表进行每篇文章重要性统计['a','b','c','d']
    x_train = tf.fit_transform(x_train)

    x_test = tf.transform(x_test)

    # 进行朴素贝叶斯算法的预测
    mlt = MultinomialNB()

    mlt.fit(x_train, y_train)

    y_predict = mlt.predict(x_test)
    # 得出准确率
    print("准确率为：", mlt.score(x_test, y_test))
    print("每个类别的精确率和召回率：", classification_report(y_test, y_predict, target_names=news.target_names))


def  naviebayes01():
    news = fetch_20newsgroups(data_home='../../datas/20Newsgroups', subset='all', download_if_missing=False)

    # 进行数据分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)

    # 对数据集进行特征抽取
    tf = TfidfVectorizer()

    # 以训练集当中的词的列表进行每篇文章重要性统计['a','b','c','d']
    x_train = tf.fit_transform(x_train)
    # print(tf.get_feature_names())

    x_test = tf.transform(x_test)

    # 进行朴素贝叶斯算法的预测
    mlt = MultinomialNB(alpha=1.0)

    # print(x_train.toarray())

    mlt.fit(x_train, y_train)

    y_predict = mlt.predict(x_test)

    print("预测的文章类别为：", y_predict)

    # 得出准确率
    print("准确率为：", mlt.score(x_test, y_test))

    print("每个类别的精确率和召回率：", classification_report(y_test, y_predict, target_names=news.target_names))

if __name__ == '__main__':
    naviebayes()
