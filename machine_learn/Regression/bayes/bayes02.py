"""
(机器学习) 投融资新闻预测
项目部署:supervisor(ml_bys)
项目描述:将国外的投融资新闻使用训练好的模型进行预测,使用预测结果作为排序依据前台展示,创投新闻发布时将新闻数据供给模型进行再一次学习.
"""
from flask import Flask,request
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os
from queue import Queue
from threading import Thread
import time

app = Flask(__name__)

q = Queue()

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# 模型
clf = None
# 词袋
vectorizer = None
# 计数,模型更新
count = 0
# 学习步长
learn_step = 50
#
data_set = {}

def load_model():
    global clf,vectorizer
    with open(os.path.join(APP_ROOT,'model'),'rb') as f:
        clf = pickle.loads(f.read())
    with open(os.path.join(APP_ROOT,'vectorizer'),'rb') as f:
        vectorizer = pickle.loads(f.read())

# 线程:轮询队列,保存供学习的数据
def append(q):
    global count,data_set
    while True:
        data = q.get()
        if data[0] in data_set:
            time.sleep(1)
            continue
        try:
            with open(os.path.join(APP_ROOT,'data'),'a',encoding='utf8')as f:
                f.write(data[0] + '----' + str(data[1]) + '\n')
                count+=1
                data_set[data[0]]=1
        except Exception as e:
            print(e)
            continue
        update_model()
        time.sleep(1)


# 落地model
def save_model():
    print('正在保存词袋!')
    with open(os.path.join(APP_ROOT,'vectorizer'),'wb')as f:
        f.write(pickle.dumps(vectorizer))
    print('保存词袋完成!')
    print('正在保存model!')
    with open(os.path.join(APP_ROOT,'model'),'wb')as f:
        f.write(pickle.dumps(clf))
    print('保存model完成!')



# 更新model
def update_model():
    global clf,vectorizer,count
    if count >=learn_step:
        print('开始更新!')
        corpus = []
        labels = []
        with open(os.path.join(APP_ROOT,'data'),'r',encoding='utf8')as f:
            for line in f.readlines():
                if line.strip():
                    sentence, label = line.split('----')
                    corpus.append(sentence)
                    labels.append(int(label.strip()))
        vectorizer = CountVectorizer()
        # 将corpus传入向量对象,产生词袋向量
        fea_train = vectorizer.fit_transform(corpus)
        # 将词袋与标签传入算法,产生model
        clf = MultinomialNB(alpha=1)
        clf.fit(fea_train, labels)
        count = 0
        save_model()



@app.route('/')
def hello():
    return 'Hello World!'

# 学习
@app.route('/learning/',methods=['POST'])
def learning():
    sentence = request.form['sentence'].replace('\n',' ')
    result = (request.form['result'])
    if sentence and result:
        q.put([sentence,result])
        return '1'
    else:
        return '0'

# 预测
@app.route('/forecast/',methods=['POST'])
def forecast():
    sentence = []
    sentence.append(request.form['sentence'])
    vectorizer2 = CountVectorizer(vocabulary=vectorizer.vocabulary_)
    fea_test = vectorizer2.fit_transform(sentence)
    pred = clf.predict(fea_test)
    # print(pred)
    return str(pred[0])


if __name__ == '__main__':
    load_model()
    t1 = Thread(target=append, args=(q,))
    t1.start()
    # app.run()
    app.run(host="0.0.0.0", port=int("7758"))
