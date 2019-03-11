import pandas  as pd
import matplotlib.pyplot  as plt
import numpy  as np
from sklearn.model_selection import train_test_split,cross_val_score

data = pd.read_csv('creditcard.csv')
# print(data.head())

"""
#   查看数据发布
count_classes=  pd.value_counts(data['Class'],sort=True).sort_index()#计数并排序
count_classes.plot(kind='bar')
plt.title("Fraud class histogram")
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()
"""


# 归一化数据，并除去无用数据
from sklearn.preprocessing import StandardScaler
data['normAmount'] = StandardScaler().fit_transform(np.array(data['Amount']).reshape(-1, 1))#做标准归一化
data = data.drop(['Time', 'Amount'], axis=1)
# print(data.head())




X=data.ix[:,data.columns !='Class']#取出所有属性，不包含class的这一列
Y= data.ix[:, data.columns == 'Class']#另y等于class这一列


# train_test_split(test_size=,random_state=)