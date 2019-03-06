from  sklearn.metrics import silhouette_score
import matplotlib.pyplot   as plt
import pandas  as  pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

prior = pd.read_csv("../datas/Instacart Market Basket Analysis/order_products__prior.csv")
products = pd.read_csv("../datas/Instacart Market Basket Analysis/products.csv")
orders = pd.read_csv("../datas/Instacart Market Basket Analysis/orders.csv")
aisles = pd.read_csv("../datas/Instacart Market Basket Analysis/aisles.csv")

# 合并四张表到一张表  （用户-物品类别）
_mg = pd.merge(prior, products, on=['product_id', 'product_id'])
_mg = pd.merge(_mg, orders, on=['order_id', 'order_id'])
mt = pd.merge(_mg, aisles, on=['aisle_id', 'aisle_id'])

# 交叉表（特殊的分组工具）
cross = pd.crosstab(mt['user_id'], mt['aisle'])

# 进行主成分分析  将特征降到原来的百分之九十
pca = PCA(n_components=0.9)

data = pca.fit_transform(cross)
# 把样本数量减少
x = data[:500]
# x.shape

# 聚类  k-means 聚为四类
km = KMeans(n_clusters=4)
km.fit(x)
predict = km.predict(x)

# 设计二维图片
plt.figure(figsize=(10, 10))
# 建立四个颜色的列表
colored = ['orange', 'green', 'blue', 'purple']
colr = [colored[i] for i in predict]
plt.scatter(x[:, 1], x[:, 20], color=colr)
plt.xlabel("1")
plt.ylabel("20")
plt.show()


# 评判聚类效果，轮廓系数
silhouette_score(x,predict)