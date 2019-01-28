from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data[:, :2]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

rnd_tree = RandomForestClassifier(n_estimators=15, max_leaf_nodes=16, n_jobs=-1)
rnd_tree.fit(X_train, y_train)
# BaggingClassifier  具有并行含义，
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(splitter="random", max_leaf_nodes=16),
    n_estimators=15, max_samples=1.0, bootstrap=True, n_jobs=1
)
bag_clf.fit(X_train, y_train)

y_pred_rf = rnd_tree.predict(X_test)
y_pred_bag = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_rf))
print(accuracy_score(y_test, y_pred_bag))



# Feature Importance   用于提取有效特征
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris["data"], iris['target'])
for name, score in zip(iris['feature_names'], rnd_clf.feature_importances_):
    print(name, score)