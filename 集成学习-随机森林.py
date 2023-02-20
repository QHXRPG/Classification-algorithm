from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
standard = StandardScaler()
data = load_wine()
data.data = standard.fit_transform(data.data)
x = data.data
y = data.target
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.5,random_state=0)
random_forest = RandomForestClassifier()
classifier_tree = DecisionTreeClassifier()
random_forest.fit(x_train,y_train)
classifier_tree.fit(x_train,y_train)
y_p1 = random_forest.predict(x_test)
y_p2 = classifier_tree.predict(x_test)
print("随机森林：",random_forest.score(x_test,y_test))
print("决策树：",classifier_tree.score(x_test,y_test))
