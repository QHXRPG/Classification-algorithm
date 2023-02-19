import graphviz
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn.preprocessing import StandardScaler

#%% 数据预处理
data = load_iris()
df = pd.DataFrame(data.data)
print(np.sum(df.isnull()))   #缺陷值检验
df.fillna(df.mean())
standard = StandardScaler()  #标准化
data.data = standard.fit_transform(data.data)
x_train,x_test,y_train,y_test = train_test_split(data.data,data.target,train_size=0.8)
classifier_tree = DecisionTreeClassifier(criterion="entropy")
classifier_tree.fit(x_train,y_train)
y_p = classifier_tree.predict(x_test)
print(classifier_tree.score(x_test,y_test))

#%% 可视化
vision_tree = tree.export_graphviz(classifier_tree,
                                   feature_names=data.feature_names,
                                   class_names=data.target_names,)
graph = graphviz.Source(vision_tree)
graph.render("iris")   #将图形保存为iris.pdf文件。
graph.view()           # 直接打开pdf文件展示