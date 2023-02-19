from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.preprocessing import StandardScaler

data = load_iris()
standard = StandardScaler()
data.data = standard.fit_transform(data.data)
x_train,x_test,y_train,y_test = train_test_split(data.data,data.target,train_size=0.8)
'''
gamma:‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features
C:C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，
  这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，
  将他们当成噪声点，泛化能力较强。
decision_function_shape: ovo：one versus one，一对一。即一对一的分类器，这时对K个类别需要构建K * (K - 1) / 2个分类器
                         ovr：one versus rest，一对其他，这时对K个类别只需要构建K个分类器。
'''
classifier = svm.SVC(kernel='linear',gamma=0.1, C=0.1, decision_function_shape='ovo')
classifier.fit(x_train,y_train)
y_p = classifier.predict(x_test)
a=classifier.score(x_test,y_test)
print(metrics.classification_report(y_test,y_p))