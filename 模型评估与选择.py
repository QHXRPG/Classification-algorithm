import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
#%% 训练一个模型
data = load_iris()
x = data.data
y = data.target
x,y = x[y!=2], y[y!=2]  #选取前两类数据并增加扰动
random_state = np.random.RandomState(0) #定义随机状态
n_sample, n_feature = x.shape
x = np.c_[x, random_state.randn(n_sample,200*n_feature)]  #将噪声作为特征直接加在x后面
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=0)
classifier = svm.SVC(kernel='linear', probability=True, random_state=random_state)
classifier.fit(x_train,y_train)

#%% 在测试集上进行预测评估
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
y_p = classifier.predict(x_test)
print("SVM 输出训练集的精度: %.3f"%classifier.score(x_test,y_test))
print("Precision: %.3f"%precision_score(y_true=y_test, y_pred=y_p))
print("Recall: %.3f"%recall_score(y_true=y_test, y_pred=y_p))
print("F1: %.3f"%f1_score(y_true=y_test, y_pred=y_p))
print("F_beta: %.3f"%fbeta_score(y_true=y_test, y_pred=y_p,beta=0.8))

#%% 绘制ROC曲线
y_score = classifier.fit(x_train,y_train).decision_function(x_test)  #decision_function: 计算样本点到分割超平面的函数距离
fpr, tpr, threshold = roc_curve(y_true=y_test, y_score=y_score)  #计算真正例率，和假正例率
roc_auc = auc(fpr, tpr) #计算AUC值  AUC:得到正样本的概率大于负样本概率的概率
plt.figure(figsize=(8,4))
plt.plot(fpr,tpr, color='darkorange', label='ROC curve(area=%0.2f)'%roc_auc)
plt.plot([0,1],[0,1],color='navy',linestyle='--')
plt.show()

#%% 简单交叉验证
data = load_iris()
x = data.data
y = data.target
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=5)

#%% k-折交叉验证
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
kf.split(x)
train_index_list=np.array([])
test_index_list=np.array([])
for train_index,test_index in kf.split(x):
    print('train:',train_index)
    print('test:',test_index)
    train_index_list=np.concatenate([train_index,train_index])
    test_index_list=np.concatenate([test_index,test_index])
