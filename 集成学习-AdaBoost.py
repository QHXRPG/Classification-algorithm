import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

#%% 数据准备
x1,y1 = make_gaussian_quantiles(cov=3.0, n_samples=500, n_features=2,
                                n_classes=3,random_state=1)
x2,y2 = make_gaussian_quantiles(mean=(3,3), cov=1.5, n_samples=400,
                                n_classes=2,random_state=1)
x = np.concatenate((x1,x2))
y = np.concatenate((y1,-y2-1))  # -y2-1: 让标签有2+3=5种
plt.scatter(x[:,0], x[:,1], c=y, marker='o')
plt.show()

#%% AdaBoost分类
adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=200)
adaboost.fit(x,y)
x_min,x_max = x[:,0].min()-1, x[:,0].max()+1
y_min,y_max = x[:,1].min()-1, x[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min,x_max,0.02),np.arange(y_min,y_max,0.02))
z=adaboost.predict(np.c_[xx.ravel(), yy.ravel()])
z=z.reshape(xx.shape)
cs = plt.contourf(xx,yy,z,cmap=plt.cm.Paired)
plt.scatter(x[:,0],x[:,1],marker='o', c=y)
plt.show()
