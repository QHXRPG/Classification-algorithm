import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

data = load_iris()
X = data.data[:,:2]
Y = data.target
cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000','#00FF00','#0000FF'])
knn = KNeighborsClassifier(n_neighbors=10, weights='uniform')
knn.fit(X,Y)
#画出决策边界
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min,x_max,0.02),np.arange(y_min, y_max,0.02))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)  # ravel():展平
plt.figure()
plt.pcolormesh(xx,yy,Z,cmap=cmap_light)
plt.scatter(X[:,0],X[:,1], c=Y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()