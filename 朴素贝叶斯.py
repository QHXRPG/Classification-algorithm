from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
data = load_iris()
classifier = GaussianNB()
x = data.data
y = data.target
classifier.fit(x,y)
y_p = classifier.predict(x)
print(classifier.score(x,y))