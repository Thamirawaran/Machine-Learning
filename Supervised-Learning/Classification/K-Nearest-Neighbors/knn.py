# -*- coding: utf-8 -*-
"""KNN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Gerjcfke6TbB0hIxlCbejtApf7xBev-q
"""

from sklearn.datasets import load_iris
iris = load_iris()

iris.feature_names

X= iris.data
Y= iris.target
X.shape

Y.shape

X

Y

from sklearn.neighbors import KNeighborsClassifier
KNN1 = KNeighborsClassifier(3)
KNN1.fit(X,Y)

print(KNN1.predict(X[[120]]))

print(KNN1.predict(X))

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

print(X_train.shape)

KNN2 = KNeighborsClassifier(3)
KNN2.fit(X_train, Y_train)
P = KNN2.predict(X_test)

Y_test

from sklearn.metrics import accuracy_score, confusion_matrix
print(P)

print(accuracy_score(Y_test,P))

print(confusion_matrix(Y_test,P))