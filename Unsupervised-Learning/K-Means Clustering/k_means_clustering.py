# -*- coding: utf-8 -*-
"""K-Means_Clustering.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1LTIokIrOcG8Kjoft-3TVKRFSdt_WTM1Z

Import required libraries
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

"""Plot"""

X,Y = make_blobs(n_samples=300,centers=4, cluster_std=0.6)
plt.scatter(X[:,0],X[:,1],s=50)
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

Kmeans_model = KMeans(n_clusters=4)
Kmeans_model.fit(X_train)