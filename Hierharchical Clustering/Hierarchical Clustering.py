#Hieharchical Clustering

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the dataset
dataset=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values

#Using the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title("Dendogram")
plt.xlabel("Customers")
plt.ylabel("Eucl Dist")
plt.show()

#Fitting hieharchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity = 'euclidean', linkage = "ward")
y_hc = hc.fit_predict(X)

#Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s= 100, c='red', label='Wise')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s= 100, c='blue', label='standard')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s= 100, c='green', label='Desirable customers')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s= 100, c='cyan', label='careless')
plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], s= 100, c='magenta', label='poor')
plt.title('Clusters of clients')
plt.xlabel('Annual Income(K$)')
plt.ylabel("Spending score (1-100)")
plt.legend()
plt.show()