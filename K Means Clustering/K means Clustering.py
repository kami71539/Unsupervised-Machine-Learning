#K means clustering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values

#Using the elbow method to find the optimal numbers of clusters
from sklearn.cluster import KMeans
WCSS= []
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', max_iter = 300, n_init= 10, random_state=2)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)
plt.plot(range(1,11), WCSS)
plt.title('The Elbow method')
plt.xlabel("Number of clusters")
plt.ylabel('WCSS')
plt.show()

#Applying k-means to the dataset
kmeans=KMeans(n_clusters=5, init = 'k-means++', max_iter = 300, n_init = 10, random_state=0)
y_kmeans=kmeans.fit_predict(X)

#Visualing the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s= 100, c='red', label='Wise')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s= 100, c='blue', label='standard')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s= 100, c='green', label='Desirable customers')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s= 100, c='cyan', label='careless')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s= 100, c='magenta', label='poor')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s= 300, c='yellow', label='centeroids')
plt.title('Clusters of clients')
plt.xlabel('Annual Income(K$)')
plt.ylabel("Spending score (1-100)")
plt.legend()
plt.show()