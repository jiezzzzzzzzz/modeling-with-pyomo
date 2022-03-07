import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


df = pd.read_excel('azs.xlsx')

scaler = StandardScaler()

X = scaler.fit_transform(df)

colors = ['b', 'g', 'c', 'r', 'm', 'y']
markers = ['o', 'v', 's', 'p', 'h', 'd']


K = 6
kmeans_model = KMeans(n_clusters=K).fit(X)

print("Координаты центроид: ", kmeans_model.cluster_centers_)
centers = np.array(kmeans_model.cluster_centers_)

plt.plot()
plt.title('Центроиды')

X1 = []
X2 = []
# рисуем центроиды
for i, l in enumerate(kmeans_model.labels_):
    plt.plot(X[i][0], X[i][1], color=colors[l], marker=markers[l], ls='None')
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    X1.append(X[i][0])
    X2.append(X[i][1])

plt.scatter(centers[:,0], centers[:,1], marker="X", color='r')
plt.show()

df['cluster'] = kmeans_model.labels_
df['Latitude'] = X1
df['Longitude'] = X2
df.to_csv('clusters_coordinates.csv')

