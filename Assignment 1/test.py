# Common imports
import numpy as np

# import data
data = np.genfromtxt('Assignment 1\Data\gdsc_expr_postCB.csv', delimiter=',')
data = data[1:,1:]
data = np.transpose(data)

from sklearn.metrics import jaccard_score
from sklearn.metrics import rand_score
from sklearn.metrics import adjusted_rand_score

from sklearn.cluster import AgglomerativeClustering
import prettytable as pt

# Eucledian distance
euc_agglo = AgglomerativeClustering(n_clusters=3, linkage='average', metric='euclidean')
euc_agglo.fit(data)
euc_labels = euc_agglo.labels_

data = np.genfromtxt('Assignment 1\Data\gdsc_expr_postCB.csv', delimiter=',')
data = data[1:,1:]
data = np.transpose(data)

cos_agglo = AgglomerativeClustering(n_clusters=3, linkage='average', metric='cosine')
cos_agglo.fit(data)
cos_labels = cos_agglo.labels_

# Jaccard score for each pair of clusters
average_jaccards = []
for i in range(3):
    for j in range(3):
        average_jaccards.append(jaccard_score(euc_labels == i, cos_labels == j).round(3))

# Make table of jaccard scores
t = pt.PrettyTable()
t.field_names = ['', 'Cluster 1', 'Cluster 2', 'Cluster 3']
t.title = 'Jaccard Score for Each Pair of Clusters'

for i in range(3):
    t.add_row([f'Cluster {i+1}', average_jaccards[i], average_jaccards[i+3], average_jaccards[i+6]])

print(t)

# Rand score
cos_rand = rand_score(euc_labels, cos_labels).__round__(3)
print("Rand Score: ", cos_rand)

# Adjusted rand score
cos_adj_rand = adjusted_rand_score(euc_labels, cos_labels).__round__(3)
print("Adjusted Rand Score: ", cos_adj_rand)