# Clustering Algorithm for Community Detection
# Using Graph Laplacian Eigenvectors
# Code writer: Nooshin Bahador

# -----------------------------------
# Import libraries
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# -----------------------------------
# Generating random graph

G = nx.random_geometric_graph(50, 0.4)
nodes=G.nodes
edges=G.edges

G.add_nodes_from(nodes)
G.add_edges_from(edges)


# -----------------------------------
# Adjacency matrix
A = nx.to_numpy_array(G)

# Diagonal matrix of node degrees
D = np.diag(A.sum(axis=1))

# -----------------------------------
# Computing graph laplacian
# Ref: https://www.frontiersin.org/articles/10.3389/fncom.2013.00189/full

Lprim=np.zeros(A.shape)
print(Lprim)
for i in nodes:
    for j in nodes:
        if i==j:
            Lprim[i,j]=1
        elif A[i,j]==1:
            Lprim[i,j]=-1/D[i,i]
        else:
            Lprim[i,j]=0

L=Lprim
# -------------------------------------
# Extracting eigen vectors and eigen values

vals,vects = np.linalg.eig(L)
p = vects

# --------------------------------------------
# Considering the first and second eigen vectors

values=np.real(p[:,[0,1]])

zero_eigen=sum(1 for i in vals if i < 0.0001)
print(zero_eigen)
kmeans = KMeans(n_clusters=max(2,zero_eigen), random_state=0).fit(values)
node_color = kmeans.labels_

plt.figure()
plt.title(f"Results of Community Clustering - Number of clusters: %d" % max(2,zero_eigen))
nx.draw(G, node_color=node_color)
plt.show()
