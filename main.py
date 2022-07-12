# Spectral Clustering Algorithm for Community Detection
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

# -----------------------------------
# Adjacency matrix
Adj = nx.to_numpy_array(G)

# Diagonal matrix of node degrees
Diag = np.diag(Adj.sum(axis=1))

# -----------------------------------
# Computing graph laplacian
# Ref: https://www.frontiersin.org/articles/10.3389/fncom.2013.00189/full

Lprim=np.zeros(Adj.shape)

for i in nodes:
    for j in nodes:
        if i==j:
            Lprim[i,j]=1
        elif Adj[i,j]==1:
            Lprim[i,j]=-1/Diag[i,i]
        else:
            Lprim[i,j]=0

L=Lprim
# -------------------------------------
# Extracting eigen vectors and eigen values

eigenvalues, eigenvectors = np.linalg.eig(L)

# --------------------------------------------
# Considering the first and second eigen vectors

values=np.real(eigenvectors[:,[0,1]])

zero_eigen=sum(1 for i in eigenvalues if i < 0.0001)
kmeans = KMeans(n_clusters=max(2,zero_eigen), random_state=0).fit(values)
node_color = kmeans.labels_

plt.figure()
plt.title(f"Results of Community Clustering - Number of clusters: %d" % max(2,zero_eigen))
nx.draw(G, node_color=node_color)
plt.show()
