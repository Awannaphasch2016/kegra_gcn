from layers.graph import GraphConvolution
from utils import *


import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence

import networkx as nx

# Define parameters
DATASET = 'cora'
FILTER = 'localpool'  # 'chebyshev' # ????
MAX_DEGREE = 2  # maximum polynomial degree # ????
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 200
PATIENCE = 10  # early stopping patience


# def load_data(path="./data/cora/", dataset="cora"):
#     """Load citation network dataset (cora only for now)"""
#     print('Loading {} dataset...'.format(dataset))
#
#     idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
#
#     exit()
#     # size = 3 885 980 = number of publication * number of uniq word features
#
#     features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#     labels = encode_onehot(idx_features_labels[:, -1])
#
#     # build graph
#     idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
#     idx_map = {j: i for i, j in enumerate(idx)}
#
#     edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
#     edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
#                      dtype=np.int32).reshape(edges_unordered.shape)
#     #unweight edges
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                         shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
#
#     # print(len(list(adj.T > adj)))
#     # print(len(list(adj)))
#     # exit()
#
#     # print(adj.T.multiply(adj.T > adj))
#     # build symmetric adjacency matrix ?? I dont get it
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#
#
#     print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
#     return features.todense(), adj, labels


# Get data
import os
# x = os.listdir(os.path.join(os.getcwd(), "kegra\\data\\cora"))
# print(x)
# exit()
path = os.path.join(os.getcwd(), "data\\cora\\")
dataset="cora"

# X, A, y = load_data(path = path,dataset=DATASET)
# y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)
idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))

edges = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.dtype(str))
uniq_nodes = idx_features_labels[:,0]
labels_list = idx_features_labels[:,-1]
node_label_dict = {node:label for node, label in zip(uniq_nodes, labels_list)}

uniq_labels = set(labels_list)
color_dict = {label: i for i, label in enumerate(uniq_labels)}

edges_list = edges.tolist()

# uniq_edges = [ (l,r) for pair in edges_list for l,r in pair ]
uniq_edges = [ tuple(pair) for pair in edges_list]
# print(uniq_edges)
# exit()

# G =
# G = nx.dodecahedral_graph()
G = nx.Graph()
G.add_nodes_from(uniq_nodes)
G.add_edges_from(uniq_edges)

def get_subgraph_disconnected():
    disconnected_G = list(nx.connected_component_subgraphs(G))
    disconnected_G = [(disconnected_G[i], len(g)) for i, g in enumerate(disconnected_G)]

    from operator import itemgetter
    disconnected_G = sorted(disconnected_G, key = itemgetter(1), reverse = True)

    disconnected_G = [g for g,l in disconnected_G]
    return disconnected_G

sorted_connected_subgraph = get_subgraph_disconnected()
#get label of nodes in connected_subgraph
labels_connected_subgraph = [ [node_label_dict[node] for node in g ]for g in sorted_connected_subgraph ]

#create color_list
color_list = [color_dict[label] for label in labels_list] #get label of nodes in G
subgraph_color_list = [[color_dict[label] for label in g]for g in labels_connected_subgraph] #get label of nodes in connected subgraph

#setting legends
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
# For color mapping
import matplotlib.colors as colors
import matplotlib.cm as cmx

jet = cm = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=max(color_list))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

# Using a figure to use it as a parameter when calling nx.draw_networkx
f = plt.figure(1)
ax = f.add_subplot(1, 1, 1)

for value in set(color_list):
    ax.plot([0], [0], #change this to plot in the top right coner
            color=scalarMap.to_rgba(value),
            label=value)

# nx.draw(G , node_color=color_list) #original graph
nx.draw_networkx(sorted_connected_subgraph[0] , node_color= subgraph_color_list[0], cmap = jet, vmin=0, vmax=max(color_list), with_labels= False , ax=ax ) #subgraph with legends

# nx.draw(G, node_color=color_list, vmin=0, vmax=max(color_list), cmap=jet, ax=ax)
plt.axis('off')
plt.axis('off')
f.set_facecolor('w')

plt.legend(loc=1)

f.tight_layout()
plt.show()


###############################3
# draw
###############################3
# get adj .content(first col = uniq nodes  and .cited (edges)
# sorted by degree
# pick top 50 nodes with the highest degree, and draw edges connected

