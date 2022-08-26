from ogb.graphproppred import PygGraphPropPredDataset
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import approximation as apxa
from scipy.sparse import csr_array
dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", root = 'dataset/')
cnt = 0

def add_vcc_data(graph):
    G = nx.Graph()
    for i in range(0, graph.num_nodes):
        G.add_node(i)
    for i in range(0, len(graph.edge_index[0]), 2):
        G.add_edge(int(graph.edge_index[0][i]), int(graph.edge_index[1][i]))

    g_decomp = apxa.k_components(G)
    neigh = np.zeros((graph.num_nodes, graph.num_nodes))
    for i in g_decomp:
        comps = g_decomp.get(i)
        for t in comps:
            for n1 in t:
                for n2 in t:
                    if n1 != n2:
                        neigh[n1][n2] += 1
    for i in range(len(neigh)):
        for j in range(len(neigh[i])):
            if neigh[i][j] <= 1:
                neigh[i][j] = 0
    ##graph.k_vcc_matrix = csr_array(neigh)
    graph.k_vcc_matrix = neigh.flatten()
    print(graph.num_nodes)
    print(graph.k_vcc_matrix)
    return graph

dataset.pre_transform = add_vcc_data

'''for g in dataset:
    cnt += 1
    G = nx.Graph()
    for i in range(0, g.num_nodes):
        G.add_node(i)
    for i in range(0, len(g.edge_index[0]), 2):
        G.add_edge(int(g.edge_index[0][i]), int(g.edge_index[1][i]))


    g_decomp = apxa.k_components(G)
    neigh = np.zeros((g.num_nodes, g.num_nodes))
    for i in g_decomp:
        comps = g_decomp.get(i)
        for t in comps:
            for n1 in t:
                for n2 in t:
                    if n1 != n2:
                        neigh[n1][n2] += 1
    for i in range(len(neigh)):
        for j in range(len(neigh[i])):
            if neigh[i][j] <= 1:
                neigh[i][j] = 0
    sparse_neigh = csr_array(neigh)'''