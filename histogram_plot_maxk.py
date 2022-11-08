import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN
#, GNN_FA, GNN_TYPE

import matplotlib.pyplot as plt

from tqdm import tqdm
import argparse
import time
import numpy as np
from networkx.algorithms import approximation as apxa
#from scipy.sparse import csr_array
import networkx as nx

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from config_vcc_gnn import MAX_K

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()
maxks = []
def add_vcc_data(graph):
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = "cpu"
    #print(device)
    G = nx.Graph()
    for i in range(0, graph.num_nodes):
        G.add_node(i)

    original_neigh = np.zeros((graph.num_nodes, graph.num_nodes))
    #graph.edge_index = graph.edge_index.cuda()
    for i in range(0, len(graph.edge_index[0]), 2):
        original_neigh[graph.edge_index[0][i]][graph.edge_index[1][i]] = 1
        original_neigh[graph.edge_index[1][i]][graph.edge_index[0][i]] = 1
        G.add_edge(int(graph.edge_index[0][i]), int(graph.edge_index[1][i]))

    g_decomp = apxa.k_components(G)
    #print(graph)
    #print(g_decomp)
    maxks.append(len(g_decomp)+1)
    return graph.cuda()

def main():
    dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", pre_transform=add_vcc_data)

    #print(maxks)
    plt.hist(maxks, bins=10)
    plt.title("ogbg-molhiv")
    plt.show()

    maxks.clear()
    dataset = PygGraphPropPredDataset(name="ogbg-moltoxcast", pre_transform=add_vcc_data)

    # print(maxks)
    plt.hist(maxks, bins=10)
    plt.title("ogbg-moltoxcast")
    plt.show()


    maxks.clear()
    dataset = PygGraphPropPredDataset(name="ogbg-moltox21", pre_transform=add_vcc_data)

    # print(maxks)
    plt.hist(maxks, bins=10)
    plt.title("ogbg-moltox21")
    plt.show()


    maxks.clear()
    dataset = PygGraphPropPredDataset(name="ogbg-molmuv", pre_transform=add_vcc_data)

    # print(maxks)
    plt.hist(maxks, bins=10)
    plt.title("ogbg-molmuv")
    plt.show()


if __name__ == "__main__":
    main()
