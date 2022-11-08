import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from encoders import EdgeEncoder, NodeEncoder
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv as NodeGINConv
import numpy as np

from config_vcc_gnn import MAX_K
import math

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim)).cuda()
        self.eps = torch.nn.Parameter(torch.Tensor([0])).cuda()

        self.emb_dim = emb_dim

        self.bond_encoder = BondEncoder(emb_dim=emb_dim).cuda()

        self.edge_encoder = EdgeEncoder(emb_dim=emb_dim).cuda()

    def forward(self, x, edge_index, edge_attr, is_fa):
        #edge_embedding = torch.zeros(len(edge_attr), self.emb_dim).cuda()
        #edge_embedding[mask1] = self.bond_encoder(edge_attr[mask1])
        #edge_embedding[mask2] = self.edge_encoder(edge_attr[mask2])
        if is_fa:
            edge_embedding = self.edge_encoder(edge_attr)
        else:
            edge_embedding = self.bond_encoder(edge_attr)
        #print(edge_attr)
        #print(edge_embedding)
        out = self.mlp(((torch.tensor([1]).cuda() + self.eps) * x).cuda() + self.propagate(edge_index, x=x, edge_attr=edge_embedding).cuda()).cuda()


        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j.cuda() + edge_attr.cuda()).cuda()

    def update(self, aggr_out):
        return aggr_out.cuda()

### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim).cuda()
        self.edge_encoder = EdgeEncoder(emb_dim).cuda()
        #self.node_encoder = NodeEncoder(emb_dim)

        self.emb_dim = emb_dim

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim).cuda())
            elif gnn_type == 'gcn':
                self.convs.append(GaCNConv(emb_dim).cuda())
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim).cuda())
        self.kfa_conv = GINConv(emb_dim).cuda()
        self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim).cuda())

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x.cuda(), batched_data.edge_index.cuda(), batched_data.edge_attr.cuda(), batched_data.batch.cuda()
        fa_edge_index_list = batched_data.fa_edge_index
        fa_edge_index = torch.hstack(fa_edge_index_list).cuda()
        #print(fa_edge_index)
        fa_edge_attr_list = []
        for i in range(len(fa_edge_index_list)):
            fa_edge_attr_list.append(torch.full((len(fa_edge_index_list[i][0]), 3), i))
        fa_edge_attr = torch.vstack(fa_edge_attr_list).cuda()
        #print(fa_edge_attr)
        #print(edge_attr)
        ### computing input node embedding
        h_list = [self.atom_encoder(x)]
        
        for layer in range(self.num_layer):

            if layer == self.num_layer - 1:
                h = self.kfa_conv(h_list[layer], fa_edge_index, fa_edge_attr, True)
                h = self.batch_norms[layer](h)
                h = F.dropout(h, self.drop_ratio, training=self.training)
                if self.residual:
                    h += h_list[layer]
                h_list.append(h)
            else:
                h = self.convs[layer](h_list[layer], edge_index, edge_attr, False)
                h = self.batch_norms[layer](h)

                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

                if self.residual:
                    h += h_list[layer]

                h_list.append(h)



        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))


    def forward(self, batched_data):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


if __name__ == "__main__":
    pass