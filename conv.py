import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn import GINConv as NodeGINConv
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.utils import degree
import numpy as np
import math

### GIN convolution along the graph structure
'''class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        
        #   emb_dim (int): node embedding dimensionality
        

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr).cuda()
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding)).cuda()

        return out.cuda()

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out.cuda()
'''
class K_VCC_Conv(MessagePassing):

    def __init__(self, max_k, wrapped_conv_layer):
        super(K_VCC_Conv, self).__init__(aggr="add")

        self.wrapped_conv_layer = wrapped_conv_layer.cuda()
        self.max_k = max_k
        self.alpha = torch.nn.Parameter(torch.tensor(np.zeros(max_k)).cuda())

    def forward(self, old_embeddings, k_vcc_edges):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #device = 'cpu'
        #print(device)

        msgs_K_N_D = torch.zeros((self.max_k, len(old_embeddings), len(old_embeddings[0]))).to(device)
        for k in range(self.max_k):
            msgs_K_N_D[k, :, :] = self.wrapped_conv_layer(old_embeddings, k_vcc_edges[k].to(device)).to(device)
        alpha1 = torch.nn.Softmax(dim=0)(self.alpha).to(device)
        msgs_N_D_K = msgs_K_N_D.permute(1, 2, 0).to(device)
        new_embeddings1 = torch.sum((msgs_N_D_K.to(device)*alpha1.to(device)).to(device), dim=2).double().to(device)
        return new_embeddings1.to(device)

### GCN convolution along the graph structure
'''class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

'''
### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, maxk, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''
        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.maxk = maxk
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #device = 'cpu'
        #print(device)


        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim).cuda()

        ###List of GNNs
        self.fa_conv = NodeGINConv(nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim)), train_eps=True).to(device)
        self.convs = torch.nn.ModuleList().to(device)
        self.batch_norms = torch.nn.ModuleList().to(device)

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(K_VCC_Conv(maxk, NodeGINConv(nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim).to(device), torch.nn.BatchNorm1d(2*emb_dim).to(device), torch.nn.ReLU().to(device), torch.nn.Linear(2*emb_dim, emb_dim).to(device)), train_eps=True).to(device)).to(device))
            '''elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))'''

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim).to(device))
        self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim).to(device))
        #self.convs.append(GINConv(emb_dim).cuda())
    def forward(self, batched_data):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #device = 'cpu'
        #print(device)

        '''print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')'''
        x, edge_index, edge_attr, batch = batched_data.x.to(device), batched_data.edge_index.to(device), batched_data.edge_attr.to(device), batched_data.batch.to(device)
        k_vcc_edges_shape = batched_data.k_vcc_edges_shape
        ptr = batched_data.ptr.to(device)
        batched_k_vcc_edges = batched_data.k_vcc_edges
        #k_vcc_edges = [np.array([[], []]) for i in range(self.maxk)]
        #k_vcc_edges2 = [[[], []] for i in range(self.maxk)]
        h_list = [self.atom_encoder(x).to(device)]
        curr_l = 0
        j = 0
        rem = sum(k_vcc_edges_shape[0])
        for i in range(len(batched_k_vcc_edges)):
            if rem == 0:
                j += 1
                rem = sum(k_vcc_edges_shape[j])
            batched_k_vcc_edges[i] += ptr[j].cuda()
        edges_K_BS = [[torch.tensor([[], []], dtype=torch.long, device=torch.device('cuda'))]*batched_data.num_graphs for i in range(self.maxk)]
        for i in range(len(k_vcc_edges_shape)):
            c_shape = k_vcc_edges_shape[i]
            k = 0
            for shape1 in c_shape:
                curr_r = curr_l + shape1
                k += 1
                #print(k, i)
                #print(len(edges_K_BS))
                #print(len(edges_K_BS[k]))
                edges_K_BS[k][i] = batched_k_vcc_edges[curr_l:curr_r].reshape(2, int(shape1/2)).cuda()
                #k_vcc_edges2[k][0].append(batched_k_vcc_edges[curr_l:curr_r].reshape(2, int(shape1/2))[0])
                #k_vcc_edges2[k][1].append(batched_k_vcc_edges[curr_l:curr_r].reshape(2, int(shape1/2))[1])
                curr_l = curr_r
        #for i in range(len(k_vcc_edges2)):
        #    print(len(k_vcc_edges2[i][0]))

        #print([x for x in k_vcc_edges2[0][0]])
        #print([x for x in k_vcc_edges2[1][0]])
        '''k_vcc_edges = []
        for i in range(len(k_vcc_edges2)):
            if len(k_vcc_edges2[i][0]) == 0:
                k_vcc_edges.append(torch.tensor([[], []], dtype=torch.long).cuda())
            else:
                k_vcc_edges.append(torch.stack((torch.tensor(np.concatenate([x for x in k_vcc_edges2[i][0]]), dtype=torch.long), torch.tensor(np.concatenate([x for x in k_vcc_edges2[i][1]]), dtype=torch.long)), dim=0).cuda())
        '''

        #k_vcc_edges = [torch.stack((torch.tensor(np.concatenate([x for x in k_vcc_edges2[i][0]])), torch.tensor(np.concatenate([x for x in k_vcc_edges2[i][1]]))), dim=0) for i in range(1, 3)]
        #print(k_vcc_edges2)
        #print(k_vcc_edges)
        #return
            #print(c_shape)

        '''for j in range(len(batched_data.k_vcc_edges_shape)):
            shape1 = batched_data.k_vcc_edges_shape[j]
            for k in range(len(shape1)):
                shape2 = shape1[k]
                #for i in range(shape2):
                #    batched_data.k_vcc_edges[last+i+1] += offset
                edges1 = np.array([batched_data.k_vcc_edges[int(last+i+1)].cpu() for i in range(int(shape2))]).reshape((2, int(shape2/2)))
                k_vcc_edges[k+1] = np.array([np.concatenate((k_vcc_edges[k+1][0], edges1[0]), axis=None), np.concatenate((k_vcc_edges[k+1][1], edges1[1]), axis=None)])
                last += shape2
            offset = batched_data.ptr[j+1]
        '''
        #print('&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        #print(k_vcc_edges1[1][0])
        #print(k_vcc_edges1[0][0])
        #print(len(k_vcc_edges1[1][0]))
        #print(len(k_vcc_edges1[1][1]))
        #print([np.array(k_vcc_edges1[i], dtype=np.float) for i in range(self.maxk)])
        #print('&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        #print(k_vcc_edges)
        #return

        k_vcc_edges = [torch.hstack([edges_K_BS[k][i] for i in range(batched_data.num_graphs)]) for k in range(self.maxk)]
        #print(k_vcc_edges)
        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], k_vcc_edges).float().to(device)

            h = self.batch_norms[layer](h).to(device)

            h = F.dropout(F.relu(h), self.drop_ratio, training=self.training).to(device)

            if self.residual:
                h += h_list[layer].to(device)

            h_list.append(h)

            if self.residual:
                h += h_list[self.num_layer-1]
                h_list.append(h)
        node_representation = 0

        if self.JK == "last":
            node_representation = h_list[-1].to(device)
        elif self.JK == "sum":
            for layer in range(self.num_layer):
                node_representation += h_list[layer].to(device)


        return node_representation.cuda()


### Virtual GNN to generate node embedding
'''class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        
        #    emb_dim (int): node embedding dimensionality
        

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
    pass'''
