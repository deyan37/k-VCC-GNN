import torch
class NodeEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(NodeEncoder, self).__init__()

        self.node_embedding_list = torch.nn.ModuleList().cuda()

        for i in range(21):
            emb = torch.nn.Embedding(2000, emb_dim)
            #torch.nn.init.xavier_uniform_(emb.weight.data)
            self.node_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        #print(len(x[0]))
        #print(x.shape[1], len(self.node_embedding_list))
        for i in range(x.shape[1]):
            #print(torch.max(x[:, i]))
            #print(torch.min(x[:, i]))
            #print(i, len(self.node_embedding_list))
            #print(x_embedding, self.node_embedding_list[i](x[:, i]))
            x_embedding += self.node_embedding_list[i](x[:, i])

        return x_embedding


class EdgeEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(EdgeEncoder, self).__init__()

        self.edge_embedding_list = torch.nn.ModuleList().cuda()

        for i in range(3):
            emb = torch.nn.Embedding(20, emb_dim)
            #torch.nn.init.xavier_uniform_(emb.weight.data)
            self.edge_embedding_list.append(emb)

    def forward(self, edge_attr):
        edge_embedding = 0
        for i in range(edge_attr.shape[1]):
            edge_embedding += self.edge_embedding_list[i](edge_attr[:, i])

        return edge_embedding
