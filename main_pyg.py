import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN, GNN_FA, GNN_TYPE

from tqdm import tqdm
import argparse
import time
import numpy as np
from networkx.algorithms import approximation as apxa
#from scipy.sparse import csr_array
import networkx as nx

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()
def add_vcc_data(graph):
    G = nx.Graph()
    for i in range(0, graph.num_nodes):
        G.add_node(i)

    original_neigh = np.zeros((graph.num_nodes, graph.num_nodes))
    for i in range(0, len(graph.edge_index[0]), 2):
        original_neigh[graph.edge_index[0][i]][graph.edge_index[1][i]] = 1
        original_neigh[graph.edge_index[1][i]][graph.edge_index[0][i]] = 1
        G.add_edge(int(graph.edge_index[0][i]), int(graph.edge_index[1][i]))

    g_decomp = apxa.k_components(G)
    neigh = np.zeros((graph.num_nodes, graph.num_nodes, graph.num_nodes))
    for i in g_decomp:
        comps = g_decomp.get(i)
        in_comps = list([])
        for node in range(graph.num_nodes):
            ic = list([])
            for comp in range(len(comps)):
                if node in comps[comp]:
                    ic.append(comp)
            in_comps.append(ic)
        for n1 in range(graph.num_nodes):
            for nb in range(graph.num_nodes):
                if original_neigh[n1][nb] == 0:
                    continue
                for c in in_comps[nb]:
                    for n2 in comps[c]:
                        if n1 == n2:
                            continue
                        neigh[i][n1][n2] += 1
        #print(neigh)
        '''
        for t in comps:
            for n1 in t:
                for n2 in t:
                    if n1 != n2:
                        neigh[i][n1][n2] += 1
        '''
    k_vcc_matrix = [neigh[i+1] for i in range(0, len(g_decomp))]
    k_vcc_edges = np.array([])
    k_vcc_edges_shape = []
    #maxk
    #print('+++++++++++++++')
    #print(k_vcc_matrix)
    for i in range(3):
        ids1 = []
        ids2 = []
        if len(k_vcc_matrix) <= i:
            continue
        for j in range(0, len(k_vcc_matrix[i])):
            for l in range(j + 1, len(k_vcc_matrix[i][j])):
                for cnt in range(int(k_vcc_matrix[i][j][l])):
                    ids1.append(j)
                    ids2.append(l)
        t = len(k_vcc_edges)
        #print(len(ids1))
        k_vcc_edges = np.append(k_vcc_edges, np.array([np.concatenate((ids1, ids2)), np.concatenate((ids2, ids1))]).flatten())
        k_vcc_edges_shape.append(len(k_vcc_edges)-t)
    #graph.k_vcc_edges = [k_vcc_edges[i] for i in range(len(k_vcc_edges))]
    #print(k_vcc_edges)
    graph.k_vcc_edges = torch.tensor(np.array(k_vcc_edges))
    graph.k_vcc_edges_shape = k_vcc_edges_shape
    #print(graph.k_vcc_edges)
    #print(graph.k_vcc_edges_shape)
    '''#print(graph.k_vcc_edges)
    #print(graph.k_vcc_edges_shape)
    '''
    graph = graph.cuda()
    return graph

def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        batch = batch.cuda()
        model.cuda()
        batch.x = batch.x.cuda()
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            #print('*********************')
            #print('eho?')
            pred = model(batch).cuda()
            #print('*********************')
            #print('eho1')
            optimizer.zero_grad()
            #print('*********************')
            #print('eho2')
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])

            #print('*********************')
            #print('eho3')
            #print(model)
            #print(batch)
            loss.backward()
            #print('*********************')
            #print('eho4')
            optimizer.step()
            #print('*********************')
            #print('eho5')

def eval(model, device, loader, evaluator):
    model.cuda()
    model.eval()
    y_true = []
    y_pred = []
    model.cuda()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cuda())
            y_pred.append(pred.detach().cuda())

    y_true = torch.cat(y_true, dim = 0).cuda().numpy()
    y_pred = torch.cat(y_pred, dim = 0).cuda().numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    # gnn_type, num_layers, dim0, h_dim, out_dim, last_layer_fully_adjacent,
    #                  unroll, layer_norm, use_activation, use_residual
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='number of GNN message passing layers (default: 2)')
    parser.add_argument('--emb_dim', type=int, default=32,
                        help='dimensionality of hidden units in GNNs (default: 32)')
    parser.add_argument('--dim0', type=int, default=300,
                        help='')
    parser.add_argument('--h_dim', type=int, default=300,
                        help='')
    parser.add_argument('--out_dim', type=int, default=300,
                        help='')
    parser.add_argument('--last_layer_fully_adjacent', type=int, default=1,
                        help='')
    parser.add_argument('--unroll', type=int, default=0,
                        help='')
    parser.add_argument('--layer_norm', type=int, default=1,
                        help='')
    parser.add_argument('--use_activation', type=int, default=1,
                        help='')
    parser.add_argument('--use_residual', type=int, default=1,
                        help='')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')

    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    print('eho1')
    dataset = PygGraphPropPredDataset(name = args.dataset, pre_transform=add_vcc_data)
    print('eho2')
    print(torch.cuda.is_available())
    #dataset.pre_transform = add_vcc_data

    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    '''print(device)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    return'''
    #print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    #print(args.num_layer)
    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', maxk=5, num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', maxk=5, num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', maxk=5, num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', maxk=5, num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []

    torch.backends.cudnn.benchmark = True
    print('*************************')
    print(device)

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset.task_type)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)


if __name__ == "__main__":
    main()
