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
def add_vcc_data(graph):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = "cpu"
    #print(device)
    graph.original_cnt_edges = len(graph.edge_attr)
    graph.original_cnt_nodes = graph.num_nodes
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
    new_nodes_count = 0
    for i in g_decomp:
        new_nodes_count += len(g_decomp.get(i))

    neigh = np.zeros((MAX_K, graph.num_nodes + new_nodes_count, graph.num_nodes + new_nodes_count))

    new_nodes_count = 0

    #print(graph.edge_index)
    #print(graph.edge_attr)
    graph.edge_index = graph.edge_index.cuda()
    graph.edge_attr = graph.edge_attr.cuda()
    #print(graph.x)
    max_node = 0
    for i in range(len(graph.edge_index[0])):
        max_node = max(max_node, graph.edge_index[0][i])
    #exit()
    for i in g_decomp:
        for comp in g_decomp.get(i):
            idx = graph.num_nodes
            #print(idx)
            graph.x = torch.vstack((graph.x.cuda(), torch.tensor([[i]*9]).cuda())).cuda()
            graph.num_nodes += 1
            for v in comp:
                graph.edge_index = torch.hstack((graph.edge_index.cuda(), torch.tensor([[v, idx], [idx, v]], dtype=torch.long).cuda())).cuda()
                graph.edge_attr = torch.vstack((graph.edge_attr.cuda(), torch.tensor([[i, i, i], [i, i, i]], dtype=torch.long).cuda())).cuda()


            new_nodes_count += 1

    #print(graph.x)
    #print(graph.edge_index)
    #print(graph.num_nodes)
    #print(graph.edge_attr)
    #print('----------------')
    #print(g_decomp)
    #exit()
    #print(flag)
    #print(graph.edge_index)
    #print(g_decomp)
    #exit()
    graph.new_cnt_edges = len(graph.edge_attr) - graph.original_cnt_edges
    graph.new_cnt_nodes = new_nodes_count
    #print(graph.edge_index)
    #print(graph.edge_attr)
    max_node = 0
    for i in range(len(graph.edge_index[0])):
        max_node = max(max_node, graph.edge_index[0][i])

    '''for i in g_decomp:
        if i >= MAX_K:
            break
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
    '''
    # We need to transpose all data, because the matching size dimensions 
    # should be [1:], while dimension 0 can be of arbitrary size.
    # We keep a complete graph as the "new" edges, and we also store the
    # weight for each of the edges.
    '''graph.edge_weight = torch.flatten(torch.Tensor(neigh), start_dim=1).T
    seq0toN = torch.arange(graph.num_nodes, device=device)
    complete_edges = torch.cartesian_prod(seq0toN, seq0toN).T
    graph.k_vcc_edges = torch.transpose(
        complete_edges[None, :, :].repeat(MAX_K, 1, 1), 0, 2
    )'''
    return graph

def train(model, device, loader, optimizer, task_type):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = "cpu"
    #print(device)
    model.to(device)
    torch.backends.cudnn.benchmark = True

    model.train()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            #print('*********************')
            #print('eho?')
            pred = model(batch)
            #print('*********************')
            #print('eho1')
            optimizer.zero_grad()
            #print('*********************')
            #print('eho2')
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss = cls_criterion(pred.to(torch.float32).to(device)[is_labeled], batch.y.to(torch.float32).to(device)[is_labeled]).to(device)
            else:
                loss = reg_criterion(pred.to(torch.float32).to(device)[is_labeled], batch.y.to(torch.float32).to(device)[is_labeled]).to(device)

            loss.backward()

            optimizer.step()


def eval(model, device, loader, evaluator):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = "cpu"
    #print(device)

    model.to(device)
    torch.backends.cudnn.benchmark = True

    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch).to(device)

            y_true.append(batch.y.cuda().view(pred.shape).detach().to(device))
            y_pred.append(pred.detach().to(device))

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim = 0).cpu().numpy()

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
    parser.add_argument('--dataset', type=str, default="ogbg-molbace",
                        help='dataset name (default: ogbg-molbace)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')

    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    # device = "cpu"
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
        dataset.data.x = dataset.data.x[:,:2].cuda()
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2].cuda()

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
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []

    torch.backends.cudnn.benchmark = True
    print('*************************')
    print(device)

    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))
    #return
    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset.task_type)
        #return
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
        average_train = sum(train_curve)/len(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)
        average_train = average(train_curve)/len(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))
    #print(test_curve)

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)


if __name__ == "__main__":
    main()
