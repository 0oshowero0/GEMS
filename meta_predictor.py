import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import torch.nn as nn
import setproctitle
import math
from scipy.stats import pearsonr,spearmanr
import matplotlib.pyplot as plt
setproctitle.setproctitle('GEMS@hanzhenyu-EXCLUSIVE-on-GPU5,6,7')


#定义模型
class MetaPred(torch.nn.Module):
    def __init__(self,EMBEDDING_SIZE, GENE_NUM,NODE_TYPE_NUM):
        super(MetaPred, self).__init__()
        self.id_embed = nn.Parameter(torch.Tensor(NODE_TYPE_NUM, EMBEDDING_SIZE))
        nn.init.kaiming_uniform_(self.id_embed, a=math.sqrt(5))
        #self.id_embed = nn.Embedding(NODE_TYPE_NUM, EMBEDDING_SIZE)
        self.conv1 = GCNConv(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.conv2 = GCNConv(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.W1 = nn.Linear(EMBEDDING_SIZE*GENE_NUM, EMBEDDING_SIZE)
        self.W2 = nn.Linear(EMBEDDING_SIZE, 1)

    def forward(self, data):
        #实际是graph number
        node_num = data[0].shape[0]
        x = [torch.tanh(self.id_embed[data[0][ind]]) for ind in range(node_num)]
        x = [torch.tanh(self.conv1(x[ind], data[1][ind])) for ind in range(node_num)]
        x = [F.dropout(tmp, p=0.4, training=self.training) for tmp in x]
        x = [torch.tanh(self.conv2(x[ind], data[1][ind])) for ind in range(node_num)]
        x = [F.dropout(tmp, p=0.2, training=self.training) for tmp in x]
        x = [torch.mean(x[ind], dim=0, keepdim=True) for ind in range(node_num)]

        x = torch.cat(x, dim=1)
        x = torch.tanh(self.W1(x))
        #x = self.W1(x)
        x = self.W2(x)

        return torch.squeeze(torch.tanh(x))



def predictor_data_init(pretrain_data_path = '/home/hanzhenyu/code_for_server/result_log/rl4/', new_data_path = '/home/hanzhenyu/code_for_server/result_log/', PATIENCE = 7, device = 'cuda:0'):
    typelist = ['SU', 'SB', 'U', 'B', 'O', 'I', 'A']
    NODE_TYPE_NUM = len(typelist)
    EMBEDDING_SIZE = 16
    type2inx = {typelist[ind]:ind for ind in range(len(typelist))}
    GENE_NUM = 5
    NODE_NUM = len(typelist)
    # 读取数据

    preformance_pre = np.load(pretrain_data_path + 'performance_of_genes.npy')[:, :, -1].reshape(-1, 1)
    preformance_new = np.load(new_data_path + 'performance_of_genes.npy')[:, :, -1].reshape(-1, 1)
    performance = np.concatenate((preformance_pre,preformance_new), axis=0)

    genes_pre = np.load(pretrain_data_path + 'best_genes.npy', allow_pickle=True).reshape(-1, 5,4)[:, :,0:2]
    genes_new = np.load(new_data_path+'best_genes.npy', allow_pickle=True).reshape(-1,5,4)[:, :, 0:2]
    genes = np.concatenate((genes_pre,genes_new),axis=0)
    for i in range(genes.shape[0]):
        for j in range(GENE_NUM):
            genes[i, j, 0][0] = 'SU'
            genes[i, j, 0][1] = 'SB'

    data = []
    for i in range(genes.shape[0]):
        name_list = [torch.tensor([type2inx[tmp] for tmp in single_gene_type]).to(device) for single_gene_type in
                     genes[i, :, 0]]
        edge_list = []
        for single_gene_edge in genes[i, :, 1]:
            tmp = np.where(np.array(single_gene_edge) > 0)
            edge_list_tmp = np.concatenate(
                (np.concatenate((tmp[0], tmp[1])).reshape(1, -1), (np.concatenate((tmp[1], tmp[0])).reshape(1, -1))),
                axis=0)
            edge_list.append(torch.tensor(edge_list_tmp).to(device))
        data.append([name_list, edge_list])

    data = np.array(data)
    train_index = np.random.random(len(data)) < 0.8
    train_data = data[train_index]
    val_data = data[~train_index]

    train_label = torch.from_numpy(performance[train_index]).to(device).type(torch.float32)
    val_label = torch.from_numpy(performance[~train_index]).to(device).type(torch.float32)

    avg = train_label.mean()
    std = train_label.std()

    train_label = (train_label - avg) / std
    val_label = (val_label - avg) / std

    model = MetaPred(EMBEDDING_SIZE, GENE_NUM, NODE_TYPE_NUM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    loss = torch.nn.MSELoss()

    early_stop = 0
    best_perf = 999
    for epoch in range(30):
        loss_sum = 0
        model.train()
        for ind in range(len(train_label)):
            optimizer.zero_grad()
            out = model(train_data[ind])
            output = loss(out, train_label[ind][0])
            output.backward()
            optimizer.step()
            loss_sum += output.detach().cpu().numpy()
        print('Epoch:{:d}, train MSE: {:.8f}'.format(epoch, loss_sum / len(train_label)))

        loss_sum = 0
        model.eval()
        for ind in range(len(val_label)):
            out = model(val_data[ind])
            output = loss(out, val_label[ind][0])
            loss_sum += output.detach().cpu().numpy()
        print('Epoch:{:d}, evaluation MSE: {:.8f}'.format(epoch, loss_sum / len(val_label)))

        if loss_sum < best_perf:
            early_stop = 0
            best_perf = loss_sum
        else:
            early_stop += 1

        if early_stop > PATIENCE:
            break

    return model, avg, std
