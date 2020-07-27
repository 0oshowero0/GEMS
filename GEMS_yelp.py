import os
from math import pow
import numpy as np
from numpy.random import random, randint, choice
import pandas as pd
import h5py
import pickle
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data as torch_data 
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.multiprocessing import Process, Pool, Manager, get_context, freeze_support

#import torchnlp.nn as nlp_nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import dropout_adj
import time
from datetime import datetime
import setproctitle
import GPUtil
import matplotlib.pyplot as plt

#for geneatic search
from searching.searchGraph import *
from searching.mutation import *
from searching.process_data import construct_adj
from searching.transfer import *

#########################################################################
setproctitle.setproctitle('GEMS')

# IMPORTANT
# setup multi-process training according to the capacity of your server!
MULTI_PROCESS_NUM_SEARCH = 60
MULTI_PROCESS_NUM_TRAIN = 8
EPOCH_NUM = 200
BATCH_SIZE = 512
NEG_SIZE_TRAIN = 4
NEG_SIZE_RANKING = 100
ID_EMBEDDING_SIZE = 64
EMBEDDING_SIZE = 32
POPULATION_SIZE = 20  # How many population in a generation
GENE_NUM = 5
GENE_POOL_SIZE = POPULATION_SIZE * GENE_NUM
GENERATION = 100

SAMPLE_SIZE = 200
ELIMINATE_RATE = 0.4
CROSS_OVER_RATE = 0.05
INIT_STABLE_PROB = 0.4
STABLE_PROB = 0.6
INIT_COMPLEX_PROB = 0.6
COMPLEX_PROB = 0.5
ADD_NODE_PROB = 0.2

PRE_MUTATE = 0
ANNEALING_EPOCH = 3

LEARNING_RATE = 0.06
LAMBDA = 0.01
EARLY_STOP = 6
WARM_UP_STEP = 20
SAVE_ADJ_EDGES = False
LOSS_MARGIN = 0.3
TRAIN_EVAL = False

DEBUG = False
LOG_DIR = './result_log/yelp/'
TIME_LOG = LOG_DIR + 'time.txt'
MF_user = './MF_pretrain/MF_userEmb64.npy'
MF_item = './MF_pretrain/MF_itemEmb64.npy'

class AliasTable():
    def __init__(self, weights, keys):
        self.keys = keys
        self.keyLen = len(keys)
        weights = weights * self.keyLen / weights.sum()

        inx = -np.ones(self.keyLen, dtype=int)
        shortPool = np.where(weights < 1)[0].tolist()
        longPool = np.where(weights > 1)[0].tolist()
        while shortPool and longPool:
            j = shortPool.pop()
            k = longPool[-1]
            inx[j] = k
            weights[k] -= (1 - weights[j])
            if weights[k] < 1:
                shortPool.append( k )
                longPool.pop()

        self.prob = weights
        self.inx = inx

    def draw(self, count=None):
        u = np.random.random(count)
        j = randint(self.keyLen, size=count)
        k = np.where(u <= self.prob[j], j, self.inx[j])
        return torch.from_numpy(self.keys[k]).type(torch.LongTensor)


##############################################################################################################

def multi_process_search_adj(sub_process_num, result_dict, mg_adj_list, id_map_list, history_dict, what_the_dict):
    start = time.perf_counter()
    new_adj = np.zeros((16239, 14284))
    #if history_dict.__contains__(str(id_map_list[sub_process_num])):
    #    result_dict[sub_process_num] = history_dict[str(id_map_list[sub_process_num])]
    #else:
    #    _, adj_pair = match_graph(new_adj, mg_adj_list[i], id_map_list[sub_process_num], what_the_dict)
    #    history_dict[str(id_map_list[i])] = adj_pair
    #    result_dict[sub_process_num] = adj_pair
    print("================================================")
    print(str(sub_process_num) + ' started, at ' +str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+', to search \n' + str(id_map_list[sub_process_num])+ '\n' + str(mg_adj_list[sub_process_num]))

    _, adj_pair, _ = match_graph(new_adj, mg_adj_list[sub_process_num], id_map_list[sub_process_num], what_the_dict, SAMPLE_SIZE)
    
    result_dict[sub_process_num] = adj_pair
    elapsed = (time.perf_counter() - start)
    print(str(sub_process_num) + ' ended')
    print("all user search done in " + str(elapsed))

def ecb_search(x):
    print('****************************************************************************************')
    print('SOMETHING WRONG!!!!!!!!!!\n'+'In search subprocess ' + str(x))
    print('****************************************************************************************')

def mutate_and_search(gen_num, input_genes, history_dict, what_the_dict, stable_prob, complex_prob, add_node_prob):
    input_genes_cp = copy.deepcopy(input_genes)
    manager = Manager()

    mg_adj_list = []
    id_map_list = []
    time_list = []
    old_path_len = []
    if gen_num > 0:
        genes = [mutate_graph(candidate, stable_prob, complex_prob, add_node_prob) for candidate in input_genes_cp]
    else:
        genes = input_genes_cp
    for i in range(len(genes)):
        if genes[i] is None:
            print('****************************************************************************************')
            print('Mutate Failure, change it into initial state\n')
            print('****************************************************************************************')
            genes[i] = mutate_graph(mutation_init())

    for candidate in genes:
        mg_adj_list.append(candidate[1])
        id_map_list.append(candidate[0])
        time_list.append(candidate[2])
        old_path_len.append(candidate[3])

    result_dict = manager.dict()
    p = Pool(MULTI_PROCESS_NUM_SEARCH)

    result = [p.apply_async(multi_process_search_adj, args=(i, result_dict, mg_adj_list, id_map_list, history_dict, what_the_dict),error_callback=ecb_search) for i in
              range(len(id_map_list))]
    for i in result:
        i.get()

    p.close()
    p.join()

    #Search Adj
    ok_index = []
    no_adj_index = []
    for key in result_dict.keys():
        if len(result_dict[key]) > 0:
            ok_index.append(key)
        else:
            no_adj_index.append(key)

    no_adj_count = len(no_adj_index)
    for key in no_adj_index:
            np.save('./error_genes_results/gen'+str(gen_num)+'_id_map_list_'+str(key)+'.npy', id_map_list[key])
            np.save('./error_genes_results/gen'+str(gen_num)+'_mg_adj_list_'+str(key)+'.npy', mg_adj_list[key])
            replace_index = int(choice(ok_index,1))
            print('****************************************************************************************')
            print('NO ADJACENCY FOR META-GRAPH\n' + str(id_map_list[key]) + '\nIn gen'+str(gen_num) + ', gene' + str(key)+'\n'+str(mg_adj_list[key]))
            print('Using ' + str(replace_index) + ' for replacement:')
            mg_adj_list[key] = mg_adj_list[replace_index]
            id_map_list[key] = id_map_list[replace_index]
            result_dict[key] = result_dict[replace_index]
            time_list[key] = time_list[replace_index]
            old_path_len[key] = old_path_len[replace_index]
            print('After replacement:\n'+ str(id_map_list[key])+'\n'+str(mg_adj_list[key]))
            print(str(result_dict[key].shape))
            print('****************************************************************************************')



    populations_genes = []
    new_gene_pools = []


    for i in range(POPULATION_SIZE):

        if np.random.random(1) < CROSS_OVER_RATE:
            pre_index = randint(GENE_NUM*i,GENE_NUM*(i+1))
            crossed_index = randint(0, GENE_POOL_SIZE)
            temp = result_dict[pre_index]
            result_dict[pre_index] = result_dict[crossed_index]
            result_dict[crossed_index] = temp
        gene_combines = []

        for j in range(GENE_NUM*i,GENE_NUM*(i+1)):
            gene_combines.append(result_dict[j])
            new_gene_pools.append([id_map_list[j], mg_adj_list[j], time_list[j], old_path_len[j]])
        populations_genes.append(gene_combines)

    populations_genes = np.array(populations_genes)
    print('===============================================================================================================')
    print('Gen'+str(gen_num) + ' Search Graph Complete!')
    print('There are ' + str(no_adj_count) + ' meta-graphs that do not valid in this graph.')
    print('===============================================================================================================')
    return populations_genes, new_gene_pools, dict(history_dict), no_adj_count




def multi_process_eval_model(gen_num, i, genes_dict, result_dict):

    print('Generation '+str(gen_num) + ', Sub Process' + str(i) + ' for GPU Training Begins!')

    # IMPORTANT
    # set the GPU allocation according to your server!
    if i < MULTI_PROCESS_NUM_TRAIN:
        if i in [0, 4, 8, 12, 16, 20, 24]:
            device = 'cuda:0'
        elif i in [1, 5, 9, 13, 17, 21, 25]:
            device = 'cuda:1'
        elif i in [2, 6, 10, 14, 18, 22]:
            device = 'cuda:2'
        else:
            device = 'cuda:3'
    else:
        tmp = GPUtil.getAvailable(order='load', limit=1, maxLoad=0.7)
        while len(tmp) < 1:
            tmp = GPUtil.getAvailable(order='load', limit=1, maxLoad=0.7)
        device = 'cuda:' + str(tmp[0])

    
    # Network
    class Net(torch.nn.Module):
        def __init__(self, node_num, pre_train_emb):
            super(Net, self).__init__()

            # self.x = nn.Parameter(torch.Tensor(node_num, ID_EMBEDDING_SIZE))
            # nn.init.kaiming_uniform_(self.x, a=math.sqrt(5))
            self.x = pre_train_emb

            # nn.init.normal_(self.x)
            # self.W_x = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)

            #self.conv1 = GATConv(ID_EMBEDDING_SIZE, EMBEDDING_SIZE)
            self.conv1 = SAGEConv(ID_EMBEDDING_SIZE, EMBEDDING_SIZE)
            self.W1 = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)
            self.W2 = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)

            #self.conv2 = GATConv(ID_EMBEDDING_SIZE, EMBEDDING_SIZE)
            self.conv2 = SAGEConv(ID_EMBEDDING_SIZE, EMBEDDING_SIZE)
            self.W3 = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)
            self.W4 = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)

            #self.conv3 = GATConv(ID_EMBEDDING_SIZE, EMBEDDING_SIZE)
            self.conv3 = SAGEConv(ID_EMBEDDING_SIZE, EMBEDDING_SIZE)
            self.W5 = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)
            self.W6 = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)

            #self.conv4 = GATConv(ID_EMBEDDING_SIZE, EMBEDDING_SIZE)
            self.conv4 = SAGEConv(ID_EMBEDDING_SIZE, EMBEDDING_SIZE)
            self.W7 = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)
            self.W8 = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)

            #self.conv5 = GATConv(ID_EMBEDDING_SIZE, EMBEDDING_SIZE)
            self.conv5 = SAGEConv(ID_EMBEDDING_SIZE, EMBEDDING_SIZE)
            self.W9 = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)
            self.W10 = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)

            # for attention
            self.W_u_att = nn.Linear(EMBEDDING_SIZE * GENE_NUM, EMBEDDING_SIZE)
            self.W_i_att = nn.Linear(EMBEDDING_SIZE * GENE_NUM, EMBEDDING_SIZE)

            # using other attention implementation
            # you need to uncomment line 16
            # self.user_attention = nlp_nn.Attention(EMBEDDING_SIZE)
            # self.item_attention = nlp_nn.Attention(EMBEDDING_SIZE)

        def forward(self, pos_pairs, neg_items, edges0, edges1, edges2, edges3, edges4, DEBUG=False):

            x = self.x
            x1 = torch.tanh(self.conv1(x, edges0))
            x1 = F.dropout(x1,p=0.2,training=self.training)
            user1 = torch.tanh(self.W1(x1[pos_pairs[:, 0]]))
            item_pos1 = torch.tanh(self.W2(x1[pos_pairs[:, 1]]))
            item_neg1 = torch.tanh(self.W2(x1[neg_items]))

            x2 = torch.tanh(self.conv2(x, edges1))
            x2 = F.dropout(x2,p=0.2,training=self.training)
            user2 = torch.tanh(self.W3(x2[pos_pairs[:, 0]]))
            item_pos2 = torch.tanh(self.W4(x2[pos_pairs[:, 1]]))
            item_neg2 = torch.tanh(self.W4(x2[neg_items]))

            x3 = torch.tanh(self.conv3(x, edges2))
            x3 = F.dropout(x3,p=0.2,training=self.training)
            user3 = torch.tanh(self.W5(x3[pos_pairs[:, 0]]))
            item_pos3 = torch.tanh(self.W6(x3[pos_pairs[:, 1]]))
            item_neg3 = torch.tanh(self.W6(x3[neg_items]))

            x4 = torch.tanh(self.conv4(x, edges3))
            x4 = F.dropout(x4,p=0.2,training=self.training)
            user4 = torch.tanh(self.W7(x4[pos_pairs[:, 0]]))
            item_pos4 = torch.tanh(self.W8(x4[pos_pairs[:, 1]]))
            item_neg4 = torch.tanh(self.W8(x4[neg_items]))

            x5 = torch.tanh(self.conv5(x, edges4))
            x5 = F.dropout(x5,p=0.2,training=self.training)
            user5 = torch.tanh(self.W9(x5[pos_pairs[:, 0]]))
            item_pos5 = torch.tanh(self.W10(x5[pos_pairs[:, 1]]))
            item_neg5 = torch.tanh(self.W10(x5[neg_items]))

            user_array_cat = torch.cat((user1, user2, user3, user4, user5), dim=1)
            item_pos_array_cat = torch.cat((item_pos1, item_pos2, item_pos3, item_pos4, item_pos5), dim=1)
            item_neg_array_cat = torch.cat((item_neg1, item_neg2, item_neg3, item_neg4, item_neg5), dim=1)
            user_logits_quary = torch.tanh(torch.unsqueeze(self.W_u_att(user_array_cat), 1))
            item_pos_logits_quary = torch.tanh(torch.unsqueeze(self.W_i_att(item_pos_array_cat), 1))
            item_neg_logits_quary = torch.tanh(torch.unsqueeze(self.W_i_att(item_neg_array_cat), 1))

            user_array_cat = user_array_cat.view(pos_pairs.shape[0], GENE_NUM, -1)
            item_pos_array_cat = item_pos_array_cat.view(pos_pairs.shape[0], GENE_NUM, -1)
            item_neg_array_cat = item_neg_array_cat.view(neg_items.shape[0], GENE_NUM, -1)

            user_pre_softmax = torch.sum(user_logits_quary * user_array_cat, dim=2)
            user_attention = F.softmax(user_pre_softmax, dim=1)

            item_pre_pos_softmax = torch.sum(item_pos_logits_quary * item_pos_array_cat, dim=2)
            item_pos_attention = F.softmax(item_pre_pos_softmax, dim=1)

            item_pre_neg_softmax = torch.sum(item_neg_logits_quary * item_neg_array_cat, dim=2)
            item_neg_attention = F.softmax(item_pre_neg_softmax, dim=1)

            user_fin_embed_pre = torch.unsqueeze(user_attention, 2) * user_array_cat
            item_fin_embed_pos_pre = torch.unsqueeze(item_pos_attention, 2) * item_pos_array_cat
            item_fin_embed_neg_pre = torch.unsqueeze(item_neg_attention, 2) * item_neg_array_cat

            user_fin_embed = torch.sum(user_fin_embed_pre, dim=1)
            item_pos_fin_embed = torch.sum(item_fin_embed_pos_pre, dim=1)
            item_neg_fin_embed = torch.sum(item_fin_embed_neg_pre, dim=1)

            pos_result = torch.sigmoid(torch.sum(torch.mul(user_fin_embed, item_pos_fin_embed), dim=1, keepdim=True))

            neg_result = torch.sigmoid(torch.sum(torch.mul(torch.unsqueeze(user_fin_embed, dim=1),
                        item_neg_fin_embed.view(user_fin_embed.shape[0], -1,EMBEDDING_SIZE)).view(-1,EMBEDDING_SIZE),dim=1, keepdim=True))

            if not DEBUG:
                return pos_result, neg_result
            if DEBUG:
                return pos_result.detach().to('cpu').numpy(), neg_result.detach().to('cpu').numpy(), [
                    x1.detach().to('cpu').numpy(), x2.detach().to('cpu').numpy(), x3.detach().to('cpu').numpy(),
                    x4.detach().to('cpu').numpy(), x5.detach().to('cpu').numpy()], \
                       [user1.detach().to('cpu').numpy(), user2.detach().to('cpu').numpy(),
                        user3.detach().to('cpu').numpy(), user4.detach().to('cpu').numpy(),
                        user5.detach().to('cpu').numpy()], \
                       [item_pos1.detach().to('cpu').numpy(), item_pos2.detach().to('cpu').numpy(),
                        item_pos3.detach().to('cpu').numpy(), item_pos4.detach().to('cpu').numpy(),
                        item_pos5.detach().to('cpu').numpy()], \
                       [item_neg1.detach().to('cpu').numpy(), item_neg2.detach().to('cpu').numpy(),
                        item_neg3.detach().to('cpu').numpy(), item_neg4.detach().to('cpu').numpy(),
                        item_neg5.detach().to('cpu').numpy()], \
                       user_fin_embed.detach().to('cpu').numpy(), item_pos_fin_embed.detach().to(
                    'cpu').numpy(), item_neg_fin_embed.detach().to('cpu').numpy(), user_attention.detach().to(
                    'cpu').numpy(), item_pos_attention.detach().to('cpu').numpy(), item_neg_attention.detach().to(
                    'cpu').numpy()

        def get_model_embedding(self):
            return self.x.detach().to('cpu').numpy()

        def get_gcn_embedding(self, edges0, edges1, edges2, edges3, edges4):
            x = self.x
            x1 = torch.tanh(self.conv1(x, edges0)).detach().to('cpu').numpy()
            x2 = torch.tanh(self.conv2(x, edges1)).detach().to('cpu').numpy()
            x3 = torch.tanh(self.conv3(x, edges2)).detach().to('cpu').numpy()
            x4 = torch.tanh(self.conv4(x, edges3)).detach().to('cpu').numpy()
            x5 = torch.tanh(self.conv5(x, edges4)).detach().to('cpu').numpy()
            return [x1, x2, x3, x4, x5]

        def get_attention_embedding(self, user_num, item_num, edges0, edges1, edges2, edges3, edges4):
            x = self.x
            x1 = torch.tanh(self.conv1(x, edges0))
            x2 = torch.tanh(self.conv2(x, edges1))
            x3 = torch.tanh(self.conv3(x, edges2))
            x4 = torch.tanh(self.conv4(x, edges3))
            x5 = torch.tanh(self.conv5(x, edges4))

            array_cat = torch.cat((x1, x2, x3, x4, x5), dim=1)
            user_logits_quary = torch.tanh(torch.unsqueeze(self.W_u_att(array_cat[0:user_num, :]), 1))
            item_logits_quary = torch.tanh(torch.unsqueeze(self.W_i_att(array_cat[user_num:, :]), 1))

            user_array_cat = array_cat[0:user_num, :].view(user_num, GENE_NUM, -1)
            item_array_cat = array_cat[user_num:, :].view(item_num, GENE_NUM, -1)

            user_pre_softmax = torch.sum(user_logits_quary * user_array_cat, dim=2)
            user_attention = F.softmax(user_pre_softmax, dim=1)

            item_pre_softmax = torch.sum(item_logits_quary * item_array_cat, dim=2)
            item_attention = F.softmax(item_pre_softmax, dim=1)

            user_fin_embed_pre = torch.unsqueeze(user_attention, 2) * user_array_cat
            item_fin_embed_pre = torch.unsqueeze(item_attention, 2) * item_array_cat

            user_fin_embed = torch.sum(user_fin_embed_pre, dim=1)
            item_fin_embed = torch.sum(item_fin_embed_pre, dim=1)
            return user_fin_embed.detach().to('cpu').numpy(), item_fin_embed.detach().to('cpu').numpy()



    dataset = './yelp_dataset.hdf5'
    with h5py.File(dataset, 'r') as f:
        train_pos = f['train_pos'][:]
        val_pos = f['val_pos'][:]
        test_pos = f['test_pos'][:]
        item_keys = f['item_keys'][:]
        item_frequency = f['item_freq'][:]
        user_num = f['user_num'][()]
        item_num = f['item_num'][()]


    train_data_pos = torch.from_numpy(train_pos).type(torch.LongTensor).to(device)
    val_data_pos = torch.from_numpy(val_pos).type(torch.LongTensor).to(device)
    test_data_pos = torch.from_numpy(test_pos).type(torch.LongTensor).to(device)


    # aliastable
    aliasTable = AliasTable(weights=item_frequency, keys=item_keys)

    def metrics(batch_pos, batch_neg, training=True):
        hit_num1 = 0.0
        hit_num3 = 0.0
        hit_num20 = 0.0
        hit_num50 = 0.0
        mrr_accu10 = 0.0
        mrr_accu20 = 0.0
        mrr_accu50 = 0.0
        ndcg_accu10 = 0.0
        ndcg_accu20 = 0.0
        ndcg_accu50 = 0.0

        if training:
            batch_neg_of_user = torch.split(batch_neg, NEG_SIZE_TRAIN, dim=0)
        else:
            batch_neg_of_user = torch.split(batch_neg, NEG_SIZE_RANKING, dim=0)
        for i in range(batch_pos.shape[0]):
            pre_rank_tensor = torch.cat((batch_pos[i].view(1, 1), batch_neg_of_user[i]), dim=0)
            _, indices = torch.topk(pre_rank_tensor, k=pre_rank_tensor.shape[0], dim=0)
            rank = torch.squeeze((indices == 0).nonzero().to('cpu'))
            rank = rank[0]
            if rank < 50:
                ndcg_accu50 = ndcg_accu50 + torch.log(torch.tensor([2.0])) / torch.log((rank + 2).type(torch.float32))
                mrr_accu50 = mrr_accu50 + 1 / (rank + 1).type(torch.float32)
                hit_num50 = hit_num50 + 1
            if rank < 20:
                ndcg_accu20 = ndcg_accu20 + torch.log(torch.tensor([2.0])) / torch.log((rank + 2).type(torch.float32))
                mrr_accu20 = mrr_accu20 + 1 / (rank + 1).type(torch.float32)
                hit_num20 = hit_num20 + 1
            if rank < 10:
                ndcg_accu10 = ndcg_accu10 + torch.log(torch.tensor([2.0])) / torch.log((rank + 2).type(torch.float32))
            if rank < 10:
                mrr_accu10 = mrr_accu10 + 1 / (rank + 1).type(torch.float32)
            if rank < 3:
                hit_num3 = hit_num3 + 1
            if rank < 1:
                hit_num1 = hit_num1 + 1
        return hit_num1 / batch_pos.shape[0], hit_num3 / batch_pos.shape[0], hit_num20 / batch_pos.shape[0], hit_num50 / \
               batch_pos.shape[0], mrr_accu10 / batch_pos.shape[0], mrr_accu20 / batch_pos.shape[0], mrr_accu50 / \
               batch_pos.shape[0], \
               ndcg_accu10 / batch_pos.shape[0], ndcg_accu20 / batch_pos.shape[0], ndcg_accu50 / batch_pos.shape[0]



    def train(model, optimizer, edges0,edges1,edges2,edges3,edges4, batch_pos, epoch, step, device,train_eval = False):
        model.train()
        optimizer.zero_grad()
        batch_neg_items = aliasTable.draw(NEG_SIZE_TRAIN * batch_pos.shape[0])
        # random neg samples
        #batch_neg_items = randint(low=item_keys.min(), high=item_keys.max() + 1, size=NEG_SIZE_TRAIN * batch_pos.shape[0])

        # Sample edges for fast training
        p0 = 100000 / edges0.shape[1]
        if p0 < 1:
            edges0_sp = dropout_adj(edges0, p=(1 - p0), force_undirected=True)[0]
        else:
            edges0_sp = edges0

        p1 = 100000 / edges1.shape[1]
        if p1 < 1:
            edges1_sp = dropout_adj(edges1, p=(1 - p1), force_undirected=True)[0]
        else:
            edges1_sp = edges1

        p2 = 100000 / edges2.shape[1]
        if p2 < 1:
            edges2_sp = dropout_adj(edges2, p=(1 - p2), force_undirected=True)[0]
        else:
            edges2_sp = edges2

        p3 = 100000 / edges3.shape[1]
        if p3 < 1:
            edges3_sp = dropout_adj(edges3, p=(1 - p3), force_undirected=True)[0]
        else:
            edges3_sp = edges3

        p4 = 100000 / edges4.shape[1]
        if p4 < 1:
            edges4_sp = dropout_adj(edges4, p=(1 - p4), force_undirected=True)[0]
        else:
            edges4_sp = edges4


        output_pos_logits, output_neg_logits = model(batch_pos, batch_neg_items,edges0_sp,edges1_sp,edges2_sp,edges3_sp,edges4_sp,)
        pos = torch.repeat_interleave(output_pos_logits, NEG_SIZE_TRAIN, dim=0)
        target = torch.ones(NEG_SIZE_TRAIN * batch_pos.shape[0], 1).to(device)
        loss = F.margin_ranking_loss(pos, output_neg_logits, target, margin=LOSS_MARGIN, reduction='sum')
        loss.backward()
        optimizer.step()
        if TRAIN_EVAL:
            HR1, HR3, MRR10, NDCG10 = metrics(output_pos_logits, output_neg_logits, training=True)
            # print("Epoch:" + str(epoch) + ", step:" + str(step) + ', Loss:' + str(loss.data.cpu().numpy()))
            return loss.to('cpu').detach().numpy(), HR1, HR3, MRR10, NDCG10, output_pos_logits.mean().to('cpu').detach().numpy(), output_neg_logits.mean().to('cpu').detach().numpy()
        else:
            return loss.to('cpu').detach().numpy(), output_pos_logits.mean().to('cpu').detach().numpy(), output_neg_logits.mean().to('cpu').detach().numpy()


    def val(model,batch_val_pos, edges0, edges1, edges2, edges3, edges4, device):
        model.eval()
        with torch.no_grad():
            p0 = 100000 / edges0.shape[1]
            if p0 < 1:
                edges0_sp = dropout_adj(edges0, p=(1-p0), force_undirected=True)[0]
            else:
                edges0_sp = edges0

            p1 = 100000 / edges1.shape[1]
            if p1 < 1:
                edges1_sp = dropout_adj(edges1, p=(1-p1), force_undirected=True)[0]
            else:
                edges1_sp = edges1

            p2 = 100000 / edges2.shape[1]
            if p2 < 1:
                edges2_sp = dropout_adj(edges2, p=(1-p2), force_undirected=True)[0]
            else:
                edges2_sp = edges2

            p3 = 100000 / edges3.shape[1]
            if p3 < 1:
                edges3_sp = dropout_adj(edges3, p=(1-p3), force_undirected=True)[0]
            else:
                edges3_sp = edges3

            p4 = 100000 / edges4.shape[1]
            if p4 < 1:
                edges4_sp = dropout_adj(edges4, p=(1-p4), force_undirected=True)[0]
            else:
                edges4_sp = edges4

            batch_neg_items = aliasTable.draw(NEG_SIZE_RANKING * batch_val_pos.shape[0])
            #batch_neg_items = randint(low=item_keys.min(), high=item_keys.max() + 1, size = NEG_SIZE_RANKING * batch_val_pos.shape[0])


            val_pos_logits, val_neg_logits = model(batch_val_pos, batch_neg_items, edges0_sp, edges1_sp, edges2_sp, edges3_sp, edges4_sp)
            target = torch.ones(NEG_SIZE_RANKING * batch_val_pos.shape[0], 1).to(device)
            pos = torch.repeat_interleave(val_pos_logits, NEG_SIZE_RANKING, dim=0)
            loss = F.margin_ranking_loss(pos, val_neg_logits, target, margin=LOSS_MARGIN, reduction='mean')
            HR1, HR3,HR20,HR50, MRR10, MRR20, MRR50, NDCG10, NDCG20, NDCG50 = metrics(val_pos_logits, val_neg_logits, training=False)


        return loss.to('cpu').detach().numpy(), HR1, HR3,HR20,HR50, MRR10, MRR20, MRR50, NDCG10, NDCG20, NDCG50

    def test(model, batch_test_pos, edges0, edges1, edges2, edges3, edges4, DEBUG=False, device = device):
        model.eval()
        with torch.no_grad():
            p0 = 100000 / edges0.shape[1]
            if p0 < 1:
                edges0_sp = dropout_adj(edges0, p=(1-p0), force_undirected=True)[0]
            else:
                edges0_sp = edges0

            p1 = 100000 / edges1.shape[1]
            if p1 < 1:
                edges1_sp = dropout_adj(edges1, p=(1-p1), force_undirected=True)[0]
            else:
                edges1_sp = edges1

            p2 = 100000 / edges2.shape[1]
            if p2 < 1:
                edges2_sp = dropout_adj(edges2, p=(1-p2), force_undirected=True)[0]
            else:
                edges2_sp = edges2

            p3 = 100000 / edges3.shape[1]
            if p3 < 1:
                edges3_sp = dropout_adj(edges3, p=(1-p3), force_undirected=True)[0]
            else:
                edges3_sp = edges3

            p4 = 100000 / edges4.shape[1]
            if p4 < 1:
                edges4_sp = dropout_adj(edges4, p=(1-p4), force_undirected=True)[0]
            else:
                edges4_sp = edges4

            batch_neg_items = aliasTable.draw(NEG_SIZE_RANKING * batch_test_pos.shape[0])
            #batch_neg_items = randint(low=item_keys.min(), high=item_keys.max() + 1, size=NEG_SIZE_RANKING * batch_test_pos.shape[0])

            if not DEBUG:
                test_pos_logits, test_neg_logits = model(batch_test_pos, batch_neg_items,edges0_sp,edges1_sp,edges2_sp,edges3_sp,edges4_sp)
            else:
                test_pos_logits, test_neg_logits,gcn_embedding,gcn_process_user_emb,gcn_process_item_pos_emb,gcn_process_item_neg_emb,user_fin_embed,item_pos_fin_embed,item_neg_fin_embed,user_attention,item_pos_attn, item_neg_attn = model(batch_test_pos, batch_neg_items, edges0, edges1, edges2,
                                                         edges3, edges4, DEBUG)
        if not DEBUG:
            target = torch.ones(NEG_SIZE_RANKING * batch_test_pos.shape[0], 1).to(device)
            pos = torch.repeat_interleave(test_pos_logits, NEG_SIZE_RANKING, dim=0)
            loss = F.margin_ranking_loss(pos, test_neg_logits, target, margin=LOSS_MARGIN, reduction='mean')
            HR1, HR3,HR20,HR50, MRR10, MRR20, MRR50, NDCG10, NDCG20, NDCG50 = metrics(test_pos_logits, test_neg_logits, training=False)
            return loss.to('cpu').detach().numpy(), HR1, HR3,HR20,HR50, MRR10, MRR20, MRR50, NDCG10, NDCG20, NDCG50
        else:
            return test_pos_logits, test_neg_logits,gcn_embedding,gcn_process_user_emb,gcn_process_item_pos_emb,gcn_process_item_neg_emb,user_fin_embed,item_pos_fin_embed,item_neg_fin_embed,user_attention,item_pos_attn, item_neg_attn



    my_edges = []
    edges_numpy = []
    for j in range(GENE_NUM):
        tmp_edges = genes_dict[str(i)+str(j)]
        tmp_numpy = np.array(tmp_edges)
        tmp = torch.from_numpy(tmp_numpy).type(torch.long)
        a = torch.cat((tmp, tmp[:, [1, 0]]), dim=0).t().contiguous()
        edges_numpy.append([tmp_edges])
        my_edges.append(a)
    edges_numpy = np.array(edges_numpy)
    #print(edges_numpy.shape)
    if SAVE_ADJ_EDGES:
        np.save('./result_edges/gen'+str(gen_num) + '_process' + str(i) + '_edges.npy', edges_numpy)

    edges0 = my_edges[0].to(device)
    edges1 = my_edges[1].to(device)
    edges2 = my_edges[2].to(device)
    edges3 = my_edges[3].to(device)
    edges4 = my_edges[4].to(device)


    user = np.load(MF_user)
    item = np.load(MF_item)
    pre_train = torch.cat((torch.from_numpy(user).to(device), torch.from_numpy(item).to(device)), dim=0)
    model = Net(int(user_num) + int(item_num), pre_train).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=LAMBDA)

    # init optimizer with dynamic LR
    update_func = lambda epoch: LEARNING_RATE if epoch <= WARM_UP_STEP else LEARNING_RATE * pow(epoch - WARM_UP_STEP,-0.5)
    scheduler = LambdaLR(optimizer, lr_lambda=update_func)

    best_val_ndcg = 0.0
    best_test_hr1 = 0.0
    best_test_hr3 = 0.0
    best_test_hr20 = 0.0
    best_test_hr50 = 0.0
    best_test_mrr10 = 0.0
    best_test_mrr20 = 0.0
    best_test_mrr50 = 0.0
    best_test_ndcg10 = 0.0
    best_test_ndcg20 = 0.0
    best_test_ndcg50 = 0.0

    writer = SummaryWriter(log_dir=LOG_DIR + 'run_'+'gen'+str(gen_num)+'_process'+str(i)+'/')
    
    early_stop = 0
    shuffle_index = np.linspace(start=0, stop=train_data_pos.shape[0],endpoint=False,num = train_data_pos.shape[0],dtype=int)
    for epoch in range(1, EPOCH_NUM):
        np.random.shuffle(shuffle_index)
        train_data_pos = train_data_pos[shuffle_index]
        batch_pos_data = torch.split(train_data_pos, BATCH_SIZE, dim=0)
        tr_loss = 0.0
        if TRAIN_EVAL:
            tr_hr1 = 0.0
            tr_hr3 = 0.0
            tr_hr20 = 0.0
            tr_hr50 = 0.0
            tr_mrr10 = 0.0
            tr_mrr20 = 0.0
            tr_mrr50 = 0.0
            tr_ndcg10 = 0.0
            tr_ndcg20 = 0.0
            tr_ndcg50 = 0.0
            for step in range(len(batch_pos_data)):
                batch_pos = batch_pos_data[step]

                tmp_tr_loss, tmp_tr_hr1, tmp_tr_hr3, tmp_tr_mrr10, tmp_tr_ndcg10, _, _ = train(model, optimizer, edges0,
                                                                                               edges1, edges2, edges3,
                                                                                               edges4, batch_pos, epoch,
                                                                                               step, device,
                                                                                               train_eval=TRAIN_EVAL)
                tr_loss += tmp_tr_loss
                tr_hr1 += tmp_tr_hr1
                tr_hr3 += tmp_tr_hr3
                tr_mrr10 += tmp_tr_mrr10
                tr_ndcg10 += tmp_tr_ndcg10

            tr_loss = tr_loss / len(batch_pos_data)
            tr_hr1 = tr_hr1 / len(batch_pos_data)
            tr_hr3 = tr_hr3 / len(batch_pos_data)
            tr_mrr10 = tr_mrr10 / len(batch_pos_data)
            tr_ndcg10 = tr_ndcg10 / len(batch_pos_data)
        else:
            for step in range(len(batch_pos_data)):
                batch_pos = batch_pos_data[step]
                tmp_tr_loss, _, _ = train(model, optimizer, edges0,edges1, edges2, edges3,edges4, batch_pos, epoch,step, device,train_eval=TRAIN_EVAL)
                tr_loss += tmp_tr_loss
            tr_loss = tr_loss / len(batch_pos_data)

        # Evaluate model
        if ((WARM_UP_STEP - epoch) < 5) and (epoch % 2 == 0):
            val_loss = 0.0
            val_hr1 = 0.0
            val_hr3 = 0.0
            val_hr20 = 0.0
            val_hr50 = 0.0
            val_mrr10 = 0.0
            val_mrr20 = 0.0
            val_mrr50 = 0.0
            val_ndcg10 = 0.0
            val_ndcg20 = 0.0
            val_ndcg50 = 0.0
            batch_val_data = torch.split(val_data_pos, BATCH_SIZE, dim=0)
            for val_step in range(len(batch_val_data)):
                batch_val_pos = batch_val_data[val_step]
                tmp_val_loss, tmp_val_hr1, tmp_val_hr3, tmp_val_hr20, tmp_val_hr50, tmp_val_mrr10, tmp_val_mrr20, tmp_val_mrr50, tmp_val_ndcg10, tmp_val_ndcg20, tmp_val_ndcg50 = val(model, batch_val_pos, edges0, edges1, edges2, edges3, edges4, device)
                val_loss = val_loss + tmp_val_loss
                val_hr1 += tmp_val_hr1
                val_hr3 += tmp_val_hr3
                val_hr20 += tmp_val_hr20
                val_hr50 += tmp_val_hr50
                val_mrr10 += tmp_val_mrr10
                val_mrr20 += tmp_val_mrr20
                val_mrr50 += tmp_val_mrr50
                val_ndcg10 += tmp_val_ndcg10
                val_ndcg20 += tmp_val_ndcg20
                val_ndcg50 += tmp_val_ndcg50
            val_loss = val_loss / len(batch_val_data)
            val_hr1 = val_hr1 / len(batch_val_data)
            val_hr3 = val_hr3 / len(batch_val_data)
            val_hr20 = val_hr20 / len(batch_val_data)
            val_hr50 = val_hr50 / len(batch_val_data)
            val_mrr10 = val_mrr10 / len(batch_val_data)
            val_mrr20 = val_mrr20 / len(batch_val_data)
            val_mrr50 = val_mrr50 / len(batch_val_data)
            val_ndcg10 = val_ndcg10 / len(batch_val_data)
            val_ndcg20 = val_ndcg20 / len(batch_val_data)
            val_ndcg50 = val_ndcg50 / len(batch_val_data)

            test_loss = 0.0
            test_hr1 = 0.0
            test_hr3 = 0.0
            test_hr20 = 0.0
            test_hr50 = 0.0
            test_mrr10 = 0.0
            test_mrr20 = 0.0
            test_mrr50 = 0.0
            test_ndcg10 = 0.0
            test_ndcg20 = 0.0
            test_ndcg50 = 0.0
            batch_test_data = torch.split(test_data_pos, BATCH_SIZE, dim=0)
            for test_step in range(len(batch_test_data)):
                batch_test_pos = batch_test_data[test_step]
                tmp_test_loss, tmp_test_hr1, tmp_test_hr3, tmp_test_hr20, tmp_test_hr50, tmp_test_mrr10, tmp_test_mrr20, tmp_test_mrr50, tmp_test_ndcg10, tmp_test_ndcg20, tmp_test_ndcg50 = test(model, batch_test_pos, edges0, edges1, edges2, edges3, edges4, DEBUG=False, device=device)
                test_loss = test_loss + tmp_test_loss
                test_hr1 += tmp_test_hr1
                test_hr3 += tmp_test_hr3
                test_hr20 += tmp_test_hr20
                test_hr50 += tmp_test_hr50
                test_mrr10 += tmp_test_mrr10
                test_mrr20 += tmp_test_mrr20
                test_mrr50 += tmp_test_mrr50
                test_ndcg10 += tmp_test_ndcg10
                test_ndcg20 += tmp_test_ndcg20
                test_ndcg50 += tmp_test_ndcg50

            test_loss = test_loss / len(batch_test_data)
            test_hr1 = test_hr1 / len(batch_test_data)
            test_hr3 = test_hr3 / len(batch_test_data)
            test_hr20 = test_hr20 / len(batch_test_data)
            test_hr50 = test_hr50 / len(batch_test_data)
            test_mrr10 = test_mrr10 / len(batch_test_data)
            test_mrr20 = test_mrr20 / len(batch_test_data)
            test_mrr50 = test_mrr50 / len(batch_test_data)
            test_ndcg10 = test_ndcg10 / len(batch_test_data)
            test_ndcg20 = test_ndcg20 / len(batch_test_data)
            test_ndcg50 = test_ndcg50 / len(batch_test_data)

            writer.add_scalar('scalar/train/loss', tr_loss, epoch)
            writer.add_scalar('scalar/val/loss', val_loss, epoch)
            writer.add_scalar('scalar/test/loss', test_loss, epoch)

            if TRAIN_EVAL:
                writer.add_scalar('scalar/train/hr1', tr_hr1, epoch)
                writer.add_scalar('scalar/train/hr3', tr_hr3, epoch)
                writer.add_scalar('scalar/train/mrr10', tr_mrr10, epoch)
                writer.add_scalar('scalar/train/ndcg10', tr_ndcg10, epoch)


            writer.add_scalar('scalar/val/hr1', val_hr1, epoch)
            writer.add_scalar('scalar/val/hr3', val_hr3, epoch)
            writer.add_scalar('scalar/val/mrr10', val_mrr10, epoch)
            writer.add_scalar('scalar/val/ndcg10', val_ndcg10, epoch)

            writer.add_scalar('scalar/test/hr1', test_hr1, epoch)
            writer.add_scalar('scalar/test/hr3', test_hr3, epoch)
            writer.add_scalar('scalar/test/mrr10', test_mrr10, epoch)
            writer.add_scalar('scalar/test/ndcg10', test_ndcg10, epoch)

            # debug
            embedding = model.get_model_embedding()
            embedding_norm = np.linalg.norm(embedding, axis=1).mean()
            writer.add_scalar('scalar/model/embedding_norm', embedding_norm, epoch)
            if DEBUG:
                if epoch % 5 == 0:
                    test_pos_logits, test_neg_logits, gcn_embedding, gcn_process_user_emb, gcn_process_item_pos_emb, \
                    gcn_process_item_neg_emb, user_fin_embed, item_pos_fin_embed, item_neg_fin_embed, user_attention, item_pos_attn, item_neg_attn = test(
                        model, batch_test_pos,
                        edges0, edges1,
                        edges2, edges3,
                        edges4, DEBUG=True)
                    np.save('./debug_info/gen' + str(gen_num) + '_process' + str(i) + '_epoch' + str(
                        epoch) + '_test_pos_logits.npy', test_pos_logits)
                    np.save('./debug_info/gen' + str(gen_num) + '_process' + str(i) + '_epoch' + str(
                        epoch) + '_test_neg_logits.npy', test_neg_logits)
                    np.save('./debug_info/gen' + str(gen_num) + '_process' + str(i) + '_epoch' + str(
                        epoch) + '_gcn_embedding.npy', gcn_embedding)
                    np.save('./debug_info/gen' + str(gen_num) + '_process' + str(i) + '_epoch' + str(
                        epoch) + '_gcn_process_user_emb.npy', gcn_process_user_emb)
                    np.save('./debug_info/gen' + str(gen_num) + '_process' + str(i) + '_epoch' + str(
                        epoch) + '_gcn_process_item_pos_emb.npy', gcn_process_item_pos_emb)
                    np.save('./debug_info/gen' + str(gen_num) + '_process' + str(i) + '_epoch' + str(
                        epoch) + '_gcn_process_item_neg_emb.npy', gcn_process_item_neg_emb)
                    np.save('./debug_info/gen' + str(gen_num) + '_process' + str(i) + '_epoch' + str(
                        epoch) + '_user_fin_embed.npy', user_fin_embed)
                    np.save('./debug_info/gen' + str(gen_num) + '_process' + str(i) + '_epoch' + str(
                        epoch) + '_item_pos_fin_embed.npy', item_pos_fin_embed)
                    np.save('./debug_info/gen' + str(gen_num) + '_process' + str(i) + '_epoch' + str(
                        epoch) + '_item_neg_fin_embed.npy', item_neg_fin_embed)

                    np.save('./debug_info/gen' + str(gen_num) + '_process' + str(i) + '_epoch' + str(
                        epoch) + '_user_attention.npy', user_attention)
                    np.save('./debug_info/gen' + str(gen_num) + '_process' + str(i) + '_epoch' + str(
                        epoch) + '_item_pos_attn.npy', item_pos_attn)
                    np.save('./debug_info/gen' + str(gen_num) + '_process' + str(i) + '_epoch' + str(
                        epoch) + '_item_neg_attn.npy', item_neg_attn)

            if val_ndcg10 > best_val_ndcg:
                early_stop = 0
                best_val_ndcg = val_ndcg10
                best_test_hr1 = test_hr1
                best_test_hr3 = test_hr3
                best_test_hr20 = test_hr20
                best_test_hr50 = test_hr50
                best_test_mrr10 = test_mrr10
                best_test_mrr20 = test_mrr20
                best_test_mrr50 = test_mrr50
                best_test_ndcg10 = test_ndcg10
                best_test_ndcg20 = test_ndcg20
                best_test_ndcg50 = test_ndcg50
            elif epoch > WARM_UP_STEP:
                early_stop = early_stop + 1

            if TRAIN_EVAL:
                tr_log = 'Generation:{:02d}, Process: {:02d}, Epoch: {:03d}, tr HR1: {:.4f}, tr HR3: {:.4f}, tr MRR10: {:.4f}, tr NDCG10: {:.4f}'
                print(tr_log.format(gen_num, i, epoch, float(tr_hr1), float(tr_hr3), float(tr_mrr10), float(tr_ndcg10)))
            val_log = 'Generation:{:02d}, Process: {:02d}, Epoch: {:03d}, val HR1: {:.4f}, val HR3: {:.4f}, val HR20: {:.4f}, val HR50: {:.4f}, val MRR10: {:.4f}, val MRR20: {:.4f}, val MRR50: {:.4f}, val NDCG10: {:.4f}, val NDCG20: {:.4f}, val NDCG50: {:.4f}'
            print(val_log.format(gen_num, i, epoch, float(val_hr1), float(val_hr3), float(val_hr20), float(val_hr50),float(val_mrr10), float(val_mrr20), float(val_mrr50), float(val_ndcg10),float(val_ndcg20), float(val_ndcg50)))
            test_log = 'Generation:{:02d}, Process: {:02d}, Epoch: {:03d}, test HR1: {:.4f}, test HR3: {:.4f}, test HR20: {:.4f}, test HR50: {:.4f}, test MRR10: {:.4f}, test MRR20: {:.4f}, test MRR50: {:.4f}, test NDCG10: {:.4f}, test NDCG20: {:.4f}, test NDCG50: {:.4f}'
            print(test_log.format(gen_num, i, epoch, float(test_hr1), float(test_hr3), float(test_hr20), float(test_hr50),float(test_mrr10), float(test_mrr20), float(test_mrr50), float(test_ndcg10),float(test_ndcg20), float(test_ndcg50)))
            best_test_log = 'Generation:{:02d}, Process: {:02d}, Epoch: {:03d}, Best Test HR1: {:.4f}, Test HR3: {:.4f}, Test HR20: {:.4f}, Test HR50: {:.4f}, Test MRR10: {:.4f}, Test MRR20: {:.4f}, Test MRR50: {:.4f}, Test NDCG10: {:.4f}, Test NDCG20: {:.4f},Test NDCG50: {:.4f}'
            print(best_test_log.format(gen_num, i, epoch, float(best_test_hr1), float(best_test_hr3), float(best_test_hr20), float(best_test_hr50), float(best_test_mrr10), float(best_test_mrr20), float(best_test_mrr50), float(best_test_ndcg10),float(best_test_ndcg20),float(best_test_ndcg50)))
        
        scheduler.step()
        if early_stop > EARLY_STOP:
            print("================================================")
            info = 'EARLY STOP:\nGeneration:{:02d}, Process: {:02d} at Epoch: {:03d}'
            print(info.format(gen_num, i, epoch))
            print("================================================")
            break


    writer.close()
    result_dict[i] = [best_test_hr1, best_test_hr3,best_test_hr20,best_test_hr50, best_test_mrr10,best_test_mrr20,best_test_mrr50, best_test_ndcg10,best_test_ndcg20,best_test_ndcg50]

    return 1


def eval_and_elimiate(gen_num, populations_genes, old_gene_pools):
    populations_genes_cp = copy.deepcopy(populations_genes)

    old_gene_pools_cp = copy.deepcopy(old_gene_pools)
    populations_genes_dict = {}
    for i in range(POPULATION_SIZE):
        for j in range(GENE_NUM):
            populations_genes_dict[str(i)+str(j)] = populations_genes_cp[i][j]
            if populations_genes_cp[i][j].shape[0] == 0:
                print('****************************************************************************************')
                print("STILL WRONG and get empty adj in eval, at "+str(gen_num)+', geti '+ str(i) + ', of gene ' + str(j))


    manager = Manager()
    result_dict = manager.dict()

    mp = get_context('forkserver')
    p = mp.Pool(MULTI_PROCESS_NUM_TRAIN, maxtasksperchild=1)

    result = [p.apply_async(multi_process_eval_model, args=(gen_num, i, populations_genes_dict, result_dict)) for i in range(len(populations_genes_cp))]
    for i in result:
        i.get()
    
    p.close()
    p.join()

    populations_performance = np.zeros((len(populations_genes),10))
    for key in result_dict.keys():
        populations_performance[key] = result_dict[key]
    ranking = np.argsort(populations_performance[:,3])
    preserved_index = ranking[int(ELIMINATE_RATE*POPULATION_SIZE):]
    new_gene_pools = []
    for i in preserved_index:
        new_gene_pools = new_gene_pools + old_gene_pools_cp[GENE_NUM*i:(GENE_NUM*i+GENE_NUM)]


    while len(new_gene_pools) < len(old_gene_pools_cp):
        pre_p = np.exp(populations_performance[preserved_index,3])
        pre_p = pre_p / pre_p.sum()
        index = choice(preserved_index,size = 1,p=pre_p)[0]
        new_gene_pools = new_gene_pools + old_gene_pools_cp[GENE_NUM*index:(GENE_NUM*index+GENE_NUM)]

    best_genes = []
    for i in range(0,int(len(old_gene_pools_cp)/GENE_NUM)):
        best_genes.append(old_gene_pools_cp[GENE_NUM*i:(GENE_NUM*i+GENE_NUM)])

    best_performace = populations_performance[ranking[-1]]
    
    return new_gene_pools, best_genes, best_performace, populations_performance


if __name__ == "__main__":
    freeze_support()
    adj, the_dict = construct_adj()


    # Meta-path examples
    UBUB = [['U', 'B', 'U', 'B'], [[0., -1., 0., 1.], [0., 0., 1., -1.], [0., 0., 0., 1.], [0., 0., 0., 0.]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 1]
    UBU = [['U', 'B', 'U'], [[0., -1., 1.], [0., 0., 1.], [0., 0., 0.]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 1]
    UBAB = [['U', 'B', 'A', 'B'], [[0., -1., 0., 1.], [0., 0., 1., -1.], [0., 0., 0., 1.], [0., 0., 0., 0.]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 1]
    UBIB = [['U', 'B', 'I', 'B'], [[0., -1., 0., 1.], [0., 0., 1., -1.], [0., 0., 0., 1.], [0., 0., 0., 0.]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 1]
    UB = [['U', 'B'], [[0, 1], [0, 0]], [[0, 0], [0, 0]], 1]

    # Auto init
    genes_pools = [mutation_init() for _ in range(GENE_POOL_SIZE)]
    #genes_pools = [UB,UBUB,UBU,UBAB,UBIB,UB,UBUB,UBU,UBAB,UBIB,UB,UBUB,UBU,UBAB,UBIB,UB,UBUB,UBU,UBAB,UBIB,UB,UBUB,UBU,UBAB,UBIB,UB,UBUB,UBU,UBAB,UBIB]

    already_searched_dict = {}

    best_performance_of_gens = np.zeros((GENERATION, 10), dtype=float)
    best_genes = []
    gen_performance = []


    stable_prob = INIT_STABLE_PROB
    complex_prob = INIT_COMPLEX_PROB
    add_node_prob = ADD_NODE_PROB
    for i in range(PRE_MUTATE):
        genes_pools = [mutate_graph(candidate, stable_prob, complex_prob, add_node_prob) for candidate in genes_pools]

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    time_log = open(TIME_LOG,'w')
    begin_time = datetime.now()
    time_log.write('Training Begins at:' + str(begin_time))
    time_log.close()
    for gen in range(GENERATION):
        # Simulated Annealing
        '''if gen > 10:
            add_edge_prob = add_edge_prob * 0.95
            add_node_prob = add_node_prob * 0.95
            delete_prob = delete_prob * 0.95
            stable_prob = 1 - add_edge_prob - add_node_prob - delete_prob'''

        if gen > ANNEALING_EPOCH:
            stable_prob = STABLE_PROB
            complex_prob = COMPLEX_PROB

        genes, genes_pools, already_searched_dict, no_adj_count = mutate_and_search(gen, genes_pools, already_searched_dict, the_dict, stable_prob, complex_prob, add_node_prob)
        genes_pools, best_genes_tmp, best_performance, populations_performance = eval_and_elimiate(gen, genes, genes_pools)
        best_performance_of_gens[gen] = best_performance
        gen_performance.append(populations_performance)
        best_genes.append(best_genes_tmp)
        print('===============================================================================================================')
        print('Generation ' + str(gen) + ' completed!')
        print('Best performance of generation ' + str(gen) + ' is' + str(best_performance_of_gens[gen]))
        print(best_genes)
        np.save(LOG_DIR+'best_genes.npy',best_genes)
        np.save(LOG_DIR+'best_performance_of_genes.npy', best_performance_of_gens)
        np.save(LOG_DIR+'performance_of_genes.npy', gen_performance)

        time_log = open(TIME_LOG, 'a')
        time_log.write('Generation '+ str(gen) + 'completed in:' + str((datetime.now() - begin_time).total_seconds() / 60) + '\n')
        time_log.close()
    print(best_genes)
    print(best_performance_of_gens)
    print(gen_performance)





