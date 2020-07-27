import math
import numpy as np
from numpy.random import random, randint, choice
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import torch.nn.functional as F
import pickle
import h5py
from torch.optim.lr_scheduler import LambdaLR
from tensorboardX import SummaryWriter
from datetime import datetime
import setproctitle

# These two settings are overwitten by line 227-228 for grid search. 
LEARNING_RATE = 0.04
LAMBDA = 0.001

BATCH_SIZE = 200
EMBEDDING_SIZE = 64   # Best embedding size for GEMS
NEG_SIZE_TRAIN = 4
NEG_SIZE_RANKING = 100

EPOCH_NUM = 100
WARM_UP_STEP = 15
EARLY_STOP = 3
LOSS_MARGIN = 0.3

TRAIN_EVAL = False
SAVE_EMBEDDING = True  # Get the pretrain embedding

setproctitle.setproctitle('BMF_yelp')
device = 'cuda:0'
###########################################################################################################################################
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


class BMF(torch.nn.Module):
    def __init__(self, user_cnt, item_cnt):
        super(BMF, self).__init__()
        self.user_num = user_cnt
        self.item_num = item_cnt
        self.user_embedding = nn.Parameter(torch.Tensor(user_cnt, EMBEDDING_SIZE))
        nn.init.kaiming_uniform_(self.user_embedding, a=math.sqrt(5))
        self.item_embedding = nn.Parameter(torch.Tensor(item_cnt, EMBEDDING_SIZE))
        nn.init.kaiming_uniform_(self.item_embedding, a=math.sqrt(5))


    def forward(self, batch_pos, neg_item_index):

        user_index = batch_pos[:,0]
        pos_item_index = batch_pos[:,1] - self.user_num
        neg_item_index = neg_item_index - self.user_num

        user_pad = (0, 1)
        item_pad = (1, 0)
        W_user = self.user_embedding[user_index]
        W_user_with_bias = F.pad(W_user,user_pad,'constant',1.0)

        W_pos_item = self.item_embedding[pos_item_index]
        W_pos_item_with_bias = F.pad(W_pos_item, item_pad, 'constant', 1.0)
        W_neg_item = self.item_embedding[neg_item_index]
        W_neg_item_with_bias = F.pad(W_neg_item, item_pad, 'constant', 1.0)

        pos_logits = torch.sigmoid(torch.sum(torch.mul(W_user_with_bias,W_pos_item_with_bias),dim=1, keepdim=True))
        neg_logits = torch.sigmoid(torch.sum(torch.mul(torch.unsqueeze(W_user_with_bias, dim=1),
                W_neg_item_with_bias.view(W_user_with_bias.shape[0], -1,EMBEDDING_SIZE+1)).view(-1,EMBEDDING_SIZE+1),dim=1, keepdim=True))

        return pos_logits, neg_logits

    def get_model_embedding(self):
        return self.user_embedding.detach().to('cpu').numpy(), self.item_embedding.detach().to('cpu').numpy()



def metrics(batch_pos, batch_neg, training = True):
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
    return hit_num1 / batch_pos.shape[0], hit_num3 / batch_pos.shape[0],hit_num20 / batch_pos.shape[0], hit_num50 / batch_pos.shape[0], mrr_accu10 / batch_pos.shape[0], mrr_accu20 / batch_pos.shape[0],mrr_accu50 / batch_pos.shape[0],\
           ndcg_accu10/batch_pos.shape[0],ndcg_accu20/batch_pos.shape[0],ndcg_accu50/batch_pos.shape[0]


def train(model, optimizer, batch_pos, aliasTable,device,train_eval = False):
    model.train()
    optimizer.zero_grad()
    batch_neg_items = aliasTable.draw(NEG_SIZE_TRAIN * batch_pos.shape[0])
    #batch_neg_items = randint(low=item_keys.min(), high=item_keys.max() + 1, size=NEG_SIZE_TRAIN * batch_pos.shape[0])

    output_pos_logits, output_neg_logits = model(batch_pos, batch_neg_items)
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

def val(model,batch_val_pos, aliasTable, device):
    model.eval()
    with torch.no_grad():
        batch_neg_items = aliasTable.draw(NEG_SIZE_RANKING * batch_val_pos.shape[0])
        #batch_neg_items = randint(low=item_keys.min(), high=item_keys.max() + 1, size = NEG_SIZE_RANKING * batch_val_pos.shape[0])


        val_pos_logits, val_neg_logits = model(batch_val_pos, batch_neg_items)
        target = torch.ones(NEG_SIZE_RANKING * batch_val_pos.shape[0], 1).to(device)
        pos = torch.repeat_interleave(val_pos_logits, NEG_SIZE_RANKING, dim=0)
        loss = F.margin_ranking_loss(pos, val_neg_logits, target, margin=LOSS_MARGIN, reduction='mean')
        HR1, HR3,HR20,HR50, MRR10, MRR20, MRR50, NDCG10, NDCG20, NDCG50 = metrics(val_pos_logits, val_neg_logits,training=False)

        return loss.to('cpu').detach().numpy(), HR1, HR3,HR20,HR50, MRR10, MRR20, MRR50, NDCG10, NDCG20, NDCG50

def test(model, batch_test_pos,aliasTable, device, DEBUG=False,):
    model.eval()
    with torch.no_grad():
        batch_neg_items = aliasTable.draw(NEG_SIZE_RANKING * batch_test_pos.shape[0])
        #batch_neg_items = randint(low=item_keys.min(), high=item_keys.max() + 1, size=NEG_SIZE_RANKING * batch_test_pos.shape[0])

        if not DEBUG:
            test_pos_logits, test_neg_logits = model(batch_test_pos, batch_neg_items)
        else:
            test_pos_logits, test_neg_logits,gcn_embedding,gcn_process_user_emb,gcn_process_item_pos_emb,\
            gcn_process_item_neg_emb,user_fin_embed,item_pos_fin_embed,item_neg_fin_embed,user_attention,item_pos_attn,\
            item_neg_attn = model(batch_test_pos, batch_neg_items,DEBUG)
    if not DEBUG:
        target = torch.ones(NEG_SIZE_RANKING * batch_test_pos.shape[0], 1).to(device)
        pos = torch.repeat_interleave(test_pos_logits, NEG_SIZE_RANKING, dim=0)
        loss = F.margin_ranking_loss(pos, test_neg_logits, target, margin=LOSS_MARGIN, reduction='mean')
        HR1, HR3,HR20,HR50, MRR10, MRR20, MRR50, NDCG10, NDCG20, NDCG50 = metrics(test_pos_logits, test_neg_logits,training=False)
        return loss.to('cpu').detach().numpy(), HR1, HR3,HR20,HR50, MRR10, MRR20, MRR50, NDCG10, NDCG20, NDCG50
    else:
        return test_pos_logits, test_neg_logits,gcn_embedding,gcn_process_user_emb,gcn_process_item_pos_emb,\
               gcn_process_item_neg_emb,user_fin_embed,item_pos_fin_embed,item_neg_fin_embed,user_attention,item_pos_attn, item_neg_attn

###########################################################################################################################################

# Read Data
dataset = '../yelp_dataset.hdf5'
with h5py.File(dataset, 'r') as f:
    train_pos = f['train_pos'][:]
    val_pos = f['val_pos'][:]
    test_pos = f['test_pos'][:]
    item_keys = f['item_keys'][:]
    item_frequency = f['item_freq'][:]
    user_num = f['user_num'][()]
    item_num = f['item_num'][()]

train_data_pos = torch.from_numpy(train_pos).type(torch.LongTensor)
val_data_pos = torch.from_numpy(val_pos).type(torch.LongTensor)
test_data_pos = torch.from_numpy(test_pos).type(torch.LongTensor)


# aliastable
aliasTable = AliasTable(weights=item_frequency, keys=item_keys)

logfile = open('./BMF_search_log_neg_new','w')
#0.3, 0.1, 0.06, 0.03,0.1, 0.01, 0.006
#0.1, 0.06, 0.03, 0.01, 0.006, 0.003, 0.001, 0.0006, 0.0003, 0.00006
run = 0
#for LEARNING_RATE in [0.1, 0.06, 0.03, 0.1, 0.01, 0.006]:
#    for LAMBDA in [ 0.01, 0.006, 0.003, 0.001, 0.0006]:
#0.06, 0.006
for LEARNING_RATE in [0.06]:
    for LAMBDA in [0.003]:

        # Dataset
        train_dataset_pos = torch_data.TensorDataset(train_data_pos)
        train_dataloader_pos = torch_data.DataLoader(dataset=train_dataset_pos, batch_size=BATCH_SIZE, shuffle=True,
                                                     num_workers=4)
        val_dataset_pos = torch_data.TensorDataset(val_data_pos)
        val_dataloader_pos = torch_data.DataLoader(dataset=val_dataset_pos, batch_size=BATCH_SIZE, shuffle=True,
                                                   num_workers=4)
        test_dataset_pos = torch_data.TensorDataset(test_data_pos)
        test_dataloader_pos = torch_data.DataLoader(dataset=test_dataset_pos, batch_size=BATCH_SIZE, shuffle=True,
                                                    num_workers=4)

        # Init model
        model = BMF(int(user_num), int(item_num)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=LAMBDA)

        # optimizer init (with dynamic LR)
        update_func = lambda epoch: LEARNING_RATE if epoch<=WARM_UP_STEP else LEARNING_RATE*pow(epoch - WARM_UP_STEP, -0.5)
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

        writer = SummaryWriter(log_dir='./result_log/BMF_search_neg_new/' + 'Tensor_HE_init_' + 'Lr_' + str(LEARNING_RATE) + '_Lamb_' + str(LAMBDA) + '_' + str(run))
        early_stop = 0
        epoch_start = datetime.now()
        for epoch in range(1, EPOCH_NUM):
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
                for step, batch_pos in enumerate(train_dataloader_pos, 0):
                    tmp_tr_loss, tmp_tr_hr1, tmp_tr_hr3, tmp_tr_hr20,tmp_tr_hr50,tmp_tr_mrr10,tmp_tr_mrr20,tmp_tr_mrr50, tmp_tr_ndcg10,tmp_tr_ndcg20,tmp_tr_ndcg50, _, _ = train(model, optimizer,
                                                                                                   batch_pos[0],
                                                                                                   aliasTable, device,
                                                                                               train_eval=TRAIN_EVAL)
                    tr_loss += tmp_tr_loss
                    tr_hr1 += tmp_tr_hr1
                    tr_hr3 += tmp_tr_hr3
                    tr_hr20 += tmp_tr_hr20
                    tr_hr50 += tmp_tr_hr50
                    tr_mrr10 += tmp_tr_mrr10
                    tr_mrr20 += tmp_tr_mrr20
                    tr_mrr50 += tmp_tr_mrr50
                    tr_ndcg10 += tmp_tr_ndcg10
                    tr_ndcg20 += tmp_tr_ndcg20
                    tr_ndcg50 += tmp_tr_ndcg50
                tr_loss = tr_loss / (step + 1)
                tr_hr1 = tr_hr1 / (step + 1)
                tr_hr3 = tr_hr3 / (step + 1)
                tr_hr20 = tr_hr20 / (step + 1)
                tr_hr50 = tr_hr50 / (step + 1)
                tr_mrr10 = tr_mrr10 / (step + 1)
                tr_mrr20 = tr_mrr20 / (step + 1)
                tr_mrr50 = tr_mrr50 / (step + 1)
                tr_ndcg10 = tr_ndcg10 / (step + 1)
                tr_ndcg20 = tr_ndcg20 / (step + 1)
                tr_ndcg50 = tr_ndcg50 / (step + 1)
            else:
                for step, batch_pos in enumerate(train_dataloader_pos, 0):
                    tmp_tr_loss, _, _ = train(model, optimizer,batch_pos[0],aliasTable, device, train_eval=TRAIN_EVAL)
                    tr_loss += tmp_tr_loss
                tr_loss = tr_loss / (step + 1)

            # Evaluate per 1 epoch
            if epoch % 1 == 0:
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

                for step, val_batch_pos in enumerate(val_dataloader_pos, 0):
                    tmp_val_loss, tmp_val_hr1, tmp_val_hr3, tmp_val_hr20,tmp_val_hr50,tmp_val_mrr10,tmp_val_mrr20,tmp_val_mrr50, tmp_val_ndcg10,tmp_val_ndcg20,tmp_val_ndcg50 = val(model, val_batch_pos[0], aliasTable,device)
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
                val_loss = val_loss / (step + 1)
                val_hr1 = val_hr1 / (step + 1)
                val_hr3 = val_hr3 / (step + 1)
                val_hr20 = val_hr20 / (step + 1)
                val_hr50 = val_hr50 / (step + 1)
                val_mrr10 = val_mrr10 / (step + 1)
                val_mrr20 = val_mrr20 / (step + 1)
                val_mrr50 = val_mrr50 / (step + 1)
                val_ndcg10 = val_ndcg10 / (step + 1)
                val_ndcg20 = val_ndcg20 / (step + 1)
                val_ndcg50 = val_ndcg50 / (step + 1)

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
                for step, test_batch_pos in enumerate(test_dataloader_pos, 0):
                    tmp_test_loss, tmp_test_hr1, tmp_test_hr3, tmp_test_hr20,tmp_test_hr50,tmp_test_mrr10,tmp_test_mrr20,tmp_test_mrr50, tmp_test_ndcg10,tmp_test_ndcg20,tmp_test_ndcg50 = test(model, test_batch_pos[0], aliasTable, device, DEBUG=False)
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
                test_loss = test_loss / (step + 1)
                test_hr1 = test_hr1 / (step + 1)
                test_hr3 = test_hr3 / (step + 1)
                test_hr20 = test_hr20 / (step + 1)
                test_hr50 = test_hr50 / (step + 1)
                test_mrr10 = test_mrr10 / (step + 1)
                test_mrr20 = test_mrr20 / (step + 1)
                test_mrr50 = test_mrr50 / (step + 1)
                test_ndcg10 = test_ndcg10 / (step + 1)
                test_ndcg20 = test_ndcg20 / (step + 1)
                test_ndcg50 = test_ndcg50 / (step + 1)

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
                user_embedding, item_embedding = model.get_model_embedding()
                user_embedding_norm = np.linalg.norm(user_embedding, axis=1).mean()
                item_embedding_norm = np.linalg.norm(item_embedding, axis=1).mean()
                writer.add_scalar('scalar/model/user_embedding_norm', user_embedding_norm, epoch)
                writer.add_scalar('scalar/model/item_embedding_norm', item_embedding_norm, epoch)

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

                epoch_end = datetime.now()
                print('Evaluate Time {} minutes'.format((epoch_end - epoch_start).total_seconds() / 60))
                if TRAIN_EVAL:
                    tr_log = 'Epoch: {:03d}, tr HR1: {:.4f}, tr HR3: {:.4f}, tr HR20: {:.4f}, tr HR50: {:.4f}, tr MRR10: {:.4f}, tr MRR20: {:.4f}, tr MRR50: {:.4f}, tr NDCG10: {:.4f}, tr NDCG20: {:.4f}, tr NDCG50: {:.4f}'
                    print(tr_log.format( epoch, float(tr_hr1), float(tr_hr3), float(tr_hr20), float(tr_hr50), float(tr_mrr10), float(tr_mrr20), float(tr_mrr50), float(tr_ndcg10), float(tr_ndcg20), float(tr_ndcg50)))
                val_log = 'Epoch: {:03d}, val HR1: {:.4f}, val HR3: {:.4f}, val HR20: {:.4f}, val HR50: {:.4f}, val MRR10: {:.4f}, val MRR20: {:.4f}, val MRR50: {:.4f}, val NDCG10: {:.4f}, val NDCG20: {:.4f}, val NDCG50: {:.4f}'
                print(val_log.format(epoch, float(val_hr1), float(val_hr3), float(val_hr20), float(val_hr50),
                                     float(val_mrr10), float(val_mrr20), float(val_mrr50), float(val_ndcg10),
                                     float(val_ndcg20), float(val_ndcg50)))

                test_log = 'Epoch: {:03d}, test HR1: {:.4f}, test HR3: {:.4f}, test HR20: {:.4f}, test HR50: {:.4f}, test MRR10: {:.4f}, test MRR20: {:.4f}, test MRR50: {:.4f}, test NDCG10: {:.4f}, test NDCG20: {:.4f}, test NDCG50: {:.4f}'
                print(test_log.format(epoch, float(test_hr1), float(test_hr3), float(test_hr20), float(test_hr50),
                                      float(test_mrr10), float(test_mrr20), float(test_mrr50), float(test_ndcg10),
                                      float(test_ndcg20), float(test_ndcg50)))

                best_test_log = 'Epoch: {:03d}, Best Test HR1: {:.4f}, Test HR3: {:.4f}, Test HR20: {:.4f}, Test HR50: {:.4f}, Test MRR10: {:.4f}, Test MRR20: {:.4f}, Test MRR50: {:.4f}, Test NDCG10: {:.4f}, Test NDCG20: {:.4f},Test NDCG50: {:.4f}'
                print(best_test_log.format(epoch, float(best_test_hr1), float(best_test_hr3), float(best_test_hr20), float(best_test_hr50), float(best_test_mrr10), float(best_test_mrr20), float(best_test_mrr50), float(best_test_ndcg10),float(best_test_ndcg20),float(best_test_ndcg50)))

            scheduler.step()
            if early_stop > EARLY_STOP:
                print("================================================")
                info = 'EARLY STOP:\nAt Epoch: {:03d}'
                print(info.format(epoch))
                print("================================================")

                logfile.write(
                    'Lr:' + str(LEARNING_RATE) + ',Lamb:' + str(LAMBDA) + ', metrics:' + best_test_log.format(epoch,
                         float(best_test_hr1),float(best_test_hr3),float(best_test_hr20),float(best_test_hr50),float(best_test_mrr10),float(best_test_mrr20),float(best_test_mrr50),float(best_test_ndcg10),float(best_test_ndcg20),float(best_test_ndcg50)) + '\n')

                logfile.close()
                logfile = open('./BMF_search_log_neg_new', 'a')
                break

            epoch_start = datetime.now()
        run = run + 1


logfile.close()

if SAVE_EMBEDDING:
    user_emb, item_emb = model.get_model_embedding()
    np.save('MF_userEmb64.npy', user_emb)
    np.save('MF_itemEmb64.npy', item_emb)