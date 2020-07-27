import h5py
import numpy as np
import os
import pandas as pd
from numpy.random import seed, random, randint
import pickle

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
        u = random(count)
        j = randint(self.keyLen, size=count)
        k = np.where(u <= self.prob[j], j, self.inx[j])
        return self.keys[k] 


#读取rating
f = open('../yelp_data/user_business.dat','r')
user_business_rating = []
for line in f:
    user,business,rating = line.strip().split('\t')
    user_business_rating.append([int(user),int(business),int(rating)])
f.close()
user_business_rating = np.array(user_business_rating) - 1
user_business_rating[:,1] = user_business_rating[:,1] + user_business_rating[:,0].max() + 1


#选择评分大于3的作为正样本，但发现太少了
#pos_data = user_business_rating[user_business_rating[:,2]>3,0:2]
pos_data = user_business_rating[user_business_rating[:,2]>-1,0:2]

item_records = pd.DataFrame(np.concatenate((pos_data[:,1].reshape(-1,1),np.ones((pos_data[:,1].shape[0],1))),axis=1),columns=['item_id','count'])
item_frequency = item_records.groupby('item_id',as_index = False).sum().sort_values('count',ascending=False)
item_frequency['count'] = item_frequency['count'] / item_frequency['count'].sum()


aliasTable = AliasTable(weights=item_frequency['count'].values, keys=item_frequency['item_id'].values.astype(int))

train_sample_index_pos = random(pos_data.shape[0]) < 0.8
train_pos = pos_data[train_sample_index_pos]
np.random.shuffle(train_pos)
pos_data_val_test = pos_data[~train_sample_index_pos]
val_sample_index_pos = random(pos_data_val_test.shape[0]) < 0.5
val_pos = pos_data_val_test[val_sample_index_pos]
np.random.shuffle(val_pos)
test_pos = pos_data_val_test[~val_sample_index_pos]
np.random.shuffle(test_pos)

user_num = pos_data[:,0].max() + 1
item_num = pos_data[:,1].max() + 1 - user_num
#转换为hdf5
dataset = './yelp_dataset.hdf5'
with h5py.File(dataset, 'w') as f:
    f.create_dataset('train_pos', data = train_pos)
    f.create_dataset('val_pos', data = val_pos)
    f.create_dataset('test_pos', data = test_pos)
    f.create_dataset('item_freq', data = item_frequency['count'].values)
    f.create_dataset('item_keys', data = item_frequency['item_id'].values.astype(int))
    f.create_dataset('user_num', data = user_num)
    f.create_dataset('item_num', data = item_num)


