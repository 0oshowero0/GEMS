from searchGraph_2 import *
import time

import numpy as np
import h5py
import numpy as np




def construct_adj():
    dataset = './yelp_new_out.hdf5'
    with h5py.File(dataset, 'r') as f:

        user_link = f['train_pos'][:]

    adj=np.zeros((16239,14284))
    u2u=[]
    with open('yelp/user_user.dat') as file_object:
    #with open('Douban Movie/u2u-50.txt') as file_object:
        lines = file_object.readlines()
        for line in lines:
            u2u.append(line)


    u2b=list(user_link)
    for i in range(len(u2b)):
        u2b[i][1]=u2b[i][1]-16238
        u2b[i]=str(u2b[i][0] + 1)+'\t'+str(u2b[i][1])




    u2o = []
    with open('yelp/user_compliment.dat') as file_object:
        lines = file_object.readlines()
        for line in lines:
            u2o.append(line)

    b2a=[]
    with open('yelp/business_category.dat') as file_object:
        lines = file_object.readlines()
        for line in lines:
            b2a.append(line)

    b2i=[]
    with open('yelp/business_city.dat') as file_object:
        lines = file_object.readlines()
        for line in lines:
            b2i.append(line)

    relation_dict={'u2u':u2u,'u2b':u2b,'u2o':u2o,'b2a':b2a,'b2i':b2i}
    return adj,relation_dict



