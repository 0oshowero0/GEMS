import networkx as nx
import numpy as np
import copy
def b_idx(bussiness):
    return int(bussiness)+16238

def comp_idx(compliment):
    return int(compliment)+30522

def cat_idx(catagory):
    return int(catagory)+30533

def city_idx(city):
    return int(city)+30580




def list_trans_dic(relation_dict):

    u2u, u2b, u2o, b2a, b2i=relation_dict['u2u'],relation_dict['u2b'],relation_dict['u2o'],relation_dict['b2a'],relation_dict['b2i']
    u2u_dict={}
    for item in u2u:
        item=item.split('\t')
        temp=int(item[0])-1
        u2u_dict.setdefault(temp,[]).append(int(item[1])-1)

    u2b_dict = {}
    for item in u2b:
        item = item.split('\t')
        temp = int(item[0])-1
        num=b_idx(item[1])
        u2b_dict.setdefault(temp, []).append(num)

    b2u_dict = {}
    for item in u2b:
        item = item.split('\t')
        temp = int(item[0]) - 1
        num = b_idx(item[1])
        b2u_dict.setdefault(num, []).append(temp)


    u2o_dict = {}
    for item in u2o:
        item = item.split('\t')
        temp = int(item[0])-1
        num = comp_idx(item[1])
        u2o_dict.setdefault(temp, []).append(num)

    o2u_dict = {}
    for item in u2o:
        item = item.split('\t')
        temp = int(item[0]) - 1
        num = comp_idx(item[1])
        o2u_dict.setdefault(num, []).append(temp)



    b2a_dict = {}
    for item in b2a:
        item = item.split('\t')
        temp = b_idx(item[0])
        num = cat_idx(item[1])
        b2a_dict.setdefault(temp, []).append(num)

    a2b_dict = {}
    for item in b2a:
        item = item.split('\t')
        temp = b_idx(item[0])
        num = cat_idx(item[1])
        a2b_dict.setdefault(num, []).append(temp)

    b2i_dict = {}
    for item in b2i:
        item = item.split('\t')
        temp = b_idx(item[0])
        num = city_idx(item[1])
        b2i_dict.setdefault(temp, []).append(num)

    i2b_dict = {}
    for item in b2i:
        item = item.split('\t')
        temp = b_idx(item[0])
        num = city_idx(item[1])
        i2b_dict.setdefault(num, []).append(temp)

    return u2u_dict,u2b_dict,u2o_dict,o2u_dict,\
           b2u_dict,b2a_dict,b2i_dict,\
           a2b_dict,i2b_dict



