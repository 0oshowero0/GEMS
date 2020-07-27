from transfer import list_trans_dic
import random
import time
import numpy as np
FALSE=666666
def control_nodes_num(prelist,max_nodes):
    if len(prelist)<=max_nodes:
        return prelist
    else:
        new_list=[]
        while len(new_list)<max_nodes:
            num = random.randint(0, len(prelist)-1)
            if prelist[num] not in new_list:
                new_list.append(prelist[num])
    return new_list
def find_type(cur,next):

    if cur=='U':
        if next=='U':
            return 'u2u'
        elif next=='B':
            return 'u2b'
        else:
            return 'u2o'
    elif cur=='B':
        if next=='U':
            return 'b2u'
        elif next=='A':
            return 'b2a'
        else:
            return 'b2i'


    elif cur=='O':
        return 'o2u'
    elif cur == 'A':
        return 'a2b'
    else:
        return 'i2b'


def find_next_node(adj,list,length):
    rank_list=np.zeros(length)
    for i in range(len(list)):
        for j in range(length):
            if adj[list[i]][j]==1 or adj[j][list[i]]==1 :
                rank_list[j]+=1
                rank_list[list[i]]+=1
    flag=0
    count=0
    max = np.argmax(rank_list)
    while flag==0:
        count+=1
        if max in list:
            rank_list[max] = 0
            max = np.argmax(rank_list)

        else:
            flag=1
        if count==100:
            return FALSE,FALSE
    return max,rank_list[max]

def calculate_length(diction):
    list1=['u2u','u2b','u2o','o2u','b2u','b2a','b2i','i2b','a2b']
    diction2={'u2u':{},'u2b':{},'u2o':{},'o2u':{},'b2u':{},'b2a':{},'b2i':{},'i2b':{},'a2b':{}}
    for rel in list1:
        for key in diction[rel].keys():
            diction2[rel][key]=len(diction[rel][key])
    return diction2

def delete_invalid_nodes(node_list,id_map,diction,next_list):
    for i in range(len(node_list)):
        for nodes in node_list[i]:
            nexts=next_list[i]

            for j in nexts:
                for k in node_list[j]:
                    conn_type=find_type(id_map[i],id_map[j])
                    if nodes in diction[conn_type].keys():
                        if k in diction[conn_type][nodes]:
                            continue
                        else:
                            node_list[i].remove(nodes)
                            break
                    else:
                        #print(nodes)
                        #print(conn_type)
                        node_list[i].remove(nodes)
                        break
                if nodes not in node_list[i]:
                    break
    return node_list

def match_graph(adj,mg_adj,id_map,relation_dict,max_sample_num):
    # result=open('find_result.txt','a+')
    # result.write('**************************************'+'\n')
    # result.write(str(id_map)+'\n')
    ub_pair_list=[]
    u2u, u2b, u2o, o2u, b2u, b2a, b2i, a2b, i2b= list_trans_dic(relation_dict)
    # u2b={0:[2,3,4],6:[2,3],7:[3]}
    # b2u={2:[0,6],3:[0,6,7],4:[0]}
    # u2o={0:[9],6:[8],7:[9]}
    # o2u={8:[6],9:[7,0]}
    # u2u,b2a,a2b,i2b,b2i=[],[],[],[],[]

    diction = {'u2u': u2u, 'u2b': u2b, 'u2o': u2o, 'o2u': o2u, 'b2u': b2u, 'b2a': b2a, 'b2i': b2i, 'i2b': i2b, 'a2b': a2b}

    diction2=calculate_length(diction)
    #start = time.clock()
    user_num=0
    user_count=0

    ub_pair = []
    for user in range(16239):
        searched_list=[0]

        old_list=[]
        next_list={}
        nodes_list=[[] for _ in range(len(id_map))]
        nodes_list[0].append(user)
        old_list.append(user)

        list_user = [x[0] for x in mg_adj]
        edge1 = np.where(np.array(mg_adj[0]) == 1)
        edge2 = np.where(np.array(list_user) == 1)
        edge = list(edge1[0]) + list(edge2[0])
        next_list[0] = edge

        for i in range(len(id_map)-1):
            nexts,edge_num=find_next_node(mg_adj,searched_list,len(id_map))
            #check validation
            if nexts==FALSE:
                break

            searched_list.append(nexts)
            list1 = [x[nexts] for x in mg_adj]
            edge1=np.where(np.array(mg_adj[nexts])==1)
            edge2 = np.where(np.array(list1) == 1)
            edge=list(edge1[0])+list(edge2[0])
            next_list[nexts]=edge
            for j in range(len(edge)):
                num=edge[j]
                conn_type=find_type(id_map[num],id_map[nexts])
                not_find = 0
                if nodes_list[num]!=[]:
                    if nodes_list[nexts] == []:
                        nodes_list[num]=control_nodes_num(nodes_list[num],max_sample_num)
                        sum_num=0
                        for node in nodes_list[num]:
                            if node in diction2[conn_type].keys():
                                sum_num=sum_num+diction2[conn_type][node]
                        if sum_num==0:
                            break
                        for node in nodes_list[num]:
                            if node in diction2[conn_type].keys():
                                sample_num = int((diction2[conn_type][node] / sum_num) * max_sample_num)

                                for k in range(min(diction2[conn_type][node], sample_num)):

                                    nums = random.choice(diction[conn_type][node])
                                    if nums not in old_list:
                                        nodes_list[nexts].append(nums)
                                        old_list.append(nums)
                        # 原来的都不和新增的节点匹配
                        if nodes_list[num] == []:
                            break

                    else:
                        conn_type1 = find_type(id_map[nexts], id_map[num])
                        for node in nodes_list[num]:
                            for next_node in nodes_list[nexts]:
                                if next_node in diction[conn_type1].keys():
                                    if node not in diction[conn_type1][next_node]:
                                        not_find += 1
                                else:
                                    nodes_list[nexts].remove(next_node)
                            if not_find==len(nodes_list[nexts]):
                                nodes_list[num].remove(node)
                                old_list.remove(node)


        #test validation
        # if f_flag!=len(id_map)-2:
        #     delete_invalid_nodes(nodes_list,id_map,diction,next_list)

        # 标记是否找到metagraph匹配
        flag=0
        for j in range(len(nodes_list)):
            if nodes_list[j]==[]:
                flag=1
        if flag==0:
            user_count+=1
            business_list=nodes_list[1]
            #print(str(user)+'\t'+str(len(business_list)))
            # print(str(user)+'\t'+str(len(business_list)))
            for b in business_list:
                #update pair
                ub_pair.append((user,b))
                ub_pair_list.append((user,b))
                #update adj
                business = b - 16239
                adj[user][business] += 1
            #ub_pair_list分别保存各个用户的连接
            #ub_pair_list.append(ub_pair)
        #else:
        #    print('user '+str(user)+' not found related business')
        #user_num+=1
        #if user_num% 10==0:
        #    elapsed = (time.clock() - start)
        #    print('10 users for ' + str(elapsed) + ' s')
        #    start = time.clock()

    #ub_pair_list = np.array(ub_pair_list)
    ub_pair = np.array(ub_pair)
    return adj, ub_pair, user_count
