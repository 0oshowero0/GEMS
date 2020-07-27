import networkx as nx
import numpy as np
import random
import copy

max_node = 7
max_edge = 10
max_try = 15


def mutation_init():

    id_map = ['U', 'B']
    meg_adj = [[0, 1], [0, 0]]

    path_exist_time = [[[0,1],1]]
    return id_map, meg_adj, path_exist_time, 1

# set the specific rule for Yelp dataset
def set_rule(list, id, cur_node):
    if cur_node == 'U':
        for i in range(len(id)):
            if id[i] == 'A' or id[i] == 'I':
                list[i] = -1
            else:
                list[i] = 0
    elif cur_node == 'B':
        for i in range(len(id)):
            if id[i] == 'O' or id[i] == 'B':
                list[i] = -1
            else:
                list[i] = 0
    elif cur_node == 'O':
        for i in range(len(id)):
            if id[i] != 'U':
                list[i] = -1
            else:
                list[i] = 0
    else:
        for i in range(len(id)):
            if id[i] != 'B':
                list[i] = -1
            else:
                list[i] = 0
    return list
def lower_bound(nums, target):
    low, high = 0, len(nums)-1
    pos = len(nums)
    while low<high:
        mid = (low+high)//2
        if nums[mid] < target:
            low = mid+1
        else:#>=
            high = mid
            #pos = high
    if nums[low]>=target:
        pos = low
    return pos
def test_connected(adj):
    copy_adj = copy.deepcopy(np.array(adj))
    count = 0
    for i in range(len(adj[0])):
        for j in range(len(adj[0])):
            if copy_adj[i][j] == -1:
                copy_adj[i][j] = 0
    G = nx.from_numpy_matrix(copy_adj, create_using=nx.Graph())
    paths = nx.has_path(G, 0, 1)
    path = nx.all_simple_paths(G, source=0, target=1)
    for i in path:
        count += 1
    if paths == True:

        return 1, count
    else:
        return 0, 0

def test_ub(adj, id, length, flag):
    path_list = []
    copy_adj = copy.deepcopy(adj)
    copy_adj = np.array(copy_adj)
    copy_id = copy.deepcopy(id)
    for i in range(len(copy_id)):
        for j in range(len(copy_id)):
            if copy_adj[i][j] == -1:
                copy_adj[i][j] = 0
    G = nx.from_numpy_matrix(copy_adj, create_using=nx.Graph())
    path = nx.all_simple_paths(G, source=0, target=1)
    for i in path:
        path_list.append(i)

        # print('path:' +str(paths))
    if len(path_list) > length:
            # print(str(id)+str(path_list))
        return 1, len(path_list), path_list

    else:
        return 0, len(path_list), path_list


def delete_path(pathlist,exist_time,x):
    if len(pathlist)!= len(exist_time):
        path_list=[]
        print('error delete')
        for item in exist_time:
            path_list.append(item[0])
        return [],exist_time,path_list
    path_num = len(pathlist)
    path_list = copy.deepcopy(pathlist)
    path_time = copy.deepcopy(exist_time)
    time_list = []  # 存每条path的存活时间
    edge_list = []
    same_list = []
    delete_list = []
    choice_edge_list = []
    count = 0
    sample_num = 1000000
    for i in range(path_num):
        choice_edge_list.append([])

    # transform path to edge list
    for item in path_list:
        for j in range(len(item)-1):
            edge_list.append((min(item[j],item[j+1]),max(item[j],item[j+1])))
            choice_edge_list[count].append((min(item[j],item[j+1]),max(item[j],item[j+1])))
        count += 1
    for item in edge_list:
        if edge_list.count(item) == 1:
            same_list.append(item)

    # 获取每条path的存活时间
    for item in path_time:
        time_list.append(item[1])
    time_exp = np.exp(np.array(time_list)*x)
    time_uniformed = np.round(time_exp/np.sum(time_exp),4)

    index = sample_num * time_uniformed
    for i in range(1,len(time_list)):
        index[i] = index[i]+index[i-1]

    max_index = np.max(index)
    random_num = random.randint(1, int(max_index))
    choice = lower_bound(index, random_num)
    path_choice = path_time[choice][0]
    # 改这个choice
    # choice = random.randint(0,len(path_list)-1)

    for item in choice_edge_list[choice]:
        if item in same_list:
            delete_list.append(item)

    path_time.pop(choice)
    if path_choice not in path_list:
        print('error')
    else:
        path_list.remove(path_choice)
    return delete_list, path_time, path_list

#test
# path_list = [[0,1],[0,2,3,1]]
# exist = [([0,1],2),([0,2,3,1],1)]
# list1, time = delete_path(path_list,exist,-8)

# 删除不在path里面的节点
def delete_outpath_nodes(adj, id, path_list,exist):
    copy_adj = copy.deepcopy(adj)
    copy_id = copy.deepcopy(id)
    copy_exist = copy.deepcopy(exist)
    node_list = []
    copy_id = list(copy_id)
    for i in range(len(id)):
        node_list.append(i)
    count = 0
    for i in range(len(id)):
        for paths in path_list:
            if i in paths:
                node_list.pop(i - count)
                count += 1
                break
    count = 0
    for k in node_list:
        copy_id.pop(k - count)
        copy_adj = np.delete(copy_adj, k - count, 1)
        copy_adj = np.delete(copy_adj, k - count, 0)

        count += 1

        for m in range(len(copy_exist)):
            index = 0
            for n in copy_exist[m][0]:
                if n >= k:
                    copy_exist[m][0][index] = n - 1
                index += 1
    return copy_id, copy_adj,copy_exist



def delete_isolated(adj, id,exist):
    copy_adj = copy.deepcopy(adj)
    true_adj = copy.deepcopy(adj)

    copy_adj = np.array(copy_adj)
    copy_id = copy.deepcopy(id)
    copy_exist = copy.deepcopy(exist)
    for i in range(len(true_adj[0])):
        for j in range(len(true_adj[0])):
            if copy_adj[i][j] == -1:
                copy_adj[i][j] = 0
    G = nx.from_numpy_matrix(copy_adj, create_using=nx.Graph())
    count = 0
    for k, v in G.degree:

        if v == 0:
            if k == 0 or k == 1:
                return copy_id,true_adj
            else:
                copy_id.pop(k - count)
                true_adj = np.delete(true_adj, k - count, 1)
                true_adj = np.delete(true_adj, k - count, 0)

                for m in range(len(copy_exist)):
                    index = 0
                    for n in copy_exist[m][0]:
                        if n >= k:
                            copy_exist[m][0][index]=n-1
                        index+=1
                # print('true adj' + str(true_adj))
                count += 1
    # print("delete isolated node")
    return copy_id, true_adj,copy_exist

def count_adj(adj):
    copy_adj = copy.deepcopy(np.array(adj))
    edge=0
    sum=0
    for i in range(len(adj[0])):
        for j in range(len(adj[0])):
            if j>i:
                if copy_adj[i][j] == 1:
                    edge += 1
                    sum += 1
                elif copy_adj[i][j] == 0:
                    sum += 1
    return edge,sum



def mutate_graph(candidates, stable_prob=0.2, complex_prob=0.6, add_node_prob=0.2, x = 1):

    mutated_cand_id = copy.deepcopy(candidates[0])
    mutated_cand_adj = copy.deepcopy(candidates[1])

    path_exist_time = copy.deepcopy(candidates[2])
    old_path_len = copy.deepcopy(candidates[3])



    choice = random.random()
    qualify = 0
    # choice = 1
    # 开始变异
    if choice > stable_prob:
        del_flag = 0
        # 控一下在边少于1的时候 不进行删除操作
        if old_path_len < 2:
            com_choice =  random.uniform(0, complex_prob)
        else:
            com_choice = random.random()
        #com_choice = 1
        # 变复杂
        if com_choice <= complex_prob:
            while qualify == 0:
                edge_num, total_num = count_adj(mutated_cand_adj)
                if len(mutated_cand_id)< max_node:
                    add_choice = random.random()
                elif len(mutated_cand_id) >= max_node and edge_num < total_num:
                    add_choice = add_node_prob
                else:
                    if qualify == 0:
                        # print('fail mutation')
                        mutated_cand_id = ['U', 'B']
                        mutated_cand_adj = [[0, 1], [0, 0]]
                        exist_time = [[[0, 1], 1]]
                        return mutated_cand_id, mutated_cand_adj, exist_time, 1
                # 加节点
                if add_choice < add_node_prob:
                    new_node = random.choice('UBOAI')
                    mutated_cand_id.append(new_node)

                    new_row = np.zeros(len(mutated_cand_id) - 1)
                    mutated_cand_adj = np.row_stack((mutated_cand_adj, new_row))

                    new_column = set_rule(np.zeros(len(mutated_cand_id)), mutated_cand_id, new_node)
                    mutated_cand_adj = np.column_stack((mutated_cand_adj, new_column))

                    node_flag = 0
                    count = 0
                    mutate_edge = random.randint(0, len(mutated_cand_id) - 2)
                    while node_flag != 1 or count > 100:
                                count += 1
                                # allowed to add
                                if mutated_cand_adj[mutate_edge][len(mutated_cand_id) - 1] == 0:
                                    mutated_cand_adj[mutate_edge][len(mutated_cand_id) - 1] = 1

                                    node_flag = 1
                                # fail to add -- repeat
                                else:
                                    node_flag = 0
                                    mutate_edge = random.randint(0, len(mutated_cand_id) - 2)
                # 加边
                else:
                    mutate_edge1 = random.randint(0, len(mutated_cand_id) - 2)
                    mutate_edge2 = random.randint(mutate_edge1 + 1, len(mutated_cand_id) - 1)
                    edge_flag = 0
                    count = 0
                    while edge_flag == 0:
                        count += 1
                        if count > 100:
                            break

                        if mutated_cand_adj[mutate_edge1][mutate_edge2] == 0:
                            mutated_cand_adj[mutate_edge1][mutate_edge2] = 1

                            edge_flag = 1
                        else:
                            mutate_edge1 = random.randint(0, len(mutated_cand_id) - 2)
                            mutate_edge2 = random.randint(mutate_edge1 + 1, len(mutated_cand_id) - 1)
                            edge_flag = 0
                qualify, y, path_list = test_ub(mutated_cand_adj, mutated_cand_id, old_path_len, del_flag)

            mutated_cand_id, mutated_cand_adj,path_exist_time = delete_outpath_nodes(mutated_cand_adj, mutated_cand_id, path_list,path_exist_time)


        # 变简单
        else:
            qualify, y, path_list = test_ub(mutated_cand_adj, mutated_cand_id, old_path_len, del_flag)
            if len(path_list) != len(path_exist_time):
                print('input error')
                # print(path_list)
                # print(path_exist_time)
                # print(mutated_cand_adj)
            deleted_edge, path_exist_time, path_list = delete_path(path_list,path_exist_time,x)
            # delete edge
            for item in deleted_edge:
                mutated_cand_adj[item[0]][item[1]] = 0


            mutated_cand_id, mutated_cand_adj, path_exist_time= delete_isolated(mutated_cand_adj, mutated_cand_id, path_exist_time)
    qualify, y, path_list = test_ub(mutated_cand_adj, mutated_cand_id, old_path_len, 0)


    for item in path_list:
        add_flag = 1
        for i in range(len(path_exist_time)):
            if item in path_exist_time[i]:
                path_exist_time[i][1]= int(path_exist_time[i][1])+1
                add_flag = 0
                break
            else:
                continue
        if add_flag == 1:
            path_exist_time.append([item,1])

    # delete exist_edge_time matrix
    for edge_path in path_exist_time:
        p = edge_path[0]
        if p not in path_list:
            path_exist_time.remove(edge_path)
    #print(mutated_cand_id)
    #print(mutated_cand_adj)
    #print(path_exist_time)
    return mutated_cand_id, mutated_cand_adj, path_exist_time, y

def expend_matrix(len, matrix):
    stack1 = np.zeros(len - 1)
    stack2 = np.zeros(len)
    matrix = np.row_stack((matrix, stack1))
    matrix = np.column_stack((matrix, stack2))
    return matrix


def construct_choice(id, loc):
    choice = []
    for i in range(len(id)):
        if (i != loc):
            choice.append(id[i])
    return choice





