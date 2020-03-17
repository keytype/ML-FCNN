#算法中所有列表均储存下标
#代表元素选择原则为： S的每一点其Voren集合中距离该点最近的点
#改良的FCNN1算法：1. 储存距离，减少重复的距离计算； 
# 2. 使用三角不等式，避免不必要的计算。
#此处使用了 1 和 2 
#nearest数组储存后继结点（即是一个数组链表），以此避免内存占用

from basic_function import *

#函数作用： 常规方法计算和更新该点的最近邻以及距离，暂时不考虑KNN
#传入参数：点t, 最近邻列表， 新增点集合set_Si， 训练集train_set
#head, 为set_S中每个点存储头部（可以用字典）
#返回值：无
def update_distance_by_normal_Vor(nearest, head, before, set_S, set_Si, train_set):
    #print("进入了update_distance_by_normal_Vor函数！")
    for t in range(len(train_set)):
        #跳过已加入的点
        #if t in set_Si + set_S:
        #    continue
            
        #set_S为空的情况，既第一次
        t_neighbors_list = nearest[t]  
        t_nearest = None
        for p in set_Si:
            #找出其最近邻
            t_neighbor_distance = t_neighbors_list[0]
            d_p_t =  d(p, t, train_set)
            if d_p_t < t_neighbor_distance:
                t_neighbors_list[0] = d_p_t
                t_nearest = p
                
        if t_nearest != None:
            if head[t_nearest] == None:
                head[t_nearest] = t
                before[t_nearest] = t
            else:
                bt = before[t_nearest]
                nearest[bt][1] = t
                before[t_nearest] = t
                
    return True

def update_distance_by_triangle_inequality_Vor(nearest, head, before, set_S, set_Si, train_set):
    #需要通过引导Vor集合来更新
    for p in set_S:
        p_head = head[p]
        if p_head != None:
            q = p_head
            #计算S中点p与△S中点的距离并排序
            p_to_Si = []
            for i in set_Si:
                d_p_i = d(p, i, train_set)
                p_to_Si.append((i,d_p_i))
                
            p_to_Si.sort(key=operator.itemgetter(1))
            
            #q是开头元素，它的上一个元素为None
            q_before = None
            while q != None:                
                q_next = nearest[q][1]
                d_p_q = nearest[q][0]

                #两倍p到q的距离
                d_p_q2 = d_p_q * 2
                q_nearest_neighbor = None
           
                for t in p_to_Si:
                    #满足三角不等式，计算距离
                    if t[1] < d_p_q2:
                        d_p_t = d(q, t[0], train_set)
                        #比较是否是q的更近点，如果是则更新
                        if d_p_t < nearest[q][0]:
                            nearest[q][0] = d_p_t
                            q_nearest_neighbor = t[0]
                    else:
                        break

                #新增的点里面有q更近的点
                if q_nearest_neighbor != None:                        
                    #为新的点更新链表：增添元素
                    if head[q_nearest_neighbor] == None:
                        head[q_nearest_neighbor] = q
                        before[q_nearest_neighbor] = q
                        
                    else:
                        #更新上一个的后继
                        bq = before[q_nearest_neighbor]
                        nearest[bq][1] = q
                        #存储这个后继，为下次更新准备
                        before[q_nearest_neighbor] = q  
                    
                    #清除原来的后续指向
                    nearest[q][1] = None
                    
                    #为旧的链表更新关系
                    if q_before == None:                        
                        head[p] = q_next
                    else:
                        nearest[q_before][1] = q_next
                else:
                    q_before = q
                q = q_next
       
    return True
    
def FCNN1_Vor(train_set, train_lable):
    k = 1
    #训练集的所有下标
    set_T = [i for i in range(len(train_set))]
    #类别
    classzz = set(train_lable)
 
    #初始化最近邻列表 nearest[5]  = [距离，下一点],  点5在集合S中的最近邻是k个元素
    inner = [float("inf"),None]
    nearest = [inner[:] for i in range(len(train_set))]
    
    #子集S
    set_S = []
    #△S, 初始化是所有类的质心
    set_Si = get_centroid_index(train_set, train_lable, set_T)
    
    #input("初始化完成，按回车进入主循环：")
    
    #记录循环次数
    loop_number = 0
    
    #头部指针
    head = dict()
    #上一个，用于更新上一个的后继点
    before = dict()
    
    #主循环
    while len(set_Si) > 0:
        loop_number += 1
        #添加新键值
        for i in  set_Si:
            head[i] = None
            before[i] = None

        #用数组链表生成vor集合    
        
        if len(set_S) >= k:
            update_distance_by_triangle_inequality_Vor(nearest, head, before, set_S, set_Si, train_set)
        else:
            update_distance_by_normal_Vor(nearest, head, before, set_S, set_Si, train_set)

        #使用KNN规则，获得当前点在S中的分类
        #k_lable = get_lable_by_k(t, nearest, classzz, train_lable, k)
        
        # S∪△S
        set_S.extend(set_Si)
        
        #初始化rep数组，集合S中当前元素的代表元素
        # rep[325] =  ?    不能单纯的使用列表, 使用字典好一些
        #储存列表，表示 该点 代表元素的下标以及距离
        rep = dict()        
        
        #统计元素
        total = 0
        #代表元素选择
        for p in set_S:
            #引导Voren集合
            h = head[p]
            p_lable = train_lable[p]
            rep[p] = [-1, float("inf")]
            p_rep = rep[p]
            while h != None:
                total += 1
                #以 p 为最近邻, 但是标签不同
                if train_lable[h] != p_lable:
                    d_h_p = nearest[h][0]                    
                    if d_h_p < p_rep[1]:
                        p_rep[0] = h
                        p_rep[1] = d_h_p
                    elif d_h_p == p_rep[1] and h < p_rep[0]:
                        p_rep[0] = h
                        #print("出现相等情况")
                        
                #下一个索引
                h = nearest[h][1]
                
        #print("第", loop_number,"次循环:", total)
        set_Si.clear()       
        for p in set_S:
            if rep[p][0] != -1 :
                set_Si.append(rep[p][0])    
    return set_S