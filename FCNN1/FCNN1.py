#算法中所有列表均储存下标
#代表元素选择原则为： S的每一点其Voren集合中距离该点最近的点
#此算法为原始版本，未添加任何优化

from basic_function import *

def FCNN1(train_set, train_lable):
    #训练集的所有下标
    set_T = [i for i in range(len(train_set))]
    #初始化最近邻列表 nearest[5]  = 120,  点5在集合S中的最近邻是120
    nearest =  [-1] * len(train_set)
    #子集S
    set_S = []
    #△S, 初始化是所有类的质心
    set_Si = get_centroid_index(train_set, train_lable, set_T)
    
    loop_number = 0
    
    #主循环
    while len(set_Si) > 0:
        loop_number += 1
        
        # S∪△S
        set_S.extend(set_Si)
        
        #初始化rep数组，集合S中当前元素的代表元素
        # rep[325] =  ?    不能单纯的使用列表, 使用字典好一些
        rep = dict()        
        for  i in set_S:
            rep[i] = None

        #集合 T - S
        T_S = [item for item in set_T if item not in set_S]
        for q in T_S:
            #更新 T-S 集合中点在S中的最近点
            for p in set_Si:
                if d(nearest[q],q, train_set) > d(p,q, train_set):
                    nearest[q] = p
            # l(q) != l(nearest[q])  --->  判断该点是否处在 Voren集合中
            #如果是，则判断 d(nearest[q],q) < d(nearest[q], rep[nearest[q]])
            #FCNN1 中，代表元素挑选当前原则：为每个S中的元素p，挑选以p为最近邻 且 以标签不同类集合中，里p最近的一个
            #nearest[q]是 set_S中的点，rep[nearest[q]]是T-S中的点
            if train_lable[q] != train_lable[nearest[q]]:
                if d(nearest[q],q,train_set) < d(nearest[q], rep[nearest[q]],train_set):
                    rep[nearest[q]] = q
            
        set_Si.clear()       
        for p in set_S:
            if rep[p] != None :
                set_Si.append(rep[p])
          
    return set_S