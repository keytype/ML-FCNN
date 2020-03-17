import numpy as np
import operator

#函数作用：计算训练集p和q之间的距离
#传入参数：p,q两点的索引以及训练集train_set
#返回值：浮点类型的距离
def d(p, q, train_set):
    distance = float("inf")
    if p!=-1 and q!=-1:
        distance = np.linalg.norm(train_set[p] - train_set[q])
        
    return distance

#函数作用：计算集合set_S[st]中每个类别的质心
#传入参数：集合set_S, 标签lable_S, 取值下标列表
#返回值：centroid_list, 每个类别的质心下标
def get_centroid_index(set_S, lable_S, st):
     #类的种类, 使用set去重后转化为 list 方便操作
    classzs = list(set(lable_S[st]))
    
    #获取每个类的几何中心
    class_center = dict()
    closet = dict()
    for i in classzs:
        s1 = set_S[st][lable_S[st]==i]
        class_center[i] = s1.mean(0)
        closet[i] = [-1, float("inf")]
    
    for j in st:
        l = lable_S[j]
        vector1 = class_center[l]
        distance = np.linalg.norm(vector1 - set_S[j])
            
        if distance < closet[l][1]:
            closet[l][0] = j
            closet[l][1] = distance

    centroid_list = []
    for key in closet:
        centroid_list.append(closet[key][0])
        
    return centroid_list

#函数作用：获得集合S与集合△S中的距离，并由从小到大排序
#参数：集合set_S, 新增点的集合set_Si，以及训练集（用来计算距离）
#返回值：S_to_Si字典，以S中的每个点为键， （q,distance）
def get_S_to_Si(set_S, set_Si, train_set):
    S_to_Si = dict()
    for p in set_S :
        S_to_Si[p] = []
        for q in set_Si:
            S_to_Si[p].append((q, d(p, q, train_set)))

        #S_to_Si[p] 中的点以距离的升序排序
        S_to_Si[p].sort(key=operator.itemgetter(1))
    
    return S_to_Si

    
#函数作用： 使用三角不等式更新该点的最近邻以及距离（KNN和NN同样使用，只是初始化的nearest不同）
#传入参数：点t, 最近邻列表， S_to_Si字典， 训练集train_set
#返回值：无
def update_distance_by_triangle_inequality(t, nearest, S_to_Si, train_set):
    #点t当前最近点的距离
    #这里出现了问题，最近距离应该和 点 t 最新的距离比较
    #而d_t_p应该是恒定不变的
    p = nearest[t][-1][0] 
    d_t_p = nearest[t][-1][1]
    for y in S_to_Si[p]:
        # set_Si中点y到p的距离
        # d_t_nearest应该是要更新的
        d_y_p = y[1]
        d_t_nearest = nearest[t][-1]
        #满足三角不等式，计算距离
        if d_y_p < 2 * d_t_p:
            d_t_y = d(t, y[0], train_set)
            #比当前p点的距离跟小，则更新距离
            if d_t_y < d_t_nearest[-1]:
                #移除与增加 ---> 替换
                nearest[t].remove(d_t_nearest)
                nearest[t].append((y[0],d_t_y))
                #重新排序
                nearest[t].sort(key=operator.itemgetter(1))
        else:
            #由于排序的关系，接下来的距离也不满足三角不等式, 跳出本次循环
            break
    return True
    
#函数作用： 常规方法计算和更新该点的最近邻以及距离（KNN和NN同样使用，只是初始化的nearest不同）
#传入参数：点t, 最近邻列表， 新增点几何set_Si， 训练集train_set
#返回值：无
def update_distance_by_normal(t, nearest, set_Si, train_set):
    #set_S为空的情况，既第一次
    t_neighbors_list = nearest[t]            
    for p in set_Si:
        #t点的k个最近邻中，距离最大的
        max_t_neighbor = t_neighbors_list[-1]                
        d_p_t =  d(p, t, train_set)
        if d_p_t < max_t_neighbor[1]:
            t_neighbors_list.remove(max_t_neighbor)
            t_neighbors_list.append((p, d_p_t))
            #更新之后重新排序，从小到大排序
            t_neighbors_list.sort(key=operator.itemgetter(1))
    return True
    
#函数作用：使用KNN分类获得标签 等价于 NNk(p,S)
#传入参数：点t, 最近邻数组nearest,所有类标签集合classzz, 训练集标签train_lable(不需要传入k的原因：初始化过程中确定了k的大小)
#返回值：返回标签， None 表示无法通过kNN进行分类
def get_lable_by_k(t, nearest, classzz, train_lable, k):
    #k==1是最近邻规则
    '''
    k_lable = None
    if k == 1:
        k_lable = train_lable[nearest[t][0]]
        if k_lable == -1
            return None
        else:
            return k_lable
    '''
    #使用KNN规则，获得当前点在S中的分类
    classzz_length = [0] * len(classzz)
    lable_total = dict(zip(classzz,classzz_length))
    
    k_neighbors = nearest[t]
    for neighbor in k_neighbors:
        n = neighbor[0]
        #当S中点不足的时候，会有 -1 存在， 此时跳过
        if n != -1:
            lable_total[train_lable[n]] += 1
        else:
            break

    k_lable = None
    if n != -1:
        #统计最多的标签
        max_lable = 0
        for keys,value in lable_total.items():
            if value > max_lable:
                max_lable = value
                k_lable = keys
                
    return k_lable
    
#函数作用：获得点point在 集合 set_S[st] 中的最近邻
#传入参数：点point， 集合set_S， 以及set_S的取值下标列表
#返回值：point的最近邻索引

def get_nearest_neighbor(point, set_S, st):
    nearest = float("inf")
    neighbor = None
    
    for i in st:
        d = np.linalg.norm(point - set_S[i])
        if d < nearest:
            nearest = d
            neighbor = i
            
    return neighbor