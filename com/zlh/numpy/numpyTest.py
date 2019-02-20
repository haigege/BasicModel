import numpy as np
from numpy.linalg import *


def main():
    ## ndarray
    lst = [[1, 3, 5], [2, 4, 6]]
    print('集合类型： ', type(lst))
    
    np_lst = np.array(lst)
    print('集合类型： ', type(np_lst))
    
    np_lst = np.array(lst, dtype = np.float)
    #dtype的类型：bool, int, int8/18/32/64/128, uint8/16/32/64/128, float, float16/32/64/128, complex64/128
    
    print(np_lst.shape) #形状
    print(np_lst.ndim) #维度
    print(np_lst.dtype) #数据类型
    print(np_lst.itemsize) #大小（单位字节）
    print(np_lst.size) #元素数
    print('----------------')
    
    ## 2.some kinds of array
    print(np.zeros([2,4])) #初始化，值为0
    print(np.ones([2,4])) #初始化，值为1
    print(np.random.rand(2,4)) #随机数，[0, 1)，均匀分布
    print(np.random.rand()) #随机数，[0, 1)，无参，生成一个
    print(np.random.randint(1, 10, 3)) #随机整数，前两个参数表示范围，第三个表示生成的个数，均匀分布
    print(np.random.randn())#随机数，无参，生成一个，正态分布
    print(np.random.randn(2, 4))#随机数，无参，生成一个，正态分布
    print(np.random.choice([10, 20, 30]))#随机数，从指定集合中选取
    print(np.random.beta(1, 10, 100))#随机数，bate分布
    
    ## 3.Array operation
    print(np.arange(1, 11).reshape([2,-1])) #等差数列
    lst = np.arange(1, 11).reshape([2,-1])
    print(np.exp(lst)) # e（自然对数的底）的幂次方
    print(np.exp2(lst)) #2的幂次方
    print(np.sqrt(lst)) #开方
    print(np.sin(lst)) #正弦
    print(np.log(lst)) #对数
    
    lst = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]],
                    [[9, 10, 11, 12], [13, 14, 15, 16]],
                    [[17, 18, 19 ,20], [21, 22, 23, 24]]])
    
    print(lst.sum()) #求和
    print(lst.sum(axis = 0)) #第一维求和，即1+9+17， 2+10+18…… 
    print(lst.sum(axis = 1)) #第二维求和，即1+5， 2+6，3+7， 4+8…… 
    print(lst.sum(axis = 2)) #第三维求和，即1+2+3+4，5+6+7+8…… 
    
    print(lst.max()) #求最大
    print(lst.max(axis = 0)) #求第一维最大
    #……
    #类似还有min()
    
    lst1 = np.array([10, 20 ,30, 40])
    lst2 = np.array([4, 3, 2, 1])
    print(lst1 + lst2)
    print(lst1 - lst2)
    print(lst1 * lst2)
    print(lst1 / lst2)
    print(lst1 // lst2)
    print(lst1 ** 3)
    print(np.dot(lst1.reshape([2,2]), lst2.reshape([2,2]))) #矩阵相乘
    print(np.concatenate((lst1, lst2), axis = 0)) #追加
    print(np.vstack((lst1, lst2))) #垂直追加
    print(np.hstack((lst1, lst2))) #水平追加
    print(np.split(lst1, 2)) #拆分集合
    print(np.copy(lst1)) #拷贝
    
    ## 4.liner
    print(np.eye(3)) #单位矩阵
    lst = np.array([[1., 2.], 
                    [3., 4.]])
    print('------',inv(lst)) #逆矩阵
    print(lst.transpose()) #转置矩阵
    print(det(lst)) #行列式
    print(eig(lst)) #特征值 和 特征向量
    y = np.array([[5], [7]])
    print(solve(lst, y)) #解方程
    
    ## 5.other
    print(np.fft.fft([1, 1, 1, 1, 1, 1, 1, 1])) #傅里叶变换
    print(np.correlate([1, 0, 1], [0, 2, 1])) #相关系数
    print(np.poly1d([1, 2, 3])) #多项式函数 
    
if __name__ == '__main__':
    main()