# ========== 线性神经网络 ==========
import math
import time
import numpy as np
import torch
from d2l import torch as d2l
import random

# ========== 概念 ==========
# 数学期望: 随机变量的平均值
# 方差：每个样本值预全体样本平均数差的平方的平均值

# ========== 代码 ==========
# 3.1 线性回归 Linear Regression
# 3.1.2 矢量化加速: 使用矢量计算，比用for速度快很多
def Test_Vectorization_for_Speed() :
    n = 1000000
    a = torch.ones(n)
    b = torch.ones(n)
    c = torch.zeros(n)
    t = time.time()
    for i in range(n):
        c[i] = a[i] + b[i]
    t1 = f'{time.time() - t:.5f} sec'
    
    t = time.time()
    d = a + b
    t2 = f'{time.time() - t:.5f} sec'
    print("t1 = ", t1)
    print("t2 = ", t2)
    
# 3.1.3 计算正态分布(高斯分布)
# param: 
#   x:      数据源
#   mu:     数学期望
#   sigma:  标准差    
def Calc_Normal(x, mu, sigma) :
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 * (x - mu)**2 / sigma**2)
def Test_Normal():
    # Use NumPy again for visualization
    x = np.arange(-7, 7, 0.01)
    # Mean and standard deviation pairs
    params = [(0, 1), (0, 2), (3, 1)]
    d2l.plot(x, [Calc_Normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
             ylabel='p(x)', figsize=(4.5, 2.5),
             legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
    
# 3.2 线性回归--从零开始实现
# 3.2.1 生成数据集
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""    
    X = np.random.normal(0, 1, (num_examples, len(w)))      # 返回一组符合高斯分布的概率密度随机数
    y = np.dot(X, w) + b
    y += np.random.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

# 3.2.2 读取数据集
# 该函数接收批量大小、特征矩阵和标签向量作为输入，
# 生成大小为batch_size的小批量。 每个小批量包含一组特征和标签。    
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
def Test_data_iter():
    w = np.array([2, -3.4])
    b = 4.2
    features, labels = synthetic_data(w, b, 1000)
    batch_size = 10
    for X, y in data_iter(batch_size, features, labels):
        print(X, '\n', y)
        break




# Run
#Test_Vectorization_for_Speed()
#Test_Normal()
Test_data_iter