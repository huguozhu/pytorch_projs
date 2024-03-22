# ========== 线性神经网络 ==========
import math
import time
import numpy as np
import torch
from d2l import torch as d2l
import random
from torch import nn

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
    
# ========== 3.1.3 计算正态分布(高斯分布) ==========
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
    
# ========== 3.2 线性回归--从零开始实现 ==========
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

# 定义线性回归模型
def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b
# 定义损失函数loss
def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
# 定义优化算法
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
  

# 使用自己的代码实现线性回归
def Linear_Regression_Scratch():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# ========== 3.3 线性回归的简洁实现(使用PyTorch的API) ==========
def Linear_Regression_Concise():    
    # 1) 生成数据集
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)
    
    # 2) 读取数据集
    # 使用PyTorch API来读取数据，功能同data_iter()函数
    def load_array(data_arrays, batch_size, is_train=True):  #@save
        """构造一个PyTorch数据迭代器"""
        dataset =  torch.utils.data.TensorDataset(*data_arrays)
        return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)   
    batch_size = 10
    data_iter = load_array((features, labels), batch_size)    
    print(next(iter(data_iter)))
    
    # 3) 定义模型
    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    
    # 4) 定义损失函数 & 定义优化算法
    #   计算均方误差使用MSELoss类，返回所有样本损失的平均值
    #   优化算法：小批量随机梯度下降算法
    loss = nn.MSELoss   
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)   
    
    # 5) 训练
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter : 
            print("X = ", X)
            print("y = ", y)
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

    w = net[0].weight.data
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的估计误差：', true_b - b)
    
# Run
#Test_Vectorization_for_Speed()
#Test_Normal()
#Test_data_iter
#Linear_Regression_Scratch()
Linear_Regression_Concise()



            