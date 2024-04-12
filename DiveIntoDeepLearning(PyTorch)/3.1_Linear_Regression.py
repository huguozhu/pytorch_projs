# ========== 概念 ==========
# 数学期望: 随机变量的平均值
# 方差：每个样本值预全体样本平均数差的平方的平均值
# 概率密度 = 一段区间（事件的取值范围）的概率除以该区间的长度
import math
import time
import numpy as np
import torch
from d2l import torch as d2l
from torch.utils import data
import random
from torch import nn
from matplotlib import pyplot as plt


# ========== 3.1 线性回归 ==========
def v31_Linear_Regression() :
    # 3.1.2 矢量化加速: 使用矢量计算，比用for速度快很多
    print("==== 矢量化加速 ====")
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
    print("用for 消耗时间=", t1)
    print("矢量化加速消耗时间=", t2)
        
    # 3.1.3 计算正态分布(高斯分布)
    # param: 
    #   x:      数据源
    #   mu:     数学期望
    #   sigma:  标准差    
    def Calc_Normal(x, mu, sigma) :
        p = 1 / math.sqrt(2 * math.pi * sigma**2)
        return p * np.exp(-0.5 * (x - mu)**2 / sigma**2)

    # 生成[-7,7],间距为0.01的数列
    x = np.arange(-7, 7, 0.01)
    # Mean and standard deviation pairs
    params = [(0, 1, 'red'), (0, 2, 'green'), (3, 1, 'blue')]
    for mu, sigma, color in params :
        plt.plot(x, Calc_Normal(x, mu, sigma), color=color, linewidth=3, label=[f'mean={mu}, std={sigma}'])
    plt.legend(ncol=1)
    plt.title('Gaussian Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()
    x = x
    


    
# ========== 3.2 线性回归--从零开始实现 ==========
# 3.2.1 生成数据集
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))      # 返回一组符合高斯分布的概率密度随机数
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
# 3.2.2 读取数据集
# 该函数接收批量大小、特征矩阵和标签向量作为输入，
# 生成大小为batch_size的小批量。 每个小批量包含一组特征和标签
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]        

# 3.2.4 定义线性回归模型: Y = w*X + b
def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b
# 3.2.5 定义损失函数loss
def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
# 3.2.6 定义优化算法
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
  
# 使用自己的代码实现线性回归
def Linear_Regression_Scratch():
    # 1.生成数据集    
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    # 2.读取数据集
    batch_size = 10
    for X, y in data_iter(batch_size, features, labels):
        print(X, '\n', y)
        break    
    # 3.定义回归模型: 使用线性回归,并初始化参数
    w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    # 4.训练
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # X和y的小批量损失
            # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
            # 并以此计算关于[w,b]的梯度
            l.sum().backward()
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    # 5.结果
    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差: {true_b - b}')




# ========== 3.3 线性回归的简洁实现(使用PyTorch的API) ==========
def Linear_Regression_Concise():    
    # 1.生成数据集
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)    
    # 2.读取数据集
    def load_array(data_arrays, batch_size, is_train=True):  #@save
        """构造一个PyTorch数据迭代器"""
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)
    batch_size = 10
    data_iter = load_array((features, labels), batch_size)    
    print(next(iter(data_iter)))    
    # 3.定义模型
    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)    
    # 4.定义损失函数
    # 计算均方误差使用MSELoss类，返回所有样本损失的平均值 -- 默认用于计算两个输入对应元素差值平方和的均值    
    loss = nn.MSELoss
    # 5.定义优化算法--SGD优化算法：随机梯度下降算法
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)       
    # 6.训练
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter :
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')
    # 7.结果
    w = net[0].weight.data
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的估计误差：', true_b - b)
    


# ========== main ==========
if __name__ == '__main__':
    #v31_Linear_Regression()
    Linear_Regression_Scratch()
    #Linear_Regression_Concise()



            