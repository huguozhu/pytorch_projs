# ========== 定义 ==========
# 多层感知机 ve 神经网络
#   定义：
#       多层感知机：是一种特殊的神经网络，它由输入层、一个或多个隐藏层和输出层组成
#       神经网络：是一个更广泛的概念，不仅包括多层感知机，还包括其他类型的网络，
#                如卷积神经网络（CNN）、循环神经网络（RNN）等
#   功能：
#       多层感知机：主要用于分类和回归问题，它通过在隐藏层中使用非线性激活函数，能够学习和表示复杂的模式。
#       神经网络：  应用范围更广，可以用于图像识别、自然语言处理、推荐系统等多个领域。


import torch
from torch import nn
from d2l import torch as d2l


# ========== 4.2 多层感知机的从零开始实现 ==========
def v42_Multilayer_preceptrons_Scratch() :
    # 加载数据
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 初始化模型参数
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    W1 = nn.Parameter(torch.randn(
        num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(
        num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    params = [W1, b1, W2, b2]

    # 4.2.2 激活函数
    def relu(X):
        a = torch.zeros_like(X)
        return torch.max(X, a)
    # 4.2.3 模型
    def net(X):
        X = X.reshape((-1, num_inputs))
        H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
        return (H@W2 + b2)
    # 4.2.4 损失函数
    loss = nn.CrossEntropyLoss(reduction='none')
    # 4.2.5 训练
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    # 测试
    d2l.predict_ch3(net, test_iter)








# ========== 4.3 多层感知机的简洁实现 ==========
def v43_Multilayer_preceptrons_Concise() :
    # 包含两个层：
    #   第一层：隐藏层，包含256个隐藏单元,并使用了ReLU激活函数
    #   第二层：输出层
    net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights);

    # 训练
    batch_size, lr, num_epochs = 256, 0.1, 10
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)    #新版d2l不支持这个函数


# ========== main ==========
v42_Multilayer_preceptrons_Scratch()
#v43_Multilayer_preceptrons_Concise()