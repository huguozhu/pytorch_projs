# ========== 定义 ==========
# 卷积神经网络
#   主要用途：主要用于处理图像数据
#           图像识别、目标检测、语义分割


import torch
from torch import nn
from d2l import torch as d2l




# ========== 6.1 从全连接层到卷积(From Fully Connected Layers to Convolutions)  ========== 




# ========== 6.2 图像卷积(Convolutions for Images)  ========== 
def v62_ConvolutionsForImages() :
# 6.2.1 互相关运算(cross-correlation)
# 对二维图像数据X，做kernel为K的卷积运算
    def corr2d(X, K):  #@save
        """计算二维互相关运算"""
        h, w = K.shape
        Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
        return Y
# 6.2.2 卷积层：对输入和卷积权重(weight)进行互相关运算,并在添加标量偏置(bias)之后产生输出
    class Conv2D(nn.Module):
        def __init__(self, kernel_size):
            super().__init__()
            self.weight = nn.Parameter(torch.rand(kernel_size))
            self.bias = nn.Parameter(torch.zeros(1))
        def forward(self, x):
            return corr2d(x, self.weight) + self.bias

# 6.2.3 图像中目标的边缘检测
#   可以检测水平边缘的kernel：K = torch.tensor([[1.0, -1.0]])
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    print("X = ", X)
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)
    print("Y = ", Y)
# 6.2.4 学习卷积核：通过仅查看“输入-输出”对来学习由X生成Y的卷积核， 步骤
#   a) 构造一个随机生成的卷积核
#   b) 在每次迭代中，比较Y与卷积层输出的平方误差，然后计算梯度来更新卷积核
#   构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
    conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)
#   这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
#   其中批量大小和通道数都为1
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    print("After-reshape--X = ", X)
    print("After-reshape--Y = ", Y)
    lr = 3e-2  # 学习率
    for i in range(30):
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        l.sum().backward()
#       迭代卷积核
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        if (i + 1) % 2 == 0:
            print(f'epoch {i+1}, loss {l.sum():.3f}')
    conv2d.weight.data.reshape((1, 2))
#   查看这个weight是否接近我们的答案[1.0, -1.0]
    print(conv2d.weight)





# ========== 6.3 填充和步幅（Padding and Stride）  ========== 
def v63_PaddingAndStride() :
    # 为了方便起见，我们定义了一个计算卷积层的函数。
    # 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
    def comp_conv2d(conv2d, X):
        # 这里的（1，1）表示批量大小和通道数都是1W
        print("X.shape=", X.shape)
        X = X.reshape((1, 1) + X.shape)
        print("X.shape=", X.shape)
        Y = conv2d(X)
        # 省略前两个维度：批量大小和通道
        return Y.reshape(Y.shape[2:])
    # 6.3.1 填充
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    X = torch.rand(size=(8, 8))
    print("result.shape=", comp_conv2d(conv2d, X).shape)
    # 当卷积核的高度和宽度不同时，我们可以填充不同的高度和宽度
    # 使输出和输入具有相同的高度和宽度
    conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
    print("result.shape", comp_conv2d(conv2d, X).shape)

    # 6.3.2 步幅
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
    comp_conv2d(conv2d, X).shape
    # 宽高处步幅不同
    conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
    print(comp_conv2d(conv2d, X).shape)





# ========== 6.4 多输入多输出通道  ========== 
def v64_Multiple_Input_Output_Channel() :
    # 6.4.1 多输入通道
    print(" ============ 6.4.1 ============")
    def corr2d_multi_in(X, K):
        # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
        return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
    X = torch.tensor([[[0.0, 1.0, 2.0],
                       [3.0, 4.0, 5.0], 
                       [6.0, 7.0, 8.0]],
                    [[1.0, 2.0, 3.0], 
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0]]])
    K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], 
                      [[1.0, 2.0], [3.0, 4.0]]])
    print("multi_input_output_value=", corr2d_multi_in(X, K))

    # 6.4.2 多输出通道
    print(" ============ 6.4.2 ============")
    def corr2d_multi_in_out(X, K):
        # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
        # 最后将所有结果都叠加在一起
        return torch.stack([corr2d_multi_in(X, k) for k in K], 0)
    K = torch.stack((K, K + 1, K + 2), 0)
    print("K=", K)
    print("K.shape=", K.shape)
    print("output=",  corr2d_multi_in_out(X, K))
    
    # 6.4.3 1x1卷积层
    print(" ============ 6.4.3 ============")
    def corr2d_multi_in_out_1x1(X, K):
        c_i, h, w = X.shape
        c_o = K.shape[0]
        X = X.reshape((c_i, h * w))
        K = K.reshape((c_o, c_i))
        # 全连接层中的矩阵乘法
        Y = torch.matmul(K, X)
        return Y.reshape((c_o, h, w))
    X = torch.normal(0, 1, (3, 3, 3))
    K = torch.normal(0, 1, (2, 3, 1, 1))

    Y1 = corr2d_multi_in_out_1x1(X, K)
    Y2 = corr2d_multi_in_out(X, K)
    assert float(torch.abs(Y1 - Y2).sum()) < 1e-6



# ========== 6.5 汇聚层（Pooling）  ========== 
# 目的：
#   1) 降低卷积层对位置的敏感性
#   2) 降低对空间采样表示的敏感性
# 种类：
#   1) 最大汇聚层
#   2) 平均汇聚层
def v65_Pooling() :
    def pool2d(X, pool_size, mode='max'):
        p_h, p_w = pool_size
        Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                if mode == 'max':       # 最大汇聚层
                    Y[i, j] = X[i: i + p_h, j: j + p_w].max()
                elif mode == 'avg':     # 平均汇聚层
                    Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
        return Y
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    print(pool2d(X, (2, 2)))
    
    # 6.5.2 填充和步幅
    X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
    print(X)
    # 默认情况下，深度学习框架中的步幅与汇聚窗口的大小相同
    pool2d = nn.MaxPool2d(3)
    print(pool2d(X))
    # 手动设置填充和步幅
    pool2d = nn.MaxPool2d(3, padding=1, stride=2)
    print(pool2d(X))
    # 可以设定一个任意大小的矩形汇聚窗口，并分别设定填充和步幅的高度和宽度
    pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
    print(pool2d(X))

    # 6.5.3 多个通道
    # 汇聚层在每个输入通道上单独运算，而不是像卷积层一样在通道上对输入进行汇总
    X = torch.cat((X, X + 1), 1)
    print(X)

# ========== 6.6 卷积神经网络（LeNet）  ========== 
# 包括两部分组成：
#   1. 卷积编码器：由两个卷积层组成
#   2. 全连接层密集块：由三个全连接层组成
def v66_LeNet() :
    #(输入图片28x28) --> (5x5卷积层，填充2) --> (2x2平均汇聚层，步幅2) --> (5x5卷积层) -->
    #(2x2平均汇聚层，步幅2) --> (全连接层120) --> (全连接层84) --> (全连接层10)
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.Sigmoid(),
        nn.Linear(84, 10))
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t',X.shape)
    def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
        """使用GPU计算模型在数据集上的精度"""
        if isinstance(net, nn.Module):
            net.eval()  # 设置为评估模式
            if not device:
                device = next(iter(net.parameters())).device
        # 正确预测的数量，总预测的数量
        metric = d2l.Accumulator(2)
        with torch.no_grad():
            for X, y in data_iter:
                if isinstance(X, list):
                    # BERT微调所需的（之后将介绍）
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                metric.add(d2l.accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]

    #@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    lr, num_epochs = 0.9, 10
    train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())




# ========== main ==========
if __name__ == '__main__':
    #v62_ConvolutionsForImages()
    #v63_PaddingAndStride()
    #v64_Multiple_Input_Output_Channel()
    v65_Pooling()
    #v66_LeNet()

