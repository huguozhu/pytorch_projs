# ========== 定义 ==========
# 现代神经网络

import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F

# ========== 7.1. 深度卷积神经网络（AlexNet）  ========== 
# Alexnet由八层组成： 5个卷积层 + 两个全连接隐藏层和一个全连接输出层
def v71_AlexNet() :
    net = nn.Sequential(
        # 这里使用一个11*11的更大窗口来捕捉对象。
        # 同时，步幅为4，以减少输出的高度和宽度。
        # 另外，输出通道的数目远大于LeNet
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 使用三个连续的卷积层和较小的卷积窗口。
        # 除了最后的卷积层，输出通道的数量进一步增加。
        # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
        nn.Linear(6400, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
        nn.Linear(4096, 10))
    X = torch.randn(1, 1, 224, 224)
    print("input shape:\t", X.shape)
    for layer in net:
        X=layer(X)
        print(layer.__class__.__name__,'output shape:\t',X.shape)
    # 7.1.3 读取数据集
    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    # 7.1.4 训练AlexNet
    lr, num_epochs = 0.01, 10
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    




# ========== 7.2. 使用块的网络（VGG）   ========== 
def V72_VGG_11() :
    # 带块的CNN（卷积神经网络）有三部分组成：
    #   1) 带填充以保持分辨率的卷积层
    #   2) 非线性激活函数，如ReLU
    #   3) 汇聚层，如最大汇聚层，以降低分辨率
    # 函数包含三个参数，分别为：
    #   1) 卷积层的数量 num_convs
    #   2) 输入通道的数量in_channels
    #   3) 输出通道的数量out_channels
    def vgg_block(num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        return nn.Sequential(*layers)
    # 7.2.2 VGG网络
    # VGG-11： 使用8个卷积层 + 3个全连接层
    # 参数conv_arch: 每个VGG块中卷积层个数和输出通道个数
    def vgg(conv_arch):
        conv_blks = []
        in_channels = 1
        # 卷积层部分
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        return nn.Sequential(
            *conv_blks, nn.Flatten(),
            # 3个全连接层部分
            nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 10))
    
    
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    # 打印VGG-11层结构：8个卷积层+3个全连接层
    net = vgg(conv_arch)    
    X = torch.randn(size=(1, 1, 224, 224))
    for blk in net:
        X = blk(X)
        print(blk.__class__.__name__,'output shape:\t',X.shape)

    # 7.2.3 训练模型
    ratio = 4
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    net = vgg(small_conv_arch)
    lr, num_epochs, batch_size = 0.05, 10, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())




# ========== 7.3. 网络中的网络（NiN）   ========== 
def V73_NiN() : 
    # 定义NiN块
    def nin_block(in_channels, out_channels, kernel_size, strides, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), 
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), 
            nn.ReLU())
    net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2d(3, stride=2),
        nin_block(96, 256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2d(3, stride=2),
        nn.Dropout(0.5),
        # 标签类别数是10
        nin_block(384, 10, kernel_size=3, strides=1, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        # 将四维的输出转成二维的输出，其形状为(批量大小,10)
        nn.Flatten())

    # 答应NiN的输出
    X = torch.rand(size=(1, 1, 224, 224))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)

    # 训练
    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())





# ========== 7.4. 含并行连结的网络（GoogLeNet）   ========== 
def V74_GoogLeNet() :
    class Inception(nn.Module):
        # c1--c4是每条路径的输出通道数
        def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
            super(Inception, self).__init__(**kwargs)
            # 线路1，单1x1卷积层
            self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
            # 线路2，1x1卷积层后接3x3卷积层
            self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
            self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
            # 线路3，1x1卷积层后接5x5卷积层
            self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
            self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
            # 线路4，3x3最大汇聚层后接1x1卷积层
            self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

        def forward(self, x):
            p1 = F.relu(self.p1_1(x))
            p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
            p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
            p4 = F.relu(self.p4_2(self.p4_1(x)))
            # 在通道维度上连结输出
            return torch.cat((p1, p2, p3, p4), dim=1)
    
    # 第一个模块:使用64个通道、7x7卷积层
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # 第二个模块:使用两个卷积层--对应Inception块中的第二条路径
    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # 第三个模块：串联两个完整的Inception块
    b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # 第四个模块：串联5个Inception块
    b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # 第五个模块：两个Inception块
    b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())
    # 组合形成GoogLeNet
    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

    # 测试打印整个输入输出过程
    X = torch.rand(size=(1, 1, 96, 96))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)

    # 训练
    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    




# ========== 7.5. 批量规范化   ========== 
def V75_BatchNormal() :
    def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
        # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
        if not torch.is_grad_enabled():
            # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
            X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
        else:
            assert len(X.shape) in (2, 4)
            if len(X.shape) == 2:
                # 使用全连接层的情况，计算特征维上的均值和方差
                mean = X.mean(dim=0)
                var = ((X - mean) ** 2).mean(dim=0)
            else:
                # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
                # 这里我们需要保持X的形状以便后面可以做广播运算
                mean = X.mean(dim=(0, 2, 3), keepdim=True)
                var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
            # 训练模式下，用当前的均值和方差做标准化
            X_hat = (X - mean) / torch.sqrt(var + eps)
            # 更新移动平均的均值和方差
            moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
            moving_var = momentum * moving_var + (1.0 - momentum) * var
        Y = gamma * X_hat + beta  # 缩放和移位
        return Y, moving_mean.data, moving_var.data
    class BatchNorm(nn.Module):
        # num_features：完全连接层的输出数量或卷积层的输出通道数。
        # num_dims：2表示完全连接层，4表示卷积层
        def __init__(self, num_features, num_dims):
            super().__init__()
            if num_dims == 2:
                shape = (1, num_features)
            else:
                shape = (1, num_features, 1, 1)
            # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
            self.gamma = nn.Parameter(torch.ones(shape))
            self.beta = nn.Parameter(torch.zeros(shape))
            # 非模型参数的变量初始化为0和1
            self.moving_mean = torch.zeros(shape)
            self.moving_var = torch.ones(shape)

        def forward(self, X):
            # 如果X不在内存上，将moving_mean和moving_var
            # 复制到X所在显存上
            if self.moving_mean.device != X.device:
                self.moving_mean = self.moving_mean.to(X.device)
                self.moving_var = self.moving_var.to(X.device)
            # 保存更新过的moving_mean和moving_var
            Y, self.moving_mean, self.moving_var = batch_norm(
                X, self.gamma, self.beta, self.moving_mean,
                self.moving_var, eps=1e-5, momentum=0.9)
            return Y
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
        nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
        nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
        nn.Linear(84, 10))
    lr, num_epochs, batch_size = 1.0, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())




# ========== 7.6. 残差网络（ResNet）   ========== 
def V76_ResNet() :
    # 残差块
    class Residual(nn.Module):  #@save
        def __init__(self, input_channels, num_channels,
                    use_1x1conv=False, strides=1):
            super().__init__()
            self.conv1 = nn.Conv2d(input_channels, num_channels,
                                kernel_size=3, padding=1, stride=strides)
            self.conv2 = nn.Conv2d(num_channels, num_channels,
                                kernel_size=3, padding=1)
            if use_1x1conv:
                self.conv3 = nn.Conv2d(input_channels, num_channels,
                                    kernel_size=1, stride=strides)
            else:
                self.conv3 = None
            self.bn1 = nn.BatchNorm2d(num_channels)
            self.bn2 = nn.BatchNorm2d(num_channels)

        def forward(self, X):
            Y = F.relu(self.bn1(self.conv1(X)))
            Y = self.bn2(self.conv2(Y))
            if self.conv3:
                X = self.conv3(X)
            Y += X
            return F.relu(Y)


    blk = Residual(3,3)
    X = torch.rand(4, 3, 6, 6)
    Y = blk(X)
    print(Y.shape)
    blk = Residual(3,6, use_1x1conv=True, strides=2)
    print(blk(X).shape)

    # ResNet模型
    def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels,
                                    use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
    # 测试： 打印输入输出的宽高
    X = torch.rand(size=(1, 1, 224, 224))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)

    # 训练
    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())




# ========== 7.7.稠密连接网络（DenseNet）   ==========
def V77_DenseNet() :
    # 定义稠密块
    def conv_block(input_channels, num_channels):
        return nn.Sequential(
            nn.BatchNorm2d(input_channels), nn.ReLU(),
            nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))


# ========== main ==========
if __name__ == '__main__':
    #v71_AlexNet()
    #V72_VGG_11()
    #V73_NiN()
    #V74_GoogLeNet()
    V75_BatchNormal()
    #V76_ResNet()
    #V77_DenseNet()



