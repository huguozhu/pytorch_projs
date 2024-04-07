import torch
from torch import nn
from torch.nn import functional as F
import numpy

# ========== 5.1 层和块（Layers and Modules） ==========   
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))

# 模拟torch.nn.Sequential    
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)
    def forward(self, X):
        return self.linear(self.net(X))
def v51_test_layers_modules() :
    X = torch.rand(2, 20)
    # 使用torch.nn
    net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    net(X)
    
    # 使用自己的函数实现nn.Sequential
    my_net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    my_net(X)

    # 测试修改前向传播函数
    fixed_hidden_net = FixedHiddenMLP()
    fixed_hidden_net(X)

    # 测试混搭
    chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
    chimera(X)    



# ========== 5.2 参数管理 Parameter Management  ==========   
def v52_Parameter_Management() :
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    X = torch.rand(size=(2, 4))
    net(X)
    # 5.2.1 参数访问
    print(net[2].state_dict())
    # 目标参数
    print(type(net[2].bias))
    print(net[2].bias)
    print(net[2].bias.data)
    print(net[2].weight.grad == None)
    # 一次性访问所有参数
    print(*[(name, param.shape) for name, param in net[0].named_parameters()])
    print(*[(name, param.shape) for name, param in net.named_parameters()])

    # 从嵌套块收集参数
    def block1():
        return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                            nn.Linear(8, 4), nn.ReLU())
    def block2():
        net = nn.Sequential()
        for i in range(4):
            # 在这里嵌套
            net.add_module(f'block {i}', block1())
        return net
    rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
    rgnet(X)
    print(rgnet)


    # 5.2.2参数初始化
    # 内置初始化
    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, mean=0, std=0.01) #将线性
            nn.init.zeros_(m.bias)
    net.apply(init_normal)
    print(net[0].weight.data[0], net[0].bias.data[0])

    # 自定义初始化
    def my_init(m):
        if type(m) == nn.Linear:
            print("Init", *[(name, param.shape)
                            for name, param in m.named_parameters()][0])
            nn.init.uniform_(m.weight, -10, 10)
            m.weight.data *= m.weight.data.abs() >= 5
    net.apply(my_init)
    print(net[0].weight[:2])


    # 5.2.3 参数绑定
    # 我们需要给共享层一个名称，以便可以引用它的参数
    shared = nn.Linear(8, 8)
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                        shared, nn.ReLU(),
                        shared, nn.ReLU(),
                        nn.Linear(8, 1))
    net(X)
    # 检查参数是否相同
    print(net[2].weight.data[0] == net[4].weight.data[0])
    net[2].weight.data[0, 0] = 100
    # 确保它们实际上是同一个对象，而不只是有相同的值
    print(net[2].weight.data[0] == net[4].weight.data[0])



# ========== 5.4 自定义层 Custom Layers  ==========   
def v54_Custom_Layers() :
    # 5.4.1 不带参数的层
    class CenteredLayer(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, X):
            return X - X.mean()
        
    layer = CenteredLayer()
    print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

    net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
    Y = net(torch.rand(4, 8))
    print(Y.mean())


    # 5.4.2 带参数的层
    class MyLinear(nn.Module):
        def __init__(self, in_units, units):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(in_units, units))
            self.bias = nn.Parameter(torch.randn(units,))
        def forward(self, X):
            linear = torch.matmul(X, self.weight.data) + self.bias.data
            return F.relu(linear)
    linear = MyLinear(5, 3)
    print(linear.weight)



# ========== 5.5 读写文件 File IO  ==========   
def v55_FileIO() :
    # 5.5.1 加载和保存张量
    x = torch.arange(4)
    print("x=", x)
    torch.save(x, 'x-file')
    x2 = torch.load('x-file')
    print("x2=", x2)

    y = torch.zeros(4)
    torch.save([x, y],'x-files')
    x2, y2 = torch.load('x-files')
    print(x2, y2)

    mydict = {'x': x, 'y': y}
    torch.save(mydict, 'mydict')
    mydict2 = torch.load('mydict')
    print(mydict2)

    # 5.5.2. 加载和保存模型参数
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(20, 256)
            self.output = nn.Linear(256, 10)
        def forward(self, x):
            return self.output(F.relu(self.hidden(x)))

    net = MLP()
    X = torch.randn(size=(2, 20))
    Y = net(X)
    torch.save(net.state_dict(), 'mlp.params')

    clone = MLP()
    clone.load_state_dict(torch.load('mlp.params'))
    clone.eval()

    #由于两个实例具有相同的模型参数，在输入相同的X时， 两个实例的计算结果应该相同。
    Y_clone = clone(X)
    print(Y_clone == Y)



# ========== 5.6 多GPU  ==========   
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
def v56_GPUs() :
    # 5.6.1 计算设备
    print(torch.device('cpu'))
    print(torch.device('cuda')) 
    print(torch.device('cuda:1'))
    print(torch.cuda.device_count())

    # 5.6.2 张量和GPU
    x = torch.tensor([1, 2, 3])
    print(x.device)
    print(x)
    y = torch.ones(2, 3, device=torch.device(try_gpu()))
    print(y.device)
    # GPU间数据复制
    X = torch.ones(2, 3, device=try_gpu(1))
    Y = torch.rand(2, 3, device=try_gpu(0))
    Z = X.cuda(0)   # 将GPU1中X的数据复制给到GPU0中的Z
    
    # 5.6.3. 神经网络与GPU
    net = nn.Sequential(nn.Linear(3, 1))
    net = net.to(device=try_gpu())      #将CPU中创建的神经网络放到GPU中
    print(net[0].weight.data.device)




# ========== main ==========
if __name__ == '__main__':    
    #v51_test_layers_modules()
    #v52_Parameter_Management()
    #v54_Custom_Layers()
    #v55_FileIO()
    v56_GPUs()



