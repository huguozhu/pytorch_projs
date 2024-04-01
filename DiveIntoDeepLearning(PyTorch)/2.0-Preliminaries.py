# ========== 预备知识 ==========
import torch
import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l
from torch.distributions import multinomial

# 2.3: 线性代数
def Test_Linear() :
    print("==== 线性代数 ====")
    a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    b = torch.arange(12, dtype=torch.float32).new_ones(3, 4)
    print("a = ", a)
    print("b = ", b)
    c = a + b
    print("c = ",c)


    x = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [8, 7, 6, 5],
        [4, 3, 2, 1]])
    y = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [8, 7, 6, 5],
        [4, 3, 2, 1]])

    print("x * y(0) = ", x*y)
    print("x * y(1) = ", torch.mm(x, y))


    u = torch.tensor([3.0, -4.0])
    print(torch.norm(u))

 
# 2.4: 微积分

# 2.4.1: 计算导数: 计算y=3x**2-4x在x=1处的导数
def f(x):
    return 3*x**2 - 4*x
def numerical_lim(f, x, h):
    return (f(x+h) - f(x)) / h
def Test_Differentiation() :
    print("==== 手写求导数 ====")
    h = 0.1
    for i in range(5):
        print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
        h *= 0.1

# 2.4.2: 计算偏导数

# 2.4.3: 梯度：梯度的本意是一个向量，表示某一函数在该点处的方向导数沿着该方向取得最大值，
#           即函数在该点处沿着该方向（此梯度的方向）变化最快，变化率最大（为该梯度的模）。    
#      计算：函数f()的梯度，即包含n个偏导数的向量

# 2.4.4 链式法则
        
def use_svg_display(): #@save
    backend_inline.set_matplotlib_formats('svg')
def set_figsize(figsize=(3.5, 2.5)): #@save
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize 
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend): #@save
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
        axes.grid()
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,   #@save
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    print("X = ", X)
    print("Y = ", Y)
    if legend is None:
        legend = []
    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()
    
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))
    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
def Test_Show_Differentiation():
    print("==== 用图显示求导 ====")
    x = np.arange(0, 3, 0.1)
    plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
    

# 2.5 自动求导
def Test_Auto_Differentiation() :
    print("==== 用backward()自动求导 ====")
    x = torch.arange(4.0)
    x.requires_grad_(True)
    y = 2*torch.dot(x, x)
    y.backward()
    print(x.grad)
    print(x.grad == 4*x)

    print(" ======= ")
    x.grad.zero_()
    y = x.sum()
    y.backward()
    print(x.grad)
    
    print(" ======= ")
    x.grad.zero_()
    y = x * x * x
    y.sum().backward()
    print(x.grad)

    print("==== 分离计算 ====")
    x.grad.zero_()
    y = x * x
    u = y.detach()
    z = y * x
    z.sum().backward()
    print(x.grad)
    print(x.grad == u)

# 2.6. 概率和统计
# 2.6.1: 采样    
def Test_Sample():
    print("==== 概率 ====")
    count = 100000
    fair_probs = torch.ones([6]) / 6
    p1 = multinomial.Multinomial(count, fair_probs).sample()/count
    print(p1)

    p2 = multinomial.Multinomial(10, fair_probs).sample((5,))
    print(p2)
    cum_counts = p2.cumsum(dim=0)
    print(cum_counts)
    
    print(cum_counts.sum(dim=1, keepdims=True))
    
    estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
    print(estimates)
    
# 3.2: 期望(求平均值)和方差
def Test_Expectation():
    x = 1
    
def Test_Variance():
    x = 1

# Main
#Test_Linear()
#Test_Differentiation()
#Test_Show_Differentiation()
#Test_Auto_Differentiation()
#Test_Sample()    
    

