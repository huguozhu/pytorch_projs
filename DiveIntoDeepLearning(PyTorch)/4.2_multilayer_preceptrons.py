#import torch
#from torch import nn
from d2l import torch as d2l

# ========== 定义 ==========
# 多层感知机 ve 神经网络
#   定义：
#       多层感知机：是一种特殊的神经网络，它由输入层、一个或多个隐藏层和输出层组成
#       神经网络：是一个更广泛的概念，不仅包括多层感知机，还包括其他类型的网络，
#                如卷积神经网络（CNN）、循环神经网络（RNN）等
#   功能：
#       多层感知机：主要用于分类和回归问题，它通过在隐藏层中使用非线性激活函数，能够学习和表示复杂的模式。
#       神经网络：  应用范围更广，可以用于图像识别、自然语言处理、推荐系统等多个领域。

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
