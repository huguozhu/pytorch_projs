

import random
import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
from matplotlib import pyplot as plt
import collections
import re

# ========== 9.1 门控循环单元GRU  ========== 
def V91_GRU():
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)








# ========== main ==========
if __name__ == '__main__':
    V91_GRU()



