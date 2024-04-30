# ========== 定义 ==========
# 循环神经网络(RNN)
#   定义：以一类序列（sequence）数据为输入，在序列的演进方向进行递归且  \\
        # 所有节点按链式连接的递归神经网络。
#   优点：可以更好地处理序列信息。
#   方式：通过引入状态变量存储过去信息和当前的输入，从而确定当前的输出。
#   种类：双向循环神经网络(Bi-RNN)、长短期记忆网络(LSTM)、门控循环神经网络(GRU)。

import random
import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
from matplotlib import pyplot as plt
import collections
import re


# ========== 通用函数/类  ========== 
# 8.2.3 建立词表
class Vocab:  #@save
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                            for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# ========== 8.1. 序列模型  ========== 
def V81_Sequences() :
    # 8.1.2 训练
    T = 1000  # 总共产生1000个点
    time = torch.arange(1, T + 1, dtype=torch.float32)
    x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
    plt.plot(time, x)
    plt.title("SinX")
    plt.xlabel("time")
    plt.ylabel("x")
    plt.xlim=[1, 1000]
    #plt.show()    
    
    # 将这个序列转换为模型的特征-标签（feature-label）对
    tau = 4
    features = torch.zeros((T - tau, tau))
    for i in range(tau):
        features[:, i] = x[i: T - tau + i]
    labels = x[tau:].reshape((-1, 1))
    print("features.shape=", features.shape)
    print("labels.shape=", labels.shape)

    batch_size, n_train = 16, 600
    # 只有前n_train个样本用于训练
    train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                                batch_size, is_train=True)

    # 初始化网络权重的函数
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    # 一个简单的多层感知机
    def get_net():
        net = nn.Sequential(nn.Linear(4, 10),
                            nn.ReLU(),
                            nn.Linear(10, 1))
        net.apply(init_weights)
        return net

    # 平方损失。注意：MSELoss计算平方误差时不带系数1/2
    loss = nn.MSELoss(reduction='none')

    def train(net, train_iter, loss, epochs, lr):
        trainer = torch.optim.Adam(net.parameters(), lr)
        for epoch in range(epochs):
            for X, y in train_iter:
                trainer.zero_grad()
                l = loss(net(X), y)
                l.sum().backward()
                trainer.step()
            print(f'epoch {epoch + 1}, '
                f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

    net = get_net()
    train(net, train_iter, loss, 5, 0.01)

    # 8.1.3 预测
    onestep_preds = net(features)
    d2l.plot([time, time[tau:]],
            [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
            'x', legend=['data', '1-step preds'], xlim=[1, 1000],
            figsize=(6, 3))
    
    multistep_preds = torch.zeros(T)
    multistep_preds[: n_train + tau] = x[: n_train + tau]
    for i in range(n_train + tau, T):
        multistep_preds[i] = net(
            multistep_preds[i - tau:i].reshape((1, -1)))

    d2l.plot([time, time[tau:], time[n_train + tau:]],
            [x.detach().numpy(), onestep_preds.detach().numpy(),
            multistep_preds[n_train + tau:].detach().numpy()], 'time',
            'x', legend=['data', '1-step preds', 'multistep preds'],
            xlim=[1, 1000], figsize=(6, 3))
    
    max_steps = 64
    features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
    # 列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
    for i in range(tau):
        features[:, i] = x[i: i + T - tau - max_steps + 1]

    # 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
    for i in range(tau, tau + max_steps):
        features[:, i] = net(features[:, i - tau:i]).reshape(-1)

    steps = (1, 4, 16, 64)
    d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
            [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
            legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
            figsize=(6, 3))




# ========== 8.2 文本预处理：将文本转化为序列数据  ========== 
def V82_Convert_Text_to_Sequence() :
    # 8.2.1 读取数据集
    d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                    '090b5e7e70c295757f55df93cb0a180b9691891a')    
    def read_time_machine():  #@save
        """将时间机器数据集加载到文本行的列表中"""
        with open(d2l.download('time_machine'), 'r') as f:
            lines = f.readlines()
        return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
    lines = read_time_machine()
    print(f'# 文本总行数: {len(lines)}')
    print(lines[0])
    print(lines[10])

    # 8.2.2 词元化（tokenize）：将文本序列转化为词元列表
    def tokenize(lines, token='word'):  #@save
        """将文本行拆分为单词或字符词元"""
        if token == 'word':
            return [line.split() for line in lines]
        elif token == 'char':
            return [list(line) for line in lines]
        else:
            print('错误：未知词元类型：' + token)

    tokens = tokenize(lines)
    for i in range(11):
        print(tokens[i])

    # 打印前几个高频词及其索引
    vocab = Vocab(tokens)
    print(list(vocab.token_to_idx.items())[:10])
    # 将每一条文本行转换成一个数字索引列表
    for i in [0, 10]:
        print('文本:', tokens[i])
        print('索引:', vocab[tokens[i]])
        
    # 8.2.4 整合所有功能
    def load_corpus_time_machine(max_tokens=-1):  #@save
        """返回时光机器数据集的词元索引列表和词表"""
        lines = read_time_machine()
        tokens = tokenize(lines, 'char')
        vocab = Vocab(tokens)
        # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
        # 所以将所有文本行展平到一个列表中
        corpus = [vocab[token] for line in tokens for token in line]
        if max_tokens > 0:
            corpus = corpus[:max_tokens]
        return corpus, vocab

    corpus, vocab = load_corpus_time_machine()
    len(corpus), len(vocab)




# ========== 8.3 语言模型和数据集  ========== 
def V83_Language_Model() :
    # 8.3.3 自然语言统计
    d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                    '090b5e7e70c295757f55df93cb0a180b9691891a')
    def read_time_machine():  #@save
        """将时间机器数据集加载到文本行的列表中"""
        with open(d2l.download('time_machine'), 'r') as f:
            lines = f.readlines()
        return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]    
    tokens = d2l.tokenize(read_time_machine())

    # 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
    # 一元词法
    corpus = [token for line in tokens for token in line]
    vocab = Vocab(corpus)
    # 二元词法
    bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
    bigram_vocab = Vocab(bigram_tokens)
    #print(bigram_vocab.token_freqs[:10])
    # 三元词法
    trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
    trigram_vocab = Vocab(trigram_tokens)
    #print(trigram_vocab.token_freqs[:10])

    freqs = [freq for token, freq in vocab.token_freqs]
    bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
    trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
    plt.plot(freqs)
    plt.plot(bigram_freqs)
    plt.plot(trigram_freqs)
    plt.xlabel('token: x')
    plt.ylabel('frequency: n(x)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(['unigram', 'bigram', 'trigram'])
    plt.show()  

    # 8.3.4 读取长序列数据
    # 8.3.4.1 随机采样:每个样本都是在原始的长序列上任意捕获的子序列
    def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
        """使用随机抽样生成一个小批量子序列"""
        # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
        corpus = corpus[random.randint(0, num_steps - 1):]
        # 减去1，是因为我们需要考虑标签
        num_subseqs = (len(corpus) - 1) // num_steps
        # 长度为num_steps的子序列的起始索引
        initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
        # 在随机抽样的迭代过程中，
        # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
        random.shuffle(initial_indices)

        def data(pos):
            # 返回从pos位置开始的长度为num_steps的序列
            return corpus[pos: pos + num_steps]

        num_batches = num_subseqs // batch_size
        for i in range(0, batch_size * num_batches, batch_size):
            # 在这里，initial_indices包含子序列的随机起始索引
            initial_indices_per_batch = initial_indices[i: i + batch_size]
            X = [data(j) for j in initial_indices_per_batch]
            Y = [data(j + 1) for j in initial_indices_per_batch]
            yield torch.tensor(X), torch.tensor(Y)
    my_seq = list(range(35))
    for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)

    # 8.3.4.2. 顺序分区
    def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
        """使用顺序分区生成一个小批量子序列"""
        # 从随机偏移量开始划分序列
        offset = random.randint(0, num_steps)
        num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
        Xs = torch.tensor(corpus[offset: offset + num_tokens])
        Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
        Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
        num_batches = Xs.shape[1] // num_steps
        for i in range(0, num_steps * num_batches, num_steps):
            X = Xs[:, i: i + num_steps]
            Y = Ys[:, i: i + num_steps]
            yield X, Y
    # 测试顺序分区
    for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)

    # 将随机和顺序分区包装到一个类中
    class SeqDataLoader:  #@save
        """加载序列数据的迭代器"""
        def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
            if use_random_iter:
                self.data_iter_fn = d2l.seq_data_iter_random
            else:
                self.data_iter_fn = d2l.seq_data_iter_sequential
            self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
            self.batch_size, self.num_steps = batch_size, num_steps

        def __iter__(self):
            return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
    def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
        """返回时光机器数据集的迭代器和词表"""
        data_iter = SeqDataLoader(
            batch_size, num_steps, use_random_iter, max_tokens)
        return data_iter, data_iter.vocab

    print("end")




# ========== 8.5 循环神经网络-从零开始  ========== 
def V85_RNN_from_scratch():
    # none





# ========== main ==========
if __name__ == '__main__':
    #V81_Sequences()
    #V82_Convert_Text_to_Sequence()
    #V83_Language_Model()
    V85_RNN_from_scratch()



