import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            out = net(X)
            out = torch.where(out < 0.5, torch.tensor(0), torch.tensor(1))
            metric.add(accuracy(out, y), y.numel())
    return metric[0] / metric[1]


def loss_compute(net, loss, features, labels, batch_size):
    # 把模型输出的值限制在1和inf之间，inf代表无穷大（infinity的缩写）
    data_iter = load_array((features, labels), batch_size)
    rmse = 0.0
    for X, y in data_iter:
        clipped_preds = torch.clamp(net(X), 1, float('inf'))
        rmse_part = loss(clipped_preds, y)
        rmse += rmse_part.item()
    return rmse


def train(net, loss, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_labels), batch_size)
    test_iter = load_array((test_features, test_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
    train_ls.append(evaluate_accuracy(net, train_iter))
    if test_labels is not None:
        test_ls.append(evaluate_accuracy(net, test_iter))
    return train_ls, test_ls


# K折交叉验证
def get_k_fold_data(k, i, X, y):  # 给定k折，给定第几折，返回相应的训练集、测试集
    assert k > 1
    fold_size = X.shape[0] // k  # 每一折的大小为样本数除以k
    X_train, y_train, X_valid, y_valid = None, None, None, None
    for j in range(k):  # 每一折
        # slice:切片函数，(begin,end,step)
        idx = slice(j * fold_size, (j + 1) * fold_size)  # 每一折的切片索引间隔
        X_part, y_part = X[idx, :], y[idx]  # 把每一折对应部分取出来
        if j == i:  # i表示第几折，把它作为验证集
            X_valid, y_valid = X_part, y_part
        elif X_train is None:  # 第一次看到X_train，则把它存起来
            X_train, y_train = X_part, y_part
        else:  # 后面再看到，除了第i外，其余折也作为训练数据集，用torch.cat将原先的合并
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid  # 返回训练集和验证集


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.
    Defined in :numref:`sec_utils`"""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)


# 返回训练和验证误差的平均值
def k_fold(net, loss, k, features, labels, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        X_train, y_train, X_valid, y_valid = get_k_fold_data(k, i, features, labels)  # 把第i折对应分开的数据集、验证集拿出来
        train_ls, valid_ls = train(net, loss, X_train, y_train, X_valid, y_valid, num_epochs, learning_rate,
                                   weight_decay, batch_size)  # 训练集、验证集丢进train函数

        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        # if i == 0:
        #     d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
        #              xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
        #              legend=['train', 'valid'], yscale='log')
        print(f'fold{i + 1},train accuracy {float(train_ls[-1]):f},'
              f'valid accuracy {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k  # 求和做平均


def readData(trainFile, TestFile):
    train_data = pd.read_csv(trainFile)
    test_data = pd.read_csv(TestFile)
    # <class 'pandas.core.frame.DataFrame'>
    # print(test_data.iloc[0:4])
    return train_data, test_data


def correlation(data, save_path):
    corrPearson = data.corr(method="pearson")  # 两种相关系数定义方法
    corrSpearman = data.corr(method="spearman")

    figure = plt.figure(figsize=(30, 25))
    sns.heatmap(corrPearson, annot=True, cmap='RdYlGn', vmin=-1, vmax=+1)
    plt.title("PEARSON")
    plt.xlabel("COLUMNS")
    plt.ylabel("COLUMNS")
    plt.savefig(save_path + "/PEARSON_COR.png")

    figure = plt.figure(figsize=(30, 25))
    sns.heatmap(corrSpearman, annot=True, cmap='RdYlGn', vmin=-1, vmax=+1)
    plt.title("SPEARMAN")
    plt.xlabel("COLUMNS")
    plt.ylabel("COLUMNS")
    plt.savefig(save_path + "/SPEARMAN_COR.png")


def pandas_plot(dataFrameY, dataFrameX, paramList):
    param_dict = {}
    for param in paramList:
        param_dict[param] = dataFrameY[dataFrameX == param].value_counts()
    df = pd.DataFrame(param_dict)
    df.plot(kind='bar', stacked=True)
    plt.show()


def pandas_plot_judgeNull(dataFrameY, dataFrameX):
    param_dict_notnull = dataFrameY[pd.notnull(dataFrameX)].value_counts()
    param_dict_null = dataFrameY[pd.isnull(dataFrameX)].value_counts()
    df = pd.DataFrame({u'notnull': param_dict_notnull, u'null': param_dict_null}).transpose()
    print(df)
    df.plot(kind='bar', stacked=True)
    plt.show()
