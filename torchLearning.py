# 随机模块
import random

# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# numpy
import numpy as np

# pytorch
import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.utils.data import Dataset,TensorDataset,DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter


# 回归类数据集创建函数
def tensorGenReg(num_examples = 1000, w = [2, -1, 1], bias = True, delta = 0.01, deg = 1):
    """回归类数据集创建函数。

    :param num_examples: 创建数据集的数据量
    :param w: 包括截距的（如果存在）特征系数向量
    :param bias：是否需要截距
    :param delta：扰动项取值
    :param deg：方程次数
    :return: 生成的特征张和标签张量
    """
    
    if bias == True:
        num_inputs = len(w)-1                                                        # 特征张量
        features_true = torch.randn(num_examples, num_inputs)                        # 不包含全是1的列的特征张量
        w_true = torch.tensor(w[:-1]).reshape(-1, 1).float()                         # 自变量系数
        b_true = torch.tensor(w[-1]).float()                                         # 截距
        if num_inputs == 1:                                                          # 若输入特征只有1个，则不能使用矩阵乘法
            labels_true = torch.pow(features_true, deg) * w_true + b_true
        else:
            labels_true = torch.mm(torch.pow(features_true, deg), w_true) + b_true
        features = torch.cat((features_true, torch.ones(len(features_true), 1)), 1)  # 在特征张量的最后添加一列全是1的列
        labels = labels_true + torch.randn(size = labels_true.shape) * delta         
                
    else: 
        num_inputs = len(w)
        features = torch.randn(num_examples, num_inputs)
        w_true = torch.tensor(w).reshape(-1, 1).float()
        if num_inputs == 1:
            labels_true = torch.pow(features, deg) * w_true
        else:
            labels_true = torch.mm(torch.pow(features, deg), w_true)
        labels = labels_true + torch.randn(size = labels_true.shape) * delta
    return features, labels


# 分类数据集的创建函数
def tensorGenCla(num_examples = 500, num_inputs = 2, num_class = 3, deg_dispersion = [4, 2], bias = False):
    """分类数据集创建函数。
    
    :param num_examples: 每个类别的数据数量
    :param num_inputs: 数据集特征数量
    :param num_class：数据集标签类别总数
    :param deg_dispersion：数据分布离散程度参数，需要输入一个列表，其中第一个参数表示每个类别数组均值的参考、第二个参数表示随机数组标准差。
    :param bias：建立模型逻辑回归模型时是否带入截距
    :return: 生成的特征张量和标签张量，其中特征张量是浮点型二维数组，标签张量是长正型二维数组。
    """
    
    cluster_l = torch.empty(num_examples, 1)                         # 每一类标签张量的形状
    mean_ = deg_dispersion[0]                                        # 每一类特征张量的均值的参考值
    std_ = deg_dispersion[1]                                         # 每一类特征张量的方差
    lf = []                                                          # 用于存储每一类特征张量的列表容器
    ll = []                                                          # 用于存储每一类标签张量的列表容器
    k = mean_ * (num_class-1) / 2                                    # 每一类特征张量均值的惩罚因子（视频中部分是+1，实际应该是-1）
    
    for i in range(num_class):
        data_temp = torch.normal(i*mean_-k, std_, size=(num_examples, num_inputs))     # 生成每一类张量
        lf.append(data_temp)                                                           # 将每一类张量添加到lf中
        labels_temp = torch.full_like(cluster_l, i)                                    # 生成类一类的标签
        ll.append(labels_temp)                                                         # 将每一类标签添加到ll中
        
    features = torch.cat(lf).float()
    labels = torch.cat(ll).long()
    
    if bias == True:
        features = torch.cat((features, torch.ones(len(features), 1)), 1)              # 在特征张量中添加一列全是1的列
    return features, labels


# 小批量切分函数
def data_iter(batch_size, features, labels):
    """
    数据切分函数
    
    :param batch_size: 每个子数据集包含多少数据
    :param featurs: 输入的特征张量
    :param labels：输入的标签张量
    :return l：包含batch_size个列表，每个列表切分后的特征和标签所组成 
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    l = []
    for i in range(0, num_examples, batch_size):
        j = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        l.append([torch.index_select(features, 0, j), torch.index_select(labels, 0, j)])
    return l


# 简单线性回归向前传播函数
def linreg(X, w):
    """
    线性网络正向传播
    """
    return torch.mm(X, w)


# MSE计算函数
def MSE_loss(y_hat, y):
    """
    MSE计算公式
    """
    num_ = y.numel()
    sse = torch.sum((y_hat.reshape(-1, 1) - y.reshape(-1, 1)) ** 2)
    return sse / num_


# 系数迭代函数
def sgd(params, lr):
    """
    系数迭代函数
    """
    params.data -= lr * params.grad 
    params.grad.zero_()


    
    

def data_split(features, labels, rate=0.7):
    """
    训练集和测试集切分函数
    
    :param features: 输入的特征张量
    :param labels：输入的标签张量
    :param rate：训练集占所有数据的比例
    :return Xtrain, Xtest, ytrain, ytest：返回特征张量的训练集、测试集，以及标签张量的训练集、测试集 
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    l = []
    num_train = int(num_examples * rate)
    indices_train = torch.tensor(indices[: num_train])
    indices_test = torch.tensor(indices[num_train: ])
    Xtrain = features[indices_train]
    ytrain = labels[indices_train]
    Xtest = features[indices_test]
    ytest = labels[indices_test]
    return Xtrain, Xtest, ytrain, ytest    


def sigmoid(z):
    """
    sigmoid函数
    """
    return 1/(1+torch.exp(-z))


def logistic(X, w):
    return sigmoid(torch.mm(X, w))

def cal(sigma, p=0.5):
    return((sigma >= p).float())

def accuracy(sigma, y):
    acc_bool = cal(y_hat).flatten() == y.flatten()
    acc = torch.mean(acc_bool.float())
    return(acc)

def cross_entropy(sigma, y):
    return(-(1/y.numel())*torch.sum((1-y)*torch.log(1-sigma)+y*torch.log(sigma)))

def sgd(params, lr):
        params.data -= lr * params.grad 
        params.grad.zero_()

def acc_zhat(zhat, y):
    """输入为线性方程计算结果，输出为逻辑回归准确率的函数

    :param zhat：线性方程输出结果 
    :param y: 数据集标签张量
    :return：准确率 
    """
    sigma = sigmoid(zhat)
    return accuracy(sigma, y)

def softmax(X, w):
    m = torch.exp(torch.mm(X, w))
    sp = torch.sum(m, 1).reshape(-1, 1)
    return m / sp

def m_cross_entropy(soft_z, y):
    y = y.long()
    prob_real = torch.gather(soft_z, 1, y)
    return (-(1/y.numel()) * torch.log(torch.prod(prob_real)))

def m_accuracy(soft_z, y):
    acc_bool = torch.argmax(soft_z, 1).flatten() == y.flatten()
    acc = torch.mean(acc_bool.float())
    return(acc)



# 常用类

# 常用数据处理类
# 适用于封装自定义数据集的类
class GenData(Dataset):
    def __init__(self, features, labels):           
        self.features = features                    
        self.labels = labels                       
        self.lens = len(features)                  

    def __getitem__(self, index):
        return self.features[index,:],self.labels[index]    

    def __len__(self):
        return self.lens


# 常用模型类
# 单层神经网络
class LR_class(nn.Module):                                         # 没有激活函数
    def __init__(self, in_features=2, out_features=1):       # 定义模型的点线结构
        super(LR_class, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
    def forward(self, x):                                    # 定义模型的正向传播规则
        out = self.linear(x)             
        return out
    

    
# Sigmoid激活函数
class Sigmoid_class1(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden=4, out_features=1, BN_model=None):       
        super(Sigmoid_class1, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden)
        self.normalize1 = nn.BatchNorm1d(n_hidden)
        self.linear2 = nn.Linear(n_hidden, out_features)
        self.BN_model = BN_model
        
    def forward(self, x):
        if self.BN_model == None:
            z1 = self.linear1(x)
            p1 = torch.sigmoid(z1)                   
            out = self.linear2(p1)
        elif self.BN_model == 'pre':
            z1 = self.normalize1(self.linear1(x))
            p1 = torch.sigmoid(z1)                   
            out = self.linear2(p1)
        elif self.BN_model == 'post':
            z1 = self.linear1(x)
            p1 = torch.sigmoid(z1)                   
            out = self.linear2(self.normalize1(p1))
        return out

    

class Sigmoid_class2(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden1=4, n_hidden2=4, out_features=1, BN_model=None):       
        super(Sigmoid_class2, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1)
        self.normalize1 = nn.BatchNorm1d(n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.normalize2 = nn.BatchNorm1d(n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, out_features) 
        self.BN_model = BN_model
        
    def forward(self, x):
        if self.BN_model == None:
            z1 = self.linear1(x)
            p1 = torch.sigmoid(z1)
            z2 = self.linear2(p1)
            p2 = torch.sigmoid(z2)
            out = self.linear3(p2)
        elif self.BN_model == 'pre':
            z1 = self.normalize1(self.linear1(x))
            p1 = torch.sigmoid(z1)
            z2 = self.normalize2(self.linear2(p1))
            p2 = torch.sigmoid(z2)
            out = self.linear3(p2)
        elif self.BN_model == 'post':
            z1 = self.linear1(x)
            p1 = torch.sigmoid(z1)
            z2 = self.linear2(self.normalize1(p1))
            p2 = torch.sigmoid(z2)
            out = self.linear3(self.normalize2(p2))
        return out


    
class Sigmoid_class3(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden1=4, n_hidden2=4, n_hidden3=4, out_features=1, BN_model=None):       
        super(Sigmoid_class3, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1)
        self.normalize1 = nn.BatchNorm1d(n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.normalize2 = nn.BatchNorm1d(n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, n_hidden3)
        self.normalize3 = nn.BatchNorm1d(n_hidden3)
        self.linear4 = nn.Linear(n_hidden3, out_features) 
        self.BN_model = BN_model
        
    def forward(self, x): 
        if self.BN_model == None:
            z1 = self.linear1(x)
            p1 = torch.sigmoid(z1)
            z2 = self.linear2(p1)
            p2 = torch.sigmoid(z2)
            z3 = self.linear3(p2)
            p3 = torch.sigmoid(z3)
            out = self.linear4(p3)
        elif self.BN_model == 'pre':
            z1 = self.normalize1(self.linear1(x))
            p1 = torch.sigmoid(z1)
            z2 = self.normalize2(self.linear2(p1))
            p2 = torch.sigmoid(z2)
            z3 = self.normalize3(self.linear3(p2))
            p3 = torch.sigmoid(z3)
            out = self.linear4(p3)
        elif self.BN_model == 'post':
            z1 = self.linear1(x)
            p1 = torch.sigmoid(z1)
            z2 = self.linear2(self.normalize1(p1))
            p2 = torch.sigmoid(z2)
            z3 = self.linear3(self.normalize2(p2))
            p3 = torch.sigmoid(z3)
            out = self.linear4(self.normalize3(p3))
        return out


    
class Sigmoid_class4(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden1=4, n_hidden2=4, n_hidden3=4, n_hidden4=4, out_features=1, BN_model=None):       
        super(Sigmoid_class4, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1)
        self.normalize1 = nn.BatchNorm1d(n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.normalize2 = nn.BatchNorm1d(n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, n_hidden3)
        self.normalize3 = nn.BatchNorm1d(n_hidden3)
        self.linear4 = nn.Linear(n_hidden3, n_hidden4)
        self.normalize4 = nn.BatchNorm1d(n_hidden4)
        self.linear5 = nn.Linear(n_hidden4, out_features) 
        self.BN_model = BN_model
        
    def forward(self, x): 
        if self.BN_model == None:
            z1 = self.linear1(x)
            p1 = torch.sigmoid(z1)
            z2 = self.linear2(p1)
            p2 = torch.sigmoid(z2)
            z3 = self.linear3(p2)
            p3 = torch.sigmoid(z3)
            z4 = self.linear4(p3)
            p4 = torch.sigmoid(z4)
            out = self.linear5(p4)
        elif self.BN_model == 'pre':
            z1 = self.normalize1(self.linear1(x))
            p1 = torch.sigmoid(z1)
            z2 = self.normalize2(self.linear2(p1))
            p2 = torch.sigmoid(z2)
            z3 = self.normalize3(self.linear3(p2))
            p3 = torch.sigmoid(z3)
            z4 = self.normalize4(self.linear4(p3))
            p4 = torch.sigmoid(z4)
            out = self.linear5(p4)
        elif self.BN_model == 'post':
            z1 = self.linear1(x)
            p1 = torch.sigmoid(z1)
            z2 = self.linear2(self.normalize1(p1))
            p2 = torch.sigmoid(z2)
            z3 = self.linear3(self.normalize1(p2))
            p3 = torch.sigmoid(z3)
            z4 = self.linear4(self.normalize1(p3))
            p4 = torch.sigmoid(z4)
            out = self.linear5(self.normalize1(p4))
        return out

    
# tanh激活函数
class tanh_class1(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden=4, out_features=1, BN_model=None):       
        super(tanh_class1, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden)
        self.normalize1 = nn.BatchNorm1d(n_hidden)
        self.linear2 = nn.Linear(n_hidden, out_features)
        self.BN_model = BN_model
        
    def forward(self, x):
        if self.BN_model == None:
            z1 = self.linear1(x)
            p1 = torch.tanh(z1)                   
            out = self.linear2(p1)
        elif self.BN_model == 'pre':
            z1 = self.normalize1(self.linear1(x))
            p1 = torch.tanh(z1)                   
            out = self.linear2(p1)
        elif self.BN_model == 'post':
            z1 = self.linear1(x)
            p1 = torch.tanh(z1)                   
            out = self.linear2(self.normalize1(p1))
        return out

    

class tanh_class2(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden1=4, n_hidden2=4, out_features=1, BN_model=None):       
        super(tanh_class2, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1)
        self.normalize1 = nn.BatchNorm1d(n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.normalize2 = nn.BatchNorm1d(n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, out_features) 
        self.BN_model = BN_model
        
    def forward(self, x):
        if self.BN_model == None:
            z1 = self.linear1(x)
            p1 = torch.tanh(z1)
            z2 = self.linear2(p1)
            p2 = torch.tanh(z2)
            out = self.linear3(p2)
        elif self.BN_model == 'pre':
            z1 = self.normalize1(self.linear1(x))
            p1 = torch.tanh(z1)
            z2 = self.normalize2(self.linear2(p1))
            p2 = torch.tanh(z2)
            out = self.linear3(p2)
        elif self.BN_model == 'post':
            z1 = self.linear1(x)
            p1 = torch.tanh(z1)
            z2 = self.linear2(self.normalize1(p1))
            p2 = torch.tanh(z2)
            out = self.linear3(self.normalize2(p2))
        return out

class tanh_class3(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden1=4, n_hidden2=4, n_hidden3=4, out_features=1, BN_model=None):       
        super(tanh_class3, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1)
        self.normalize1 = nn.BatchNorm1d(n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.normalize2 = nn.BatchNorm1d(n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, n_hidden3)
        self.normalize3 = nn.BatchNorm1d(n_hidden3)
        self.linear4 = nn.Linear(n_hidden3, out_features) 
        self.BN_model = BN_model
        
    def forward(self, x):
        if self.BN_model == None:
            z1 = self.linear1(x)
            p1 = torch.tanh(z1)
            z2 = self.linear2(p1)
            p2 = torch.tanh(z2)
            z3 = self.linear3(p2)
            p3 = torch.tanh(z3)
            out = self.linear4(p3)
        elif self.BN_model == 'pre':
            z1 = self.normalize1(self.linear1(x))
            p1 = torch.tanh(z1)
            z2 = self.normalize2(self.linear2(p1))
            p2 = torch.tanh(z2)
            z3 = self.normalize3(self.linear3(p2))
            p3 = torch.tanh(z3)
            out = self.linear4(p3)
        elif self.BN_model == 'post':
            z1 = self.linear1(x)
            p1 = torch.tanh(z1)
            z2 = self.linear2(self.normalize1(p1))
            p2 = torch.tanh(z2)
            z3 = self.linear3(self.normalize2(p2))
            p3 = torch.tanh(z3)
            out = self.linear4(self.normalize3(p3))
        return out


    
class tanh_class4(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden1=4, n_hidden2=4, n_hidden3=4, n_hidden4=4, out_features=1, BN_model=None):       
        super(tanh_class4, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1)
        self.normalize1 = nn.BatchNorm1d(n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.normalize2 = nn.BatchNorm1d(n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, n_hidden3)
        self.normalize3 = nn.BatchNorm1d(n_hidden3)
        self.linear4 = nn.Linear(n_hidden3, n_hidden4)
        self.normalize4 = nn.BatchNorm1d(n_hidden4)
        self.linear5 = nn.Linear(n_hidden4, out_features) 
        self.BN_model = BN_model
        
    def forward(self, x): 
        if self.BN_model == None:
            z1 = self.linear1(x)
            p1 = torch.tanh(z1)
            z2 = self.linear2(p1)
            p2 = torch.tanh(z2)
            z3 = self.linear3(p2)
            p3 = torch.tanh(z3)
            z4 = self.linear4(p3)
            p4 = torch.tanh(z4)
            out = self.linear5(p4)
        elif self.BN_model == 'pre':
            z1 = self.normalize1(self.linear1(x))
            p1 = torch.tanh(z1)
            z2 = self.normalize2(self.linear2(p1))
            p2 = torch.tanh(z2)
            z3 = self.normalize3(self.linear3(p2))
            p3 = torch.tanh(z3)
            z4 = self.normalize4(self.linear4(p3))
            p4 = torch.tanh(z4)
            out = self.linear5(p4)
        elif self.BN_model == 'post':
            z1 = self.linear1(x)
            p1 = torch.tanh(z1)
            z2 = self.linear2(self.normalize1(p1))
            p2 = torch.tanh(z2)
            z3 = self.linear3(self.normalize2(p2))
            p3 = torch.tanh(z3)
            z4 = self.linear4(self.normalize3(p3))
            p4 = torch.tanh(z4)
            out = self.linear5(self.normalize4(p4))
        return out
    
    
    
# ReLU激活函数
class ReLU_class1(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden=4, out_features=1, bias = True, BN_model=None):       
        super(ReLU_class1, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden, bias=bias)
        self.normalize1 = nn.BatchNorm1d(n_hidden)
        self.linear2 = nn.Linear(n_hidden, out_features, bias=bias)
        self.BN_model = BN_model
        
    def forward(self, x):
        if self.BN_model == None:
            z1 = self.linear1(x)
            p1 = torch.relu(z1)                  
            out = self.linear2(p1)
        elif self.BN_model == 'pre':
            z1 = self.linear1(self.normalize1(x))
            p1 = torch.relu(z1)                  
            out = self.linear2(p1)
        elif self.BN_model == 'post':
            z1 = self.linear1(x)
            p1 = torch.relu(z1)                  
            out = self.linear2(self.normalize1(p1))
        return out

    
class ReLU_class2(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden1=4, n_hidden2=4, out_features=1, bias=True, BN_model=None):       
        super(ReLU_class2, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1, bias=bias)
        self.normalize1 = nn.BatchNorm1d(n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2, bias=bias)
        self.normalize2 = nn.BatchNorm1d(n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, out_features, bias=bias)
        self.BN_model = BN_model
        
    def forward(self, x): 
        if self.BN_model == None:
            z1 = self.linear1(x)
            p1 = torch.relu(z1)
            z2 = self.linear2(p1)
            p2 = torch.relu(z2)
            out = self.linear3(p2)
        elif self.BN_model == 'pre':
            z1 = self.normalize1(self.linear1(x))
            p1 = torch.relu(z1)
            z2 = self.normalize2(self.linear2(p1))
            p2 = torch.relu(z2)
            out = self.linear3(p2)
        elif self.BN_model == 'post':
            z1 = self.linear1(x)
            p1 = torch.relu(z1)
            z2 = self.linear2(self.normalize1(p1))
            p2 = torch.relu(z2)
            out = self.linear3(self.normalize2(p2))
        return out

    
class ReLU_class3(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden1=4, n_hidden2=4, n_hidden3=4, out_features=1, bias=True, BN_model=None):       
        super(ReLU_class3, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1, bias=bias)
        self.normalize1 = nn.BatchNorm1d(n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2, bias=bias)
        self.normalize2 = nn.BatchNorm1d(n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, n_hidden3, bias=bias)
        self.normalize3 = nn.BatchNorm1d(n_hidden3)
        self.linear4 = nn.Linear(n_hidden3, out_features, bias=bias)
        self.BN_model = BN_model
        
    def forward(self, x):  
        if self.BN_model == None:
            z1 = self.linear1(x)
            p1 = torch.relu(z1)
            z2 = self.linear2(p1)
            p2 = torch.relu(z2)
            z3 = self.linear3(p2)
            p3 = torch.relu(z3)
            out = self.linear4(p3)
        elif self.BN_model == 'pre':
            z1 = self.normalize1(self.linear1(x))
            p1 = torch.relu(z1)
            z2 = self.normalize2(self.linear2(p1))
            p2 = torch.relu(z2)
            z3 = self.normalize3(self.linear3(p2))
            p3 = torch.relu(z3)
            out = self.linear4(p3)
        elif self.BN_model == 'post':
            z1 = self.linear1(x)
            p1 = torch.relu(z1)
            z2 = self.linear2(self.normalize1(p1))
            p2 = torch.relu(z2)
            z3 = self.linear3(self.normalize2(p2))
            p3 = torch.relu(z3)
            out = self.linear4(self.normalize3(p3))
        return out

    
class ReLU_class4(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden1=4, n_hidden2=4, n_hidden3=4, n_hidden4=4, out_features=1, bias=True, BN_model=None):       
        super(ReLU_class4, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1, bias=bias)
        self.normalize1 = nn.BatchNorm1d(n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2, bias=bias)
        self.normalize2 = nn.BatchNorm1d(n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, n_hidden3, bias=bias)
        self.normalize3 = nn.BatchNorm1d(n_hidden3)
        self.linear4 = nn.Linear(n_hidden3, n_hidden4, bias=bias)
        self.normalize4 = nn.BatchNorm1d(n_hidden4)
        self.linear5 = nn.Linear(n_hidden4, out_features, bias=bias) 
        self.BN_model = BN_model
        
    def forward(self, x):
        if self.BN_model == None:
            z1 = self.linear1(x)
            p1 = torch.relu(z1)
            z2 = self.linear2(p1)
            p2 = torch.relu(z2)
            z3 = self.linear3(p2)
            p3 = torch.relu(z3)
            z4 = self.linear4(p3)
            p4 = torch.relu(z4)
            out = self.linear5(p4)
        elif self.BN_model == 'pre':
            z1 = self.normalize1(self.linear1(x))
            p1 = torch.relu(z1)
            z2 = self.normalize2(self.linear2(p1))
            p2 = torch.relu(z2)
            z3 = self.normalize3(self.linear3(p2))
            p3 = torch.relu(z3)
            z4 = self.normalize4(self.linear4(p3))
            p4 = torch.relu(z4)
            out = self.linear5(p4)
        elif self.BN_model == 'post':
            z1 = self.linear1(x)
            p1 = torch.relu(z1)
            z2 = self.linear2(self.normalize1(p1))
            p2 = torch.relu(z2)
            z3 = self.linear3(self.normalize2(p2))
            p3 = torch.relu(z3)
            z4 = self.linear4(self.normalize3(p3))
            p4 = torch.relu(z4)
            out = self.linear5(self.normalize4(p4))
        return out  
    
    
    
    
def split_loader(features, labels, batch_size=10, rate=0.7):
    """数据封装、切分和加载函数：
    
    :param features：输入的特征 
    :param labels: 数据集标签张量
    :param batch_size：数据加载时的每一个小批数据量 
    :param rate: 训练集数据占比
    :return：加载好的训练集和测试集
    """    
    data = GenData(features, labels) 
    num_train = int(data.lens * 0.7)
    num_test = data.lens - num_train
    data_train, data_test = random_split(data, [num_train, num_test])
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False)
    return(train_loader, test_loader)



def fit(net, criterion, optimizer, batchdata, epochs=3, cla=False):
    """模型训练函数
    
    :param net：待训练的模型 
    :param criterion: 损失函数
    :param optimizer：优化算法
    :param batchdata: 训练数据集
    :param cla: 是否是分类问题
    :param epochs: 遍历数据次数
    """
    for epoch  in range(epochs):
        for X, y in batchdata:
            if cla == True:
                y = y.flatten().long()          # 如果是分类问题，需要对y进行整数转化
            yhat = net.forward(X)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            
def mse_cal(data_loader, net):
    """mse计算函数
    
    :param data_loader：加载好的数据
    :param net: 模型
    :return：根据输入的数据，输出其MSE计算结果
    """
    data = data_loader.dataset                # 还原Dataset类
    X = data[:][0]                            # 还原数据的特征
    y = data[:][1]                            # 还原数据的标签
    yhat = net(X)
    return F.mse_loss(yhat, y)


def accuracy_cal(data_loader, net):
    """准确率
    
    :param data_loader：加载好的数据
    :param net: 模型
    :return：根据输入的数据，输出其准确率计算结果
    """
    data = data_loader.dataset                # 还原Dataset类
    X = data[:][0]                            # 还原数据的特征
    y = data[:][1]                            # 还原数据的标签
    zhat = net(X)                             # 默认是分类问题，并且输出结果是未经softmax转化的结果
    soft_z = F.softmax(zhat, 1)                  # 进行softmax转化
    acc_bool = torch.argmax(soft_z, 1).flatten() == y.flatten()
    acc = torch.mean(acc_bool.float())
    return acc 



def model_train_test(model, 
                     train_data,
                     test_data,
                     num_epochs = 20, 
                     criterion = nn.MSELoss(), 
                     optimizer = optim.SGD, 
                     lr = 0.03, 
                     cla = False, 
                     eva = mse_cal):
    """模型误差测试函数：
    
    :param model_l：模型
    :param train_data：训练数据
    :param test_data: 测试数据   
    :param num_epochs：迭代轮数
    :param criterion: 损失函数
    :param lr: 学习率
    :param cla: 是否是分类模型
    :return：MSE列表
    """  
    # 模型评估指标矩阵
    train_l = []
    test_l = []
    # 模型训练过程
    for epochs in range(num_epochs):
        model.train()
        fit(net = model, 
            criterion = criterion, 
            optimizer = optimizer(model.parameters(), lr = lr), 
            batchdata = train_data, 
            epochs = epochs, 
            cla = cla)
        model.eval()
        train_l.append(eva(train_data, model).detach())
        test_l.append(eva(test_data, model).detach())
    return train_l, test_l


def model_comparison(model_l, 
                     name_l, 
                     train_data,
                     test_data,
                     num_epochs = 20, 
                     criterion = nn.MSELoss(), 
                     optimizer = optim.SGD, 
                     lr = 0.03, 
                     cla = False,
                     eva = mse_cal):
    """模型对比函数：
    
    :param model_l：模型序列
    :param name_l：模型名称序列
    :param train_data：训练数据
    :param test_data：测试数据    
    :param num_epochs：迭代轮数
    :param criterion: 损失函数
    :param lr: 学习率
    :param cla: 是否是分类模型
    :param eva: 模型评估指标
    :return：评估指标张量矩阵 
    """
    # 模型评估指标矩阵
    train_l = torch.zeros(len(model_l), num_epochs)
    test_l = torch.zeros(len(model_l), num_epochs)
    # 模型训练过程
    for epochs in range(num_epochs):
        for i, model in enumerate(model_l):
            model.train()
            fit(net = model, 
                criterion = criterion, 
                optimizer = optimizer(model.parameters(), lr = lr), 
                batchdata = train_data, 
                epochs = epochs, 
                cla = cla)
            model.eval()
            train_l[i][epochs] = eva(train_data, model).detach()
            test_l[i][epochs] = eva(test_data, model).detach()
    return train_l, test_l





def weights_vp(model, att="grad"):
    """观察各层参数取值和梯度的小提琴图绘图函数。
    
    :param model：观察对象（模型）
    :param att：选择参数梯度（grad）还是参数取值（weights）进行观察
    :return: 对应att的小提琴图    
    """
    vp = []
    for i, m in enumerate(model.modules()):
        if isinstance(m, nn.Linear):
            if att == "grad":
                vp_x = m.weight.grad.detach().reshape(-1, 1).numpy()
            else:
                vp_x = m.weight.detach().reshape(-1, 1).numpy()
            vp_y = np.full_like(vp_x, i)
            vp_a = np.concatenate((vp_x, vp_y), 1)
            vp.append(vp_a)
    vp_r = np.concatenate((vp), 0)
    ax = sns.violinplot(y = vp_r[:, 0], x = vp_r[:, 1])
    ax.set(xlabel='num_hidden', title=att)
    
    
    
def Z_ScoreNormalization(data):
    """Z-Score标准化函数。
    
    :param data：需要标准化的数据，往往是训练集特征
    :return: Z-Score标准化后的数据    
    """
    stdDf = data.std(0)
    meanDf = data.mean(0)
    normSet = (data - meanDf) / stdDf
    return normSet 



# 带BN层的神经网络
# Sigmoid1前置BN层
class Sigmoid_class1_norm1(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden=4, out_features=1):       
        super(Sigmoid_class1_norm1, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden)
        self.normalize1 = nn.BatchNorm1d(n_hidden)
        self.linear2 = nn.Linear(n_hidden, out_features)
        
    def forward(self, x):                                   
        z1 = self.normalize1(self.linear1(x))
        p1 = torch.sigmoid(z1)                   
        out = self.linear2(p1)
        return out
    
class Sigmoid_class1_norm2(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden=4, out_features=1):       
        super(Sigmoid_class1_norm2, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden)
        self.normalize1 = nn.BatchNorm1d(n_hidden)
        self.linear2 = nn.Linear(n_hidden, out_features)
        
    def forward(self, x):                                   
        z1 = self.linear1(x)
        p1 = torch.sigmoid(z1)                   
        out = self.linear2(self.normalize1(p1))
        return out



# Sigmoid2前置BN层
class Sigmoid_class2_norm1(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden1=4, n_hidden2=4, out_features=1):       
        super(Sigmoid_class2_norm1, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1)
        self.normalize1 = nn.BatchNorm1d(n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.normalize2 = nn.BatchNorm1d(n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, out_features) 
        
    def forward(self, x):                                    
        z1 = self.normalize1(self.linear1(x))
        p1 = torch.sigmoid(z1)
        z2 = self.normalize2(self.linear2(p1))
        p2 = torch.sigmoid(z2)
        out = self.linear3(p2)
        return out

# Sigmoid2后置BN层    
class Sigmoid_class2_norm2(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden1=4, n_hidden2=4, out_features=1):       
        super(Sigmoid_class2_norm2, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1)
        self.normalize1 = nn.BatchNorm1d(n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.normalize2 = nn.BatchNorm1d(n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, out_features) 
        
    def forward(self, x):                                    
        z1 = self.linear1(x)
        p1 = torch.sigmoid(z1)
        z2 = self.linear2(self.normalize1(p1))
        p2 = torch.sigmoid(z2)
        out = self.linear3(self.normalize2(p2))
        return out

# Sigmoid3前置BN层
class Sigmoid_class3_norm1(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden1=4, n_hidden2=4, n_hidden3=4, out_features=1):       
        super(Sigmoid_class3_norm1, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1)
        self.normalize1 = nn.BatchNorm1d(n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.normalize2 = nn.BatchNorm1d(n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, n_hidden3)
        self.normalize3 = nn.BatchNorm1d(n_hidden3)
        self.linear4 = nn.Linear(n_hidden3, out_features) 
        
    def forward(self, x):                                    
        z1 = self.normalize1(self.linear1(x))
        p1 = torch.sigmoid(z1)
        z2 = self.normalize2(self.linear2(p1))
        p2 = torch.sigmoid(z2)
        z3 = self.normalize3(self.linear3(p2))
        p3 = torch.sigmoid(z3)
        out = self.linear4(p3)
        return out

# Sigmoid3后置BN层
class Sigmoid_class3_norm2(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden1=4, n_hidden2=4, n_hidden3=4, out_features=1):       
        super(Sigmoid_class3_norm2, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1)
        self.normalize1 = nn.BatchNorm1d(n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.normalize2 = nn.BatchNorm1d(n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, n_hidden3)
        self.normalize3 = nn.BatchNorm1d(n_hidden3)
        self.linear4 = nn.Linear(n_hidden3, out_features) 
        
    def forward(self, x):                                    
        z1 = self.linear1(x)
        p1 = torch.sigmoid(z1)
        z2 = self.linear2(self.normalize1(p1))
        p2 = torch.sigmoid(z2)
        z3 = self.linear3(self.normalize2(p2))
        p3 = torch.sigmoid(z3)
        out = self.linear4(self.normalize3(p3))
        return out

# tanh1前置BN层
class tanh_class1_norm1(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden=4, out_features=1):       
        super(tanh_class1_norm1, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden)
        self.normalize1 = nn.BatchNorm1d(n_hidden)
        self.linear2 = nn.Linear(n_hidden, out_features)
        
    def forward(self, x):                                   
        z1 = self.normalize1(self.linear1(x))
        p1 = torch.tanh(z1)                   
        out = self.linear2(p1)
        return out

# tanh1后置BN层
class tanh_class1_norm2(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden=4, out_features=1):       
        super(tanh_class1_norm2, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden)
        self.normalize1 = nn.BatchNorm1d(n_hidden)
        self.linear2 = nn.Linear(n_hidden, out_features)
        
    def forward(self, x):                                   
        z1 = self.linear1(x)
        p1 = torch.tanh(z1)                   
        out = self.linear2(self.normalize1(p1))
        return out    

# tanh2前置BN层
class tanh_class2_norm1(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden1=4, n_hidden2=4, out_features=1):       
        super(tanh_class2_norm1, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1)
        self.normalize1 = nn.BatchNorm1d(n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.normalize2 = nn.BatchNorm1d(n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, out_features) 
        
    def forward(self, x):                                    
        z1 = self.normalize1(self.linear1(x))
        p1 = torch.tanh(z1)
        z2 = self.normalize2(self.linear2(p1))
        p2 = torch.tanh(z2)
        out = self.linear3(p2)
        return out    
    
# tanh2后置BN层
class tanh_class2_norm2(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden1=4, n_hidden2=4, out_features=1):       
        super(tanh_class2_norm2, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1)
        self.normalize1 = nn.BatchNorm1d(n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.normalize2 = nn.BatchNorm1d(n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, out_features) 
        
    def forward(self, x):                                    
        z1 = self.linear1(x)
        p1 = torch.tanh(z1)
        z2 = self.linear2(self.normalize1(p1))
        p2 = torch.tanh(z2)
        out = self.linear3(self.normalize2(p2))
        return out

# tanh3前置BN层
class tanh_class3_norm1(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden1=4, n_hidden2=4, n_hidden3=4, out_features=1):       
        super(tanh_class3_norm1, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1)
        self.normalize1 = nn.BatchNorm1d(n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.normalize2 = nn.BatchNorm1d(n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, n_hidden3)
        self.normalize3 = nn.BatchNorm1d(n_hidden3)
        self.linear4 = nn.Linear(n_hidden3, out_features) 
        
    def forward(self, x):                                    
        z1 = self.normalize1(self.linear1(x))
        p1 = torch.tanh(z1)
        z2 = self.normalize2(self.linear2(p1))
        p2 = torch.tanh(z2)
        z3 = self.normalize3(self.linear3(p2))
        p3 = torch.tanh(z3)
        out = self.linear4(p3)
        return out
    
# tanh3后置BN层
class tanh_class3_norm2(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden1=4, n_hidden2=4, n_hidden3=4, out_features=1):       
        super(tanh_class3_norm2, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1)
        self.normalize1 = nn.BatchNorm1d(n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.normalize2 = nn.BatchNorm1d(n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, n_hidden3)
        self.normalize3 = nn.BatchNorm1d(n_hidden3)
        self.linear4 = nn.Linear(n_hidden3, out_features) 
        
    def forward(self, x):                                    
        z1 = self.linear1(x)
        p1 = torch.tanh(z1)
        z2 = self.linear2(self.normalize1(p1))
        p2 = torch.tanh(z2)
        z3 = self.linear3(self.normalize2(p2))
        p3 = torch.tanh(z3)
        out = self.linear4(self.normalize3(p3))
        return out
    
# ReLU1前置BN层
class ReLU_class1_norm1(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden=4, out_features=1, bias = True):       
        super(ReLU_class1_norm1, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden, bias=bias)
        self.normalize1 = nn.BatchNorm1d(n_hidden)
        self.linear2 = nn.Linear(n_hidden, out_features, bias=bias)
        
    def forward(self, x):                                   
        z1 = self.linear1(self.normalize1(x))
        p1 = torch.relu(z1)                  
        out = self.linear2(p1)
        return out
    
# ReLU1后置BN层
class ReLU_class1_norm2(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden=4, out_features=1, bias = True):       
        super(ReLU_class1_norm2, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden, bias=bias)
        self.normalize1 = nn.BatchNorm1d(n_hidden)
        self.linear2 = nn.Linear(n_hidden, out_features, bias=bias)
        
    def forward(self, x):                                   
        z1 = self.linear1(x)
        p1 = torch.relu(z1)                  
        out = self.linear2(self.normalize1(p1))
        return out
    
# ReLU2前置BN层
class ReLU_class2_norm1(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden_1=4, n_hidden_2=4, out_features=1, bias=True):       
        super(ReLU_class2_norm1, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden_1, bias=bias)
        self.normalize1 = nn.BatchNorm1d(n_hidden1)
        self.linear2 = nn.Linear(n_hidden_1, n_hidden_2, bias=bias)
        self.normalize2 = nn.BatchNorm1d(n_hidden2)
        self.linear3 = nn.Linear(n_hidden_2, out_features, bias=bias)
        
    def forward(self, x):                                   
        z1 = self.normalize1(self.linear1(x))
        p1 = torch.relu(z1)
        z2 = self.normalize2(self.linear2(p1))
        p2 = torch.relu(z2)
        out = self.linear3(p2)
        return out

# ReLU2后置BN层
class ReLU_class2_norm2(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden_1=4, n_hidden_2=4, out_features=1, bias=True):       
        super(ReLU_class2_norm2, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden_1, bias=bias)
        self.normalize1 = nn.BatchNorm1d(n_hidden1)
        self.linear2 = nn.Linear(n_hidden_1, n_hidden_2, bias=bias)
        self.normalize2 = nn.BatchNorm1d(n_hidden2)
        self.linear3 = nn.Linear(n_hidden_2, out_features, bias=bias)
        
    def forward(self, x):                                   
        z1 = self.linear1(x)
        p1 = torch.relu(z1)
        z2 = self.linear2(self.normalize1(p1))
        p2 = torch.relu(z2)
        out = self.linear3(self.normalize2(p2))
        return out

# ReLU3前置BN层
class ReLU_class3_norm1(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden1=4, n_hidden2=4, n_hidden3=4, out_features=1, bias=True):       
        super(ReLU_class3_norm1, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1, bias=bias)
        self.normalize1 = nn.BatchNorm1d(n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2, bias=bias)
        self.normalize2 = nn.BatchNorm1d(n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, n_hidden3, bias=bias)
        self.normalize3 = nn.BatchNorm1d(n_hidden3)
        self.linear4 = nn.Linear(n_hidden3, out_features, bias=bias) 
        
    def forward(self, x):                                    
        z1 = self.normalize1(self.linear1(x))
        p1 = torch.relu(z1)
        z2 = self.normalize2(self.linear2(p1))
        p2 = torch.relu(z2)
        z3 = self.normalize3(self.linear3(p2))
        p3 = torch.relu(z3)
        out = self.linear4(p3)
        return out

# ReLU3后置BN层
class ReLU_class3_norm2(nn.Module):                                   
    def __init__(self, in_features=2, n_hidden1=4, n_hidden2=4, n_hidden3=4, out_features=1, bias=True):       
        super(ReLU_class3_norm2, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1, bias=bias)
        self.normalize1 = nn.BatchNorm1d(n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2, bias=bias)
        self.normalize2 = nn.BatchNorm1d(n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, n_hidden3, bias=bias)
        self.normalize3 = nn.BatchNorm1d(n_hidden3)
        self.linear4 = nn.Linear(n_hidden3, out_features, bias=bias) 
        
    def forward(self, x):                                    
        z1 = self.linear1(x)
        p1 = torch.relu(z1)
        z2 = self.linear2(self.normalize1(p1))
        p2 = torch.relu(z2)
        z3 = self.linear3(self.normalize2(p2))
        p3 = torch.relu(z3)
        out = self.linear4(self.normalize3(p3))
        return out
    
    
    
    
class net_class1(nn.Module):                                   
    def __init__(self, act_fun= torch.relu, in_features=2, n_hidden=4, out_features=1, bias=True, BN_model=None, momentum=0.1):       
        super(net_class1, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden, bias=bias)
        self.normalize1 = nn.BatchNorm1d(n_hidden, momentum=momentum)
        self.linear2 = nn.Linear(n_hidden, out_features, bias=bias)
        self.BN_model = BN_model
        self.act_fun = act_fun
        
    def forward(self, x):
        if self.BN_model == None:
            z1 = self.linear1(x)
            p1 = self.act_fun(z1)                  
            out = self.linear2(p1)
        elif self.BN_model == 'pre':
            z1 = self.normalize1(self.linear1(x))
            p1 = self.act_fun(z1)                  
            out = self.linear2(p1)
        elif self.BN_model == 'post':
            z1 = self.linear1(x)
            p1 = self.act_fun(z1)                  
            out = self.linear2(self.normalize1(p1))
        return out    

class net_class2(nn.Module):                                   
    def __init__(self, act_fun= torch.relu, in_features=2, n_hidden1=4, n_hidden2=4, out_features=1, bias=True, BN_model=None, momentum=0.1):       
        super(net_class2, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1, bias=bias)
        self.normalize1 = nn.BatchNorm1d(n_hidden1, momentum=momentum)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2, bias=bias)
        self.normalize2 = nn.BatchNorm1d(n_hidden2, momentum=momentum)
        self.linear3 = nn.Linear(n_hidden2, out_features, bias=bias)
        self.BN_model = BN_model
        self.act_fun = act_fun
        
    def forward(self, x): 
        if self.BN_model == None:
            z1 = self.linear1(x)
            p1 = self.act_fun(z1)
            z2 = self.linear2(p1)
            p2 = self.act_fun(z2)
            out = self.linear3(p2)
        elif self.BN_model == 'pre':
            z1 = self.normalize1(self.linear1(x))
            p1 = self.act_fun(z1)
            z2 = self.normalize2(self.linear2(p1))
            p2 = self.act_fun(z2)
            out = self.linear3(p2)
        elif self.BN_model == 'post':
            z1 = self.linear1(x)
            p1 = self.act_fun(z1)
            z2 = self.linear2(self.normalize1(p1))
            p2 = self.act_fun(z2)
            out = self.linear3(self.normalize2(p2))
        return out
    
class net_class3(nn.Module):                                   
    def __init__(self, act_fun= torch.relu, in_features=2, n_hidden1=4, n_hidden2=4, n_hidden3=4, out_features=1, bias=True, BN_model=None, momentum=0.1):       
        super(net_class3, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1, bias=bias)
        self.normalize1 = nn.BatchNorm1d(n_hidden1, momentum=momentum)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2, bias=bias)
        self.normalize2 = nn.BatchNorm1d(n_hidden2, momentum=momentum)
        self.linear3 = nn.Linear(n_hidden2, n_hidden3, bias=bias)
        self.normalize3 = nn.BatchNorm1d(n_hidden3, momentum=momentum)
        self.linear4 = nn.Linear(n_hidden3, out_features, bias=bias)
        self.BN_model = BN_model
        self.act_fun = act_fun
        
    def forward(self, x):  
        if self.BN_model == None:
            z1 = self.linear1(x)
            p1 = self.act_fun(z1)
            z2 = self.linear2(p1)
            p2 = self.act_fun(z2)
            z3 = self.linear3(p2)
            p3 = self.act_fun(z3)
            out = self.linear4(p3)
        elif self.BN_model == 'pre':
            z1 = self.normalize1(self.linear1(x))
            p1 = self.act_fun(z1)
            z2 = self.normalize2(self.linear2(p1))
            p2 = self.act_fun(z2)
            z3 = self.normalize3(self.linear3(p2))
            p3 = self.act_fun(z3)
            out = self.linear4(p3)
        elif self.BN_model == 'post':
            z1 = self.linear1(x)
            p1 = self.act_fun(z1)
            z2 = self.linear2(self.normalize1(p1))
            p2 = self.act_fun(z2)
            z3 = self.linear3(self.normalize2(p2))
            p3 = self.act_fun(z3)
            out = self.linear4(self.normalize3(p3))
        return out
    
class net_class4(nn.Module):                                   
    def __init__(self, act_fun= torch.relu, in_features=2, n_hidden1=4, n_hidden2=4, n_hidden3=4, n_hidden4=4, out_features=1, bias=True, BN_model=None, momentum=0.1):       
        super(net_class4, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1, bias=bias)
        self.normalize1 = nn.BatchNorm1d(n_hidden1, momentum=momentum)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2, bias=bias)
        self.normalize2 = nn.BatchNorm1d(n_hidden2, momentum=momentum)
        self.linear3 = nn.Linear(n_hidden2, n_hidden3, bias=bias)
        self.normalize3 = nn.BatchNorm1d(n_hidden3, momentum=momentum)
        self.linear4 = nn.Linear(n_hidden3, n_hidden4, bias=bias)
        self.normalize4 = nn.BatchNorm1d(n_hidden4, momentum=momentum)
        self.linear5 = nn.Linear(n_hidden4, out_features, bias=bias) 
        self.BN_model = BN_model
        self.act_fun = act_fun
        
    def forward(self, x):
        if self.BN_model == None:
            z1 = self.linear1(x)
            p1 = self.act_fun(z1)
            z2 = self.linear2(p1)
            p2 = self.act_fun(z2)
            z3 = self.linear3(p2)
            p3 = self.act_fun(z3)
            z4 = self.linear4(p3)
            p4 = self.act_fun(z4)
            out = self.linear5(p4)
        elif self.BN_model == 'pre':
            z1 = self.normalize1(self.linear1(x))
            p1 = self.act_fun(z1)
            z2 = self.normalize2(self.linear2(p1))
            p2 = self.act_fun(z2)
            z3 = self.normalize3(self.linear3(p2))
            p3 = self.act_fun(z3)
            z4 = self.normalize4(self.linear4(p3))
            p4 = self.act_fun(z4)
            out = self.linear5(p4)
        elif self.BN_model == 'post':
            z1 = self.linear1(x)
            p1 = self.act_fun(z1)
            z2 = self.linear2(self.normalize1(p1))
            p2 = self.act_fun(z2)
            z3 = self.linear3(self.normalize2(p2))
            p3 = self.act_fun(z3)
            z4 = self.linear4(self.normalize3(p3))
            p4 = self.act_fun(z4)
            out = self.linear5(self.normalize4(p4))
        return out  
    
    
def fit_rec(net, 
            criterion, 
            optimizer, 
            train_data,
            test_data,
            epochs = 3, 
            cla = False, 
            eva = mse_cal):
    """模型训练函数（记录每一次遍历后模型评估指标）
    
    :param net：待训练的模型 
    :param criterion: 损失函数
    :param optimizer：优化算法
    :param train_data：训练数据
    :param test_data: 测试数据 
    :param epochs: 遍历数据次数
    :param cla: 是否是分类问题
    :param eva: 模型评估方法
    :return：模型评估结果
    """
    train_l = []
    test_l = []
    for epoch  in range(epochs):
        net.train()
        for X, y in train_data:
            if cla == True:
                y = y.flatten().long()          # 如果是分类问题，需要对y进行整数转化
            yhat = net.forward(X)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        net.eval()
        train_l.append(eva(train_data, net).detach())
        test_l.append(eva(test_data, net).detach())
    return train_l, test_l


def fit_rec_sc(net, 
               criterion, 
               optimizer, 
               train_data,
               test_data,
               scheduler,
               epochs = 3, 
               cla = False, 
               eva = mse_cal):
    """加入学习率调度后的模型训练函数（记录每一次遍历后模型评估指标）
    
    :param net：待训练的模型 
    :param criterion: 损失函数
    :param optimizer：优化算法
    :param train_data：训练数据
    :param test_data: 测试数据 
    :param scheduler: 学习率调度器
    :param epochs: 遍历数据次数
    :param cla: 是否是分类问题
    :param eva: 模型评估方法
    :return：模型评估结果
    """
    train_l = []
    test_l = []
    for epoch  in range(epochs):
        net.train()
        for X, y in train_data:
            if cla == True:
                y = y.flatten().long()          # 如果是分类问题，需要对y进行整数转化
            yhat = net.forward(X)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        net.eval()
        train_l.append(eva(train_data, net).detach())
        test_l.append(eva(test_data, net).detach())
    return train_l, test_l
    
