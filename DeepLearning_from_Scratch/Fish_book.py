##该书是一个入门教程，涉及大量的基础理论知识，适合于实操教程双线作战，最大化学习效率
###chapter 1：基本python入门，Numpy的广播特性，matplotlib画图的基本逻辑
###chapter 2：感知机的含义和原理，逻辑电路，多层感知机的实现（多层门控），以及如何用python进行实现。
###chapter 3：神经网络入门

###激活函数: h(x),将输入的信号的总和，即前述感知机的计算结果，利用一定的计算规则产生输出结果y 。
import numpy as np ##提供高性能矩阵数据处理的package
import matplotlib.pyplot as plt ##作图的包
import pandas as pd##数据结构和数据分析包
from ipywidgets import Password


### sigmoid()函数 h(x)= 1/[1+exp(-x)]
### 阶跃函数，像阶梯一样突变的函数
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
### X 只能为实数，下述为nparray格式的形式

def step_function_np(x):
    y = x > 0
    return y.astype(np.int_)##这里相比较于书上由np.int 变成了 np.int_,也可以指定参数 dtype=int实现

def step_function_np2(x):
    return np.array(x > 0, dtype=np.int_)

# x = np.arange(-5, 5, 0.1)
# y = step_function_np2(x)
# plt.plot(x, y)##画图
# plt.ylim(-0.1, 1.1)###指定y轴
# plt.show()

##sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
###这里得益于nparray的广播特性，该函数的输出也是一个nparray数据，类似于R的不匹配长度的列表的自动对其/补全计算。
# x = np.array([1, 2, 3, 4, 5])
# y = sigmoid(x)
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

###sigmoid函数比较阶跃函数的特点就是其平滑性，对于神经网络学习具有重要意义。
###不论是阶跃还是sigmoid，都是非linear function，而神经网络的激活函数不能使用linear，如果是，则加深层数将会没有意义，因为n层的线性可以直接被一个函数表示。

##ReLU(Rectified Linear Unit)函数是现在更常用的函数相较于早期的sigmoid
###ReLU 在函数大于0，输出该数，否则输出
def relu(x):
    return np.maximum(0, x)
x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-0.1, 5)
plt.show()

###3.3 多维数组的运算

import numpy as np
A = np.array([1,2,3,4,5])

np.ndim(A) ## 查看数组的维数
A.shape ##形状，就是查看数组的结构。是一个元组变量，每个数字代表这个结构内的元素数量，是按照一，二，三维...的方式排序的。
A.shape[0]
B = np.array([[1,2],[3,4],[5,6]])
np.ndim(B)
B.shape
print(B)
### 维度就是由大到小结构的数量
### 2维的就是我们说的矩阵
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
np.dot(A, B)
###矩阵运算，两个矩阵的乘积。 感觉一维数组像个万精油，根据所称的另一个矩阵，可以自由变化为 1 * n或者 n * 1
### A * B != B * A, A * B = C, C的row为A row，line为B line

###神经网络的内积，通过矩阵实现神经网络,多层计算的一种简洁实现方式，优于循环语法
##例如由 x1 x2 到 y1 y2 y3的一层网络 权重参数为 w1 w2 w3 w4 w5 w6
X = np.array([1,2])
w1,w2,w3,w4,w5,w6 = 1,2,3,4,5,6
W = np.array([[w1,w3,w5],[w2,w4,w6]])
Y = np.dot(X, W)
print(Y)

###符号规则：权重W 1,2 ^(1): (1)代表是第一层的权重，1，2，代表是哪些神经元，1，表示后一层的第一个神经元，2代表前一层的第二个神经元。
### 实现： A = X*W + B

X = np.array([1,2])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])
print(X.shape) ##(2,)
print(W1.shape)##(2,3)
print(B1.shape)##(3,)
A1 = np.dot(X, W1) + B1
print(A1)
###产生的输出结果通过我们的激活函数形成第二层的输入。
Z1 = sigmoid(A1)
print(Z1)
###类似的，可以继续写第二层的信号传递。
W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2 = np.array([0.1,0.2])
print(W2.shape)##(3,2)
print(B2.shape)##(2,)
A2 = np.dot(Z1, W2) + B2
print(A2)
Z2 = sigmoid(A2)
print(Z2)
###最后就是由第二层到输出层的传递
W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])
print(W3.shape)
print(B3.shape)
A3 = np.dot(Z2, W3) + B3
print(A3)
def identity_function(x):
    return x

Z3 = np.dot(X, W3) + B3
Y = identity_function(Z3) ### Y  是我们的最终的输出结果，输出层的激活函数用σ()表示，这里是恒等函数（即不改变）
print(Y)
### 这里，除了输入层和输出层，其他的层都成为隐藏层。隐藏层的激活函数用h()表示。

###小结汇总：

def init_network():
    network = {'W1': W1, 'W2': W2, 'W3': W3, 'B1': B1, 'B2': B2, 'B3': B3}
    ###这里方便起见，我直接使用前面的变量。
    return network

def forward(network,X):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    B1, B2, B3 = network['B1'], network['B2'], network['B3']

    a1 = np.dot(X, W1) + B1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + B2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + B3
    Y = identity_function(a3)
    return Y
network = init_network()
X = np.array([1,2])
Y = forward(network,X)
print(Y)

### forward 意思为向前，这里就是向前传播的意思。即由输入到输出的过程，还有反向传播，由输出到输入

###输出层的设计： nn可以用于分类和回归（预测），但要根据情况改变输出层的激活函数，一般，回归问题用恒等，分类问题用softmax
### softmax function: yk = exp(ak)/∑exp(ai), 因此相较于恒等的单指向输出，softmax的每个输出收到所有输入信号的影响

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
x = np.array([1,2,3,4,5])
y = softmax(x)
print(y)
print(np.sum(y)) ##1
### 问题：softmax函数往往exp(a)会非常大，导致数超出计算机运行的规定的范围，导致so called：溢出 的现象，需要时刻注意。
###imprvoment : 通过数学计算可知，a加减常数C后进行运算并不会改变softmax函数的输出结果，所以我们可以将a减去其中的max value，防止溢出

def softmax_improve(a):
    C = np.max(a)
    exp_a = np.exp(a-C)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
x1 = np.array([1001,1002,1003,1004,1005])
x2 = np.array([1,2,3,4,5])
y = softmax_improve(x1)
print(y)
y = softmax_improve(x2)
print(y)###两次输出一致

###这里，softmax的元素和为1，每个元素都介于(0,1),因此可以将其视为一种概率模型，而不同元素的值就代表了这个可能的概率，所以可以用于分类,而神经元的灰度，就是这个概率的一种体现
###但是因为exp本身为单调函数，而运行分类识别的时候，根本不会影响输出层各个神经元的大小关系，而实际使用是，可以忽略softmax直接输出最大神经元，从而节约计算资源。
###机器学习分为学习和推理 两个步骤，用train数据进行学习，test数据用于学习后建立的模型的推理, the same as nerou network
###in fact 学习就是获取我们这里未知的最佳权重等参数的过程
###因此这里不难理解，输出层的神经元数量，要根据解决的问题决定，例如在分类问题中，其数量应该和分类的类数一致

###3.6 手写识别
### dataset: MNIST images of num 1-10, train_set size 60000, test_set size 10000































