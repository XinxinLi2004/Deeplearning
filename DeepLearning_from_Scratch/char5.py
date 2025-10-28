####charpter 5 误差反向传播
###局部运算和链式法则。
###计算图的局部运算与其他部分互不干扰，是非常重要的特征和实现的基础，而链式法则计算导数是数学的基本原理。
###反向传播的计算就是下游传入的参数×节点对各个上游分支的导数
import numpy as np
import pandas

from DeepLearning_from_Scratch.Fish_book import iters_num
from DeepLearning_from_Scratch.code.common.functions import cross_entropy_error, softmax


###层的实现非为两个接口，forward（）和backward（），前者正向传播，后者反向传播
###乘法层的实现，乘法层为MulLayer类

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self,dout):
        dx = dout * self.y####f = xy 对x求导就是ydx，所以这里反向传播到x就是直接×上y
        dy = dout * self.x

        return dx, dy

apple = 100
apple_num = 2
tax = 1.1
mu_apple = MulLayer()
mu_tax = MulLayer()
###forward
apple_price = mu_apple.forward(apple_num, apple)
price = mu_tax.forward(apple_price, tax)
print(price)

###backward
dprice = 1
dapple_price, dtax = mu_tax.backward(dprice)
dapple, dapple_num = mu_apple.backward(dapple_price)
print(dapple, dapple_num, dtax)

###加法层实现：

class Addlayer:
    def __init__(self):
        self.x = None
        self.y = None
        ###可以用pass替代，因为不像乘法，这里不需要调用self.x
    def forward(self,x,y):
        # self.x = x
        # self.y = y
        out = x + y
        return out

    def backward(self,dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

###现在来实现苹果和橘子的计算

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

mu_apple = MulLayer()
mu_orange = MulLayer()
ad_ao = Addlayer()
mu_tax = MulLayer()
###forward
apple_price = mu_apple.forward(apple_num, apple)
orange_price = mu_orange.forward(orange_num, orange)
add_price = ad_ao.forward(apple_price, orange_price)
price = mu_tax.forward(add_price, tax)
print(add_price,orange_price,add_price,price)
###backward
dprice = 1
dadd_price, dtax = mu_tax.backward(dprice)
dapple_price, dorange_price = ad_ao.backward(dadd_price)
dapple, dapple_num = mu_apple.backward(dapple_price)
dorange, dorange_num = mu_orange.backward(dorange_price)
print(dapple, dorange, dapple_num, dorange_num, dapple_price, dorange_price,dadd_price, dtax)


###激活函数层的实现。
###Relu
###这里的输入X为array数组
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout

        return dx

x = np.array([[1,2],[-1,2]])
x.T
print(x)
relu = Relu()
m = relu.forward(x)
print(relu.mask)
###Relu层就像一个开关一样，控制电路的通断
###sigmoid
###sigmoid的计算层的实现要复杂一些。除了X和+节点外，还有exp和除法节点,这里书上为了公式更加简洁，最终并没有写成对X的导数形式，而是将变量变为y，最后结果为y(1-y)

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self,x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self,dout):
        dx = dout * (1-self.out) * self.out

        return dx

###正向传播的结果保存在实例的out中，反向传播的时候用这个out进行计算

###Affine和Softmax层的实现
###矩阵的乘积运算在几何学中称为仿射变换，故这里成为Affine层
###因为是矩阵，所以变换的核心就是保证矩阵的形状一致。
###因为X*W is (3,),so dL/dy (3,), but dL/dx (2,),所以前者需要*(3,2)
###批版本的Affine,就是形状发生一点改变，但是别的基本一致。
###同时偏置B。因为计算的时候，X*W的每一行，都加上了一个偏置，对应的是Y（N，3），但是B的结构还是（3，），所以需要用np.sum将Y（N,3）变成（3，）

class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self,x):
        self.x = x
        out = np.dot(x,self.W) + self.b
        return out

    def backward(self,dout):
        dx = np.dot(dout,self.W.T)
        self.dW = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis=0)

        return dx


###Softmax-with-Loss层，通常，神经网络的推理可以不需要softmax层，但是一般来说，学习会加入

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,t)

        return self.loss

    def backward(self,dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        ###利用局部运算简化计算。
        return dx

###TwoLayerNet的实现
import sys,os
sys.path.append(os.path.abspath('../..'))
import numpy as np
from DeepLearning_from_Scratch.code.common.layers import *
from DeepLearning_from_Scratch.code.common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.params = {}###初始化权重
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

        ###生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'],self.params['b2'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self,x,t):
        y = self.predict(x)
        return self.last_layer.forward(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self,x,t):
        loss_W = lambda W : self.loss(x,t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.params['b2'])
        return grads

    def gradient(self,x,t):
        ###forward
        self.loss(x,t)
        ###backward
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)###dout不断迭代，生成每层的dout

        grads = {}
        grads["W1"] = self.layers['Affine1'].dW
        grads["b1"] = self.layers['Affine1'].db
        grads["W2"] = self.layers['Affine2'].dW
        grads["b2"] = self.layers['Affine2'].db
        return grads


#####数值微分计算比较慢，耗时间，但是实现起来是比较简单，不易出错的，而反向传播是基于数学推到，计算快，但是比较复杂容易出错，因此往往会通过梯度确认操作来保证正确
#####梯度确认就是比较两种方法算出来的结果是否一致。
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from DeepLearning_from_Scratch.code.dataset.mnist import load_mnist
from DeepLearning_from_Scratch.code.ch05.two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))

###利用误差反向传播进行学习。
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from DeepLearning_from_Scratch.code.dataset.mnist import load_mnist
from DeepLearning_from_Scratch.code.ch05.two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size,1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size, replace=False)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    grad = network.gradient(x_batch, t_batch)
    ###update
    # network.params['W1'] -= learning_rate * grad['W1']
    # network.params['b1'] -= learning_rate * grad['b1']
    # network.params['W2'] -= learning_rate * grad['W2']
    # network.params['b2'] -= learning_rate * grad['b2']
    for key in ['W1', 'b1', 'W2', 'b2']:
        network.params[key] -= learning_rate * grad[key]
    train_loss_list.append(network.loss(x_train, t_train))
    print(i)
    if i % iter_per_epoch == 0:
        train_acc_list.append(network.accuracy(x_train, t_train))
        test_acc_list.append(network.accuracy(x_test, t_test))
















