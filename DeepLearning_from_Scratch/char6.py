###chapter 6:和学习相关的技巧
# SGD 随机梯度下降法，也就是之前我们寻找最优参数所使用的方法，但是其效率较低，学习慢
from DeepLearning_from_Scratch.code.ch08.half_float_network import param


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]

###optimizer:进行最优化的人，这里指的就是我们所说的优化函数。
###在很多情况下，梯度函数实际上所指出的方向并不是我们所期望的最小点，所以导致SGD方法移动以“Z”形式，造成效率的降低。
###为了进行改进，接下来介绍Momentum,AdaGrad,Adam三种可以取代SGD的方法。

###Momentum  dong liang
import numpy as np
class MomentumSGD:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

### AddGrad 对学习率进行更新
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] = self.h[key] + grads[key] * grads[key]
            params[key] -= self.lr * grads[key]/(np.sqrt(self.h[key]) + 1e-7)

###Adam 就是上述两个方法的融合版本

class Adam:

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

###6.2 权重设置

import numpy as np
from matplotlib import pyplot as plt
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.random.randn(1000, 100)
node_num = 100
hidden_num = 5
activation = {}

for i in range(hidden_num):
    if i != 0:
        x = activation[i-1]
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
    # w = np.random.randn(node_num, node_num) * 0.01
    z = np.dot(x, w)
    a = sigmoid(z)
    activation[i] = a

for i, a in activation.items():
    plt.subplot(1, len(activation), i + 1)####多个子图在一张图
    plt.title(str(i + 1) + "_layer")
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()

###Xavier 初始值，使用标准差为1/sqrt（n）的高斯分布，n为前一层的神经元数量
w = np.random.randn(node_num, node_num) / np.sqrt(node_num)

###当激活函数为Relu时，使用He初始值

import numpy as np
from matplotlib import pyplot as plt
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def Relu(x):
    mask = (x <= 0)
    x[mask] = 0
    return x

x = np.random.randn(1000, 100)
node_num = 100
hidden_num = 5
activation = {}
h = [0.01, np.sqrt(1/node_num), np.sqrt(2/node_num)]

b = 0
for j in h:
    for i in range(hidden_num):
        if i != 0:
            x = activation[i-1]
        w = np.random.randn(node_num, node_num) * j
        # w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
        # w = np.random.randn(node_num, node_num) * 0.01
        z = np.dot(x, w)
        a = Relu(z)
        activation[i] = a

    for i, a in activation.items():
        plt.subplot(3, len(activation), i + 1 + 5 * b)####多个子图在一张图
        plt.title(str(i + 1) + "_layer")
        plt.hist(a.flatten(), 30, range=(0,1))
        plt.ylim(0, 7000)

    b += 1

 if plt.show():
     print('Yes')

###在每一层中加入新的一层Batch Normalize layer，可以减少对初始权重设置的依赖
###按照MINIBATCH标准化输入or输出







