####charpter 5 误差反向传播
###局部运算和链式法则。
###计算图的局部运算与其他部分互不干扰，是非常重要的特征和实现的基础，而链式法则计算导数是数学的基本原理。
###反向传播的计算就是下游传入的参数×节点对各个上游分支的导数
import numpy as np

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













