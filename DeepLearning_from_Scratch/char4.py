###引入损失函数来评估神经网络的性能
###损失函数的值越大，代表模型的性能越差
###常见两种损失函数：均方误差和交叉熵误差

import numpy as np
from pycparser import ply


###均方误差：E=1/2∑(yk-tk)**2(y为预测值，t为监督值（即训练数据的正确值）)
def mean_squared_error(t_true, y_pred):
    return 0.5 * np.sum((t_true - y_pred)**2)

###交叉熵误差：E=-∑(tk*log(yk))，注意这里是向量积的形式,所以yk越接近1，E越小
def cross_entropy_error(t_true, y_pred):
    delta = 1e-7
    return np.sum(-t_true * np.log(y_pred + delta))###加入微小值避免计算溢出

t = np.array([0,0,1,0,0,0,0,0,0,0])
y = np.array([0.1,0.1,0.6,0.1,0.1,0,0,0,0,0])
print(mean_squared_error(t, y))
print(cross_entropy_error(t, y))

###机器学习的过程就是找到一个使得损失函数足够小的参数，如果训练函数有100个，那么就要让这100个数据的损失函数的和最小。
###例如E=-∑∑(tnk*log(ynk))/N,除N得到平均误差函数。
###而一次性如果使用太多的数据会造成运算量过大，例如MNIST 60000个数据运算较大，可以改为每次随机选择100个训练，这种训练方式就是mini-batch训练

import sys,os
sys.path.append(os.pardir)
import numpy as np
from DeepLearning_from_Scratch.code.dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)###one hot，t数组只有正解为1，其余为0
print(x_train.shape)
print(x_test.shape)
###np.random.choice()随机选择，当然，torchvision的dataloader也可以方便实现。
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print(x_batch.shape)
print(t_batch.shape)

def cross_entropy_error(t_true, y_pred):
    delta = 1e-7
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, y_pred.size)
        t_true = t_true.reshape(1, t_true.size)
        ####if函数防止当输入一个数据时，batchsize和数据数量不匹配
    batch_size = t_true.shape[0]
    return -np.sum(t_true * np.log(y_pred + delta))/batch_size

def cross_entropy_error(t_true, y_pred):
    delta = 1e-7
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, y_pred.size)
        t_true = t_true.reshape(1, t_true.size)
        ####if函数防止当输入一个数据时，batchsize和数据数量不匹配
    batch_size = t_true.shape[0]
    return -np.sum( np.log(y_pred[np.arange(batch_size),t] + delta))/batch_size
###改进后，可以处理t不是onehot形式的数据

###引入损失函数的目的是为了方便寻找参数，因为参数的求需要计算参数的导数（当然，因为这里参数众多，所以针对单独的一个参数，我们这里球的应该是梯度），找到最小值的地方。
###重要的是，识别精度对参数的变化不灵敏，而损失函数对其变化很灵敏，是连续变化的。
###微分，求导。数值微分，中值微分
def numerical_diff(f,x):
    h = 1e-4#0.0001
    return (f(x+h) - f(x-h))/(2*h)
###与真实的导数有差异
def function_1(x):
    return 0.01*x**2 + 0.1*x
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0, 20.0, 0.1)
y = function_1(x)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, y)
plt.show()

###计算5和10处的导数
print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))

###偏导数
def function_2(x):
    return x[0]**2 + x[1]**2
    ###reture np.sum(x**2)
###求偏导就是固定一个变量，其他当常量
###而由全部变量的偏导数构成的一个新的向量就成为梯度（gradient）
### f(x)=x1**2 + x2**2 梯度计算

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)###生成一个形状和X一样的元素都为0的数组。
    for i in range(x.size):
        temp_x = x[i]
        x[i] = temp_x + h
        print(temp_x,x)
        fxh1 = f(x)###f(x+h)

        x[i] = temp_x - h
        print(temp_x,x)
        fxh2 = f(x)###f(x-h)
        grad[i] = (fxh1 - fxh2)/(2*h)
        x[i] = temp_x###复原，用于下个循环
    return grad

x = np.array([1.0,1.0])###注意，一定要有float，要不然输入1，2，3默认int，计算是h这个微小值就会被忽略，计算就全错了
grad = numerical_gradient(function_2, x)
print(grad)

###梯度表示的是歌典出的函数值减小最多的方向，但是无法保证指向的一定是最小值或者应该前进的真正的方向。
###函数的极小值最小值和鞍点，梯度为0，梯度法所求是梯度最小的地方，但不代表这个点是最小值，学习高原：复杂函数的学习进入平坦地界，无法前进
###机器学习和深度学习的梯度法就是不断的沿着梯度前进，不断计算新的梯度值，并最终减小函数值的过程。是最常用的方法
###根据目的不同寻找最大or最小值，分别叫梯度梯度上升和梯度下降法。但是神经网络一般是梯度下降法

###数学表达式表示梯度： x0 = x0 - ηf'(x0)     x1 = x1 - nf'(x1)
### n 学习率：更新量，决定在一次学习中，应该学习多少，以及在多大程度上更新参数

def gradient_descent(f, init_x, lr=0.01, step_num=100):###lr learn rate 学习率,f 为需要最优化的函数。init_X 为初始值。step是重复次数。
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x
###这个可以求极小值，顺利的话就是局部最小值。
###求f(x0+x1)= x1**2 + x2**2 的最小值。
def function_2(x):
    return x[0]**2 + x[1]**2
init_x = np.array([-3.0,4.0])
gradient_descent(function_2, init_x, lr=0.1, step_num=100)
###为什么不是return f(x)呢？我们不是要的他的最小值吗？这里返回的是此时x的取值吧，当然放到深度学习中就是我们要的参数。
###学习率的取值非常重要，要选择合适的学习率，学习率这样的参数也成为超参数，和权重和偏置不同，后两者是学习得到的，而前者是指定的






















