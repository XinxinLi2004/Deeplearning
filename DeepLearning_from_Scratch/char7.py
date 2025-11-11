###卷积神经网络，增加卷积层和池化层优点：保留数据的结构特征，用于图象和语音识别
###im2col,将高维数据转换为2维矩阵，利用矩阵计算的高速性进行运算
from collections import OrderedDict

import numpy as np
from numpy.ma.core import reshape

from DeepLearning_from_Scratch.code.common.layers import Relu, Affine, SoftmaxWithLoss
from DeepLearning_from_Scratch.code.common.util import conv_output_size
from code.common.util import im2col, col2im

x1 = np.random.rand(1,3,7,7)
col1 = im2col(x1, 5,5, stride=1, pad=0)
print(col1.shape)

x2 = np.random.rand(10,3,7,7)
col2 = im2col(x2, 5,5, stride=1, pad=0)
print(col2.shape)

###卷积层实现
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self,x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, stride=self.stride, pad=self.pad)
        col_W = self.W.reshape(FN, -1).T ###滤波器展开
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) ###transpose 改变轴的顺序
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

###Pool实现
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self,x):
        N, C, H, W = x.shape
        out_h = int(1 + (H -self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx

###CNN实现
class SimpleConvnet:
    def __init__(self, input_dim =(1,28,28), conv_param = {'filter_num':30,'filter_size':5,'stride':1,'pad':0},hidden_size = 100, output_size = 10, weight_std = 0.01):
        self.input_dim = input_dim
        self.cov_param = cov_param
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weight_std = weight_std
        input_size = input_dim[1]
        filter_size = self.cov_param['filter_size']
        filter_num = self.cov_param['filter_num']
        filter_pad = self.cov_param['pad']
        filter_stride = self.cov_param['stride']
        conv_output_size = (input_size - filter_size + 2 * filter_pad ) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        self.params = {}
        self.params['W1'] = np.random.randn(filter_num, input_dim[0], filter_size, filter_size) * weight_std
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = np.random.randn(pool_output_size, hidden_size) * weight_std
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = np.random.randn(hidden_size, output_size) * weight_std
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h = 2, pool_w = 2, stride = 2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.loss(y, t)

    def gradient(self, x, t):
        ###forward
        self.loss(x, t)

        ###backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grad = {}
        grad['W1'] = self.layers['Conv1'].dW
        grad['b1'] = self.layers['Conv1'].db
        grad['W2'] = self.layers['Affine1'].dW
        grad['b2'] = self.layers['Affine1'].db
        grad['W3'] = self.layers['Affine2'].dW
        grad['b3'] = self.layers['Affine2'].db

        return grad


























