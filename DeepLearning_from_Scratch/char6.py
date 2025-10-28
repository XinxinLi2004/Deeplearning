###chapter 6:和学习相关的技巧
# SGD 随机梯度下降法，也就是之前我们寻找最优参数所使用的方法，但是其效率较低，学习慢
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]

###optimizer:进行最优化的人，这里指的就是我们所说的优化函数。
###在很多情况下，梯度函数实际上所指出的方向并不是我们所期望的最小点，所以导致SGD方法移动以“Z”形式，造成效率的降低。
###为了进行改进，接下来介绍Momentum,AdaGrad,Adam三种可以取代SGD的方法。














