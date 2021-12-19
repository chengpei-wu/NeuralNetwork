import numpy as np


class layer:
    # 一层网络
    def __init__(self, n_input, n_output, activation=None, weights=None, bias=None):
        # 初始化参数
        if weights:
            self.weights = weights
        else:
            self.weights = np.zeros((n_input, n_output))
        # 初始化bias
        if bias:
            self.bias = bias
        else:
            self.bias = np.zeros(n_output)
        # 激活函数
        self.activation = activation
        # 激活函数输出
        self.activation_output = None
        self.error = None
        self.delta = None

    def activate(self, X):
        # 前向计算
        r = np.dot(X, self.weights) + self.bias
        # 激活
        self.activation_output = self._apply_activation(r)
        # 返回激活输出
        return self.activation_output

    def _apply_activation(self, r):
        # 激活函数
        if not self.activation:
            return r
        if self.activation == 'ReLU':
            return np.maximum(r, 0)
        if self.activation == 'tanh':
            return np.tanh(r)
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        return r

    def apply_activation_derivative(self, r):
        # 计算激活函数的导数
        if not self.activation:
            return np.ones_like(r)
        elif self.activation == 'relu':
            grad = np.array(r, copy=True)
            grad[r > 0] = 1.
            grad[r <= 0] = 0.
            return grad
        elif self.activation == 'tanh':
            return 1 - r ** 2
        elif self.activation == 'sigmoid':
            return r * (1 - r)
        return r
