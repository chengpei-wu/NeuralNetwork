import numpy as np


class NeuralNetwork:
    # 神经网络
    def __init__(self):
        # 所有网络层
        self.layers = []

    def add_layer(self, layer):
        # 添加网络全连接层
        self.layers.append(layer)

    def forward_propagation(self, X):
        # 前向传播，返回一次前向传播的输出
        for layer in self.layers:
            X = layer.activate(X)
        return X

    def back_propagation(self, X, y, rate):
        # 反向传播
        # 保存整个网络一次前向传播的输出
        output = self.forward_propagation(X)
        # 反向计算，保存每个节点的delta，用于更新参数
        for i in range(len(self.layers))[-1::-1]:
            layer = self.layers[i]
            if layer == self.layers[-1]:
                # 输出层
                # error 为损失函数中的平方项求导
                layer.error = y - output
                # delta 为平方项的导数 * 对激活函数求导
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else:
                # 隐含层
                next_layer = self.layers[i + 1]
                # 由链式求导运算，隐含层的 error 为反向传播的前一层(正向传播的下一层) delta * 参数w
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                # 得到隐含层的 delta
                layer.delta = layer.error * layer.apply_activation_derivative(layer.activation_output)

        # 前向传播得到了每层的输出，反向计算得到了每层的 delta
        # 由链式求导法则化简，损失函数对每个参数w的偏导数 = w左边连接神经元的输入 * w右边连接神经元的 delta
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == 0:
                # 如果是第一个隐含层，则左边的输入为输出的样本特征X
                output = np.atleast_2d(X)
            else:
                # 不是第一层，则输入为上一层的输出
                output = np.atleast_2d(self.layers[i - 1].activation_output)
            # 更新w
            layer.weights += layer.delta * output.T * rate
            layer.bias += layer.delta * rate
