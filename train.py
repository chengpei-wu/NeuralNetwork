import shelve
import random
import numpy as np
import pandas as pd
from NeuralNetwork import NeuralNetwork
from layer import layer


def init_iris_XY():
    data_path = './data/iris.csv'
    data = pd.read_csv(data_path)
    cols = data.shape[1]
    X = data.iloc[:, 0:cols - 1]
    Y = data.iloc[:, cols - 1:cols]
    x = np.array(X.values)
    y = np.array(Y.values)
    y[y == 'setosa'] = 0
    y[y == 'versicolor'] = 1
    y[y == 'virginica'] = 2
    y = list(map(int, y.reshape((len(y),)).tolist()))
    # 分类问题，将y设置为one-hot编码
    y = np.eye(3)[y]
    return x, y


def init_student_XY():
    # 初始化训练数据
    data_path = './data/Admission_Predict.csv'
    data = pd.read_csv(data_path)[100:400]
    cols = data.shape[1]
    X = data.iloc[:, 1:cols - 1]
    X['GRE Score'] = (X['GRE Score'] - X['GRE Score'].mean()) / X['GRE Score'].std()
    X['TOEFL Score'] = (X['TOEFL Score'] - X['TOEFL Score'].mean()) / X['TOEFL Score'].std()
    Y = data.iloc[:, cols - 1:cols]
    x = np.array(X.values)
    y = np.array(Y.values)
    y[y > 0.7] = 1
    y[y <= 0.7] = 0
    y = list(map(int, y.reshape((len(y),)).tolist()))
    # 分类问题，将y设置为one-hot编码
    y = np.eye(2)[y]
    return x, y


# NN = NeuralNetwork()
# # 设计神经网络：3隐含层
# NN.add_layer(layer(4, 25, 'sigmoid'))
# NN.add_layer(layer(25, 25, 'sigmoid'))
# NN.add_layer(layer(25, 10, 'sigmoid'))
# NN.add_layer(layer(10, 3, 'sigmoid'))
# x, y = init_iris_XY()
# for epoch in range(20000):
#     print(f'epoch:{epoch}')
#     for i in range(len(x)):
#         # index = i
#         index = random.randint(0, len(x) - 1)
#         NN.back_propagation(x[index], y[index], 0.01)
#
# # 将训练好的模型保存到文件中
# layers = NN.layers
# file = shelve.open('./model/iris/layers.dat')
# file['layers'] = layers
# file.close()

NN = NeuralNetwork()
# 设计神经网络：3隐含层
NN.add_layer(layer(7, 25, 'sigmoid'))
NN.add_layer(layer(25, 25, 'sigmoid'))
NN.add_layer(layer(25, 10, 'sigmoid'))
NN.add_layer(layer(10, 2, 'sigmoid'))
x, y = init_student_XY()
for epoch in range(20000):
    print(f'epoch:{epoch}')
    for i in range(len(x)):
        # index = i
        index = random.randint(0, len(x) - 1)
        NN.back_propagation(x[index], y[index], 0.01)

# 将训练好的模型保存到文件中
layers = NN.layers
file = shelve.open('./model/student/layers.dat')
file['layers'] = layers
file.close()
