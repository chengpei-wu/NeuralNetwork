import shelve
from NeuralNetwork import NeuralNetwork
import numpy as np
import pandas as pd


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
    data_path = './data/Admission_Predict.csv'
    data = pd.read_csv(data_path)[0:100]
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


# 加载训练好的模型
file = shelve.open('./model/iris/layers.dat')
layers = file['layers']
file.close()
# 加载测试数据
x, y = init_iris_XY()

NN = NeuralNetwork()
NN.layers = layers

m, n = 0, 0
for i in range(len(x)):
    pre = NN.forward_propagation(x[i])
    # print(np.dot(pre, y[i][-1::-1]))
    if np.dot(pre, y[i]) == max(
            np.dot(pre, np.array([0, 0, 1])), np.dot(pre, np.array([0, 1, 0])), np.dot(pre, np.array([1, 0, 0]))):
        print(f'{i}分类成功', pre, y[i])
        n += 1
    else:
        m += 1
        print(f'\n\n\n{i}分类失败', pre, y[i])

print(m / 150, n / 150)

# 加载训练好的模型
file = shelve.open('./model/student/layers.dat')
layers = file['layers']
file.close()
# 加载测试数据
x, y = init_student_XY()

NN = NeuralNetwork()
NN.layers = layers

m, n = 0, 0
for i in range(len(x)):
    pre = NN.forward_propagation(x[i])
    # print(np.dot(pre, y[i][-1::-1]))
    if np.dot(pre, y[i]) == max(
            np.dot(pre, np.array([0, 1])), np.dot(pre, np.array([1, 0]))):
        print(f'{i}分类成功', pre, y[i])
        n += 1
    else:
        m += 1
        print(f'\n\n\n{i}分类失败', pre, y[i])
print(m / 100, n / 100)
