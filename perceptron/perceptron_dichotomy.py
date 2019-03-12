from __future__ import print_function
import random
import numpy as np
import matplotlib.pyplot as plt

def sign(y):
    # 该函数主要将输入经过线性函数，之后进行判别是否大于0
    if y>0:
        return 1
    else:
        return -1


def train(train_epoch, train_data, lr):
    w, b = [0, 0], 0
    for i in range(train_epoch):
        x = random.choice(train_data)
        x1, x2, y = x
        if (-y*(sign(x1*w[0] + x2*w[1] +b)))>0:
            w[0] -= lr * -y*x1
            w[1] -= lr * -y*x2
            b -= lr * -y
    return w,b

def plot_point(train_data, w, b):
    plt.figure()
    x1 = np.linspace(0, 8, 100)
    x2 = (-b-w[0]*x1)/w[1]
    plt.plot(x1, x2, color='r', label='y1 data')
    for i in range(len(train_data)):
        if train_data[i][-1] == 1:
            plt.scatter(train_data[i][0], train_data[i][1], s=50)
        else:
            plt.scatter(train_data[i][0], train_data[i][1], marker='x', s=50)
    plt.show()
if __name__ == '__main__':
    train_data1 = [[1, 3, 1], [2, 2, 1], [3, 8, 1], [2, 6, 1]]  # 正样本
    train_data2 = [[2, 1, -1], [4, 1, -1], [6, 2, -1], [7, 3, -1]]  # 负样本
    train_data = train_data1 + train_data2
    w, b = train(50, train_data, 0.01)
    plot_point(train_data, w, b)