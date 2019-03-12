# -*- coding: utf-8 -*-
# Date: 2019-02-25
# Author: juingzhou
# Email: juingzhou@163.com
'''
-----------------------------
数据集: MNIST
训练集数量: 60000
测试集数量: 10000
准确率: 0.8141
运行时长: 88.9s
-----------------------------
'''
import numpy as np
import time

def load_data(file_name):
    '''
    加载MNIST数据集
    :param file_name: 要加载的数据集路径
    :return: list 形式的数据集及标记  
    '''
    print('Start to read MNIST data')
    # 存放数据的list
    data_arr = []
    # 存放数据集对应的label的list
    label_arr = []
    # 打开文件
    with open(file_name, 'r') as f:
        for line in f.readlines(): # 按行读取
            cur_line = line.strip().split(',') # 对每行数据按行切割符','进行切割, 返回字段列表
            
            # MNIST 有0-9是个label,由于是二分类任务, 所以将大于等于5的作为1, 小于5的作为-1
            if int(cur_line[0]) >= 5:
                label_arr.append(1)
            else:
                label_arr.append(-1)
            
            # 存放label
            # [int(num) for num in curLine[1:]] -> 遍历每一行中除了以第一哥元素（标记）外将所有元素转换成int类型
            # [int(num)/255 for num in curLine[1:]] -> 将所有数据除255归一化(非必须步骤，可以不归一化)
            data_arr.append([int(num)/255 for num in cur_line[1:]])

    # 返回data和label
    return data_arr, label_arr

def perceptron(data_arr, label_arr,  epoch):
    '''
    感知机训练过程
    :param data_arr:训练集的数据
    :param label_arr:训练集的标签
    :param epoch:迭代次数, 默认次数50
    :return:训练好的w和b
    '''
    print('Start to train')
    # 将数据转换为矩阵形式(在机器学习中因为通常都是向量的运算转换为矩阵形式方便运算)
    # 转换为矩阵形式之后的数据每一行都是一个样本的数据, 而每一列都是对应的特征
    data_mat = np.mat(data_arr) 
    # 将标签转换成矩阵，之后转置(.T为转置)。
    # 转置是因为在运算中需要单独取label中的某一个元素，如果是1xN的矩阵的话，无法用label[i]的方式读取
    # 对于只有1xN的label可以不转换成矩阵，直接label[i]即可，这里转换是为了格式上的统一
    label_mat = np.mat(label_arr).T
    # 获取训练集的大小, 为(m,n)
    m, n = np.shape(data_mat)
    # 初始化权重w为0, 维数为(1,n),这样与每一个样本的数据长保持一致
    w = np.zeros((1,n))
    # 初始化偏置b为0
    b = 0
    # 初始化学习lr
    lr = 0.0001
    # 以上步骤都是训练开始之前的数据预处理


    # 进行迭代训练
    for num in range(epoch):
    # 对于每一个样本进行梯度下降
    # 李航书中在2.3.1开头部分使用的梯度下降，是全部样本都算一遍以后，统一
    # 进行一次梯度下降
    # 在2.3.1的后半部分可以看到（例如公式2.6 2.7），求和符号没有了，此时用
    # 的是随机梯度下降，即计算一个样本就针对该样本进行一次梯度下降。
    # 两者的差异各有千秋，但较为常用的是随机梯度下降。 
        for i in range(m):
            # 获取当前样本的向量
            xi = data_mat[i]
            # 获取当前样本对应的标签值
            yi = label_mat[i]
            # 判断是否为五分类样本
            # 五分类样本的特征为: -yi(w*xi + b) >= 0
            # 在书中的公式写为》0，但是实际上如果=0，说明该点在超平面上，也是不正确的
            if -1 * yi * (w * xi.T + b) >= 0 :
                #对于误分类样本，进行梯度下降，更新w和b
                w += lr * yi * xi
                b += lr * yi
        print('Round %d:%d training'%(num, epoch))
    # 返回训练完的w, b
    return w, b

def test(data_arr, label_arr, w, b):
    '''
    测试准确率
    :param data_arr: 测试集
    :param label_arr: 测试集label
    :param w: 训练获得的权重w
    :param b: 训练获得的偏置b
    :return: 正确率
    '''
    print('Start to test')
    # 将数据集转换为矩阵形式方便运算
    data_mat = np.mat(data_arr)
    # 将label转换为矩阵并转置，详细信息参考上文perceptron中
    # 对于这部分的解说
    label_mat = np.mat(label_arr).T

    # 获取测试集的大小维度
    m, n = np.shape(data_mat)
    # 错误样本数计数
    error_cnt = 0
    # 遍历所有测试计数
    for i in range(m):
        # 获取当个样本的向量
        xi = data_mat[i]
        # 获取该样本的label
        yi = label_mat[i]
        # 获取运行结果, 如果-yi(w*xi + b)>=0, 说明该样本被误分类, 错误样本数加一
        if (-yi * (w*xi.T + b)) >= 0:
            error_cnt += 1
    # 正确率 = 1 - （样本分类错误数 / 样本总数）
    accurate = 1 - (error_cnt / m)
    # 返回正确率
    return accurate

if __name__ == "__main__":
    # 获取当前时间
    # 在文末同样获取时间, 两时间差积为程序运行时间
    start = time.time()

    # 获取训练集及标签
    train_data, train_label = load_data('transmnist.\Mnist\mnist_train.csv')
    test_data, test_label = load_data('transmnist.\Mnist\mnist_test.csv')

    # 训练获得权重以及偏置
    w, b = perceptron(train_data, train_label, 50)
    # 进行测试, 获取正确率
    accurate = test(test_data, test_label, w, b)
    # 获取程序最后运行结束的时间
    end = time.time()
    # 显示正确率
    print('accracy rate is: ', accurate)
    # 显示用时时长
    print('time span: ', end - start)

