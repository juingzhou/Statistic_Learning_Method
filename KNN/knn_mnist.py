# -*- coding: utf-8 -*-
# Date: 2019-02-25
# Author: juingzhou
# Email: juingzhou@163.com
'''
-----------------------------
数据集: MNIST
训练集数量: 60000
测试集数量: 10000(实际用1000)
-----------------------------
运行结果:(邻近k数量: 25)
向量距离使用算法-- 欧式距离
    准确率: 98.0 %
    运行时长: 143.7865924835205s
向量距离使用算法--曼哈顿距离
    准确率: 
    运行时长: 
------
'''

import numpy as np
import time 
import operator

def load_data(file_name):
    '''
    加载文件
    :param file_name: 要加载的文件路径
    :return: 数据集和标签集
    '''
    print('Start read file')
    # 存放数据及标签
    data_arr = []
    label_arr = []
    # 使用' with open() as f '形式读取文件可以自动帮我们调用close()方法
    with open(file_name,'r') as f: 
        # 遍历文件中的每一行
        for line in f.readlines():
            # 获取当前行，并按“，”切割成字段放入列表中
            # strip：去掉每行字符串首尾指定的字符（默认空格或换行符）
            # split：按照指定的字符将字符串切割成每个字段，返回列表形式
            cur_line = line.strip().split(',')
            # 将标记信息放入标记集中
            # 放入的同时需要将标记转为整型
            label_arr.append(int(cur_line[0]))
            # 将每一行除标记信息外的数据存放入数据集中
            # 同时需要将原先字符串形式的数据转为整型
            data_arr.append([int(num) for num in cur_line[1:-1]])
    
    # 返回数据集和标记
    return data_arr, label_arr


def calc_dist(x1, x2):
    '''
    计算两个样本点向量之间的距离
    使用的是欧氏距离，即 样本点每个元素相减的平方  再求和  再开方
    欧式举例公式这里不方便写，可以百度或谷歌欧式距离（也称欧几里得距离）
    :param x1:向量1
    :param x2:向量2
    :return:向量之间的欧式距离
    '''
    return np.sqrt(np.sum(np.square(x1 - x2)))
    

def get_closest(train_data_mat, train_label_mat, x, top_k):
    '''
    预测样本x的标记。
    获取方式通过找到与样本x最近的topK个点，并查看它们的标签。
    查找里面占某类标签最多的那类标签
    （书中3.1 3.2节）
    :param train_data_mat:训练集数据集
    :param train_label_mat:训练集标签集
    :param x:要预测的样本x
    :param top_k:选择参考最邻近样本的数目（样本数目的选择关系到正确率，详看3.2.3 K值的选择）
    :return:预测的标记
    '''
    #建立一个存放向量x与每个训练集中样本距离的列表
    #列表的长度为训练集的长度，distList[i]表示x与训练集中第
    ## i个样本的距离
    dist_list = [0] * len(train_label_mat)
    #遍历训练集中所有的样本点，计算与x的距离
    for i in range(len(train_data_mat)):
        #获取训练集中当前样本的向量
        x1 = train_data_mat[i]
        #计算向量x与训练集样本x的距离
        cur_dist = calc_dist(x1, x)
        #将距离放入对应的列表位置中
        dist_list[i] = cur_dist
    # argsort：函数将数组的值从小到大排序后，并按照其相对应的索引值输出
    # 例如：
    #   >>> x = np.array([3, 1, 2])
    #   >>> np.argsort(x)
    #   array([1, 2, 0])
    # 返回的是列表中从小到大的元素索引值，对于我们这种需要查找最小距离的情况来说很合适
    top_k_list = np.argsort(np.array(dist_list))[:top_k]        #升序排序
    # 定一个记录类别次数的字典

    label_list = [0] * 10
    #对topK个索引进行遍历
    for index in top_k_list:
        #train_label_mat[index]：在训练集标签中寻找topK元素索引对应的标记
        #int(train_label_mat[index])：将标记转换为int（实际上已经是int了，但是不int的话，报错）
        #labelList[int(train_label_mat[index])]：找到标记在label_list中对应的位置
        #最后加1，表示投了一票
        label_list[int(train_label_mat[index])] += 1
    #max(label_list)：找到选票箱中票数最多的票数值
    #label_list.index(max(label_list))：再根据最大值在列表中找到该值对应的索引，等同于预测的标记
    return label_list.index(max(label_list))

def test(train_data_arr, train_label_arr, test_data_arr, test_label_arr, top_k):
    # 将所有列表转为矩阵形式, 方便计算

    print('Start Test!')
    train_data_mat = np.mat(train_data_arr)
    train_label_mat = np.mat(train_label_arr).T
    test_data_mat = np.mat(test_data_arr)
    test_label_mat = np.mat(test_label_arr).T


    # 错误值计算
    error_cnt = 0
    # 遍历测试集，对每个测试集样本进行测试
    # 由于计算向量与向量之间的时间耗费太大，测试集有6000个样本，所以这里人为改成了
    # 测试1000个样本点，如果要全跑，将行注释取消，再下一行for注释即可，同时下面的print
    # 和return也要相应的更换注释行
    # for i in range(len(testDataMat)):
    for i in range(100):
        print('test %d:%d'%(i, 100))
        # 读取测试集当前测试样本的向量
        x = test_data_mat[i]
        # 获取预测的标记
        y = get_closest(train_data_mat, train_label_mat, x, top_k)
        print(y)
        # 如果预测标记与实际标记不符, 错误值计数加1
        if y != test_label_mat[i]:
            print(test_label_mat[i])
            error_cnt += 1
        
    # 返回正确率
    return 1 - (error_cnt / 100)


if __name__ == '__main__':
    start = time.time()

    #获取训练集
    train_data_arr, train_label_arr = load_data('G:\\Git_Repo\\Statistic_Learning_Method\\transmnist\\Mnist\\mnist_train.csv')
    #获取测试集
    test_data_arr, test_label_arr = load_data('G:\\Git_Repo\\Statistic_Learning_Method\\transmnist\\Mnist\\mnist_test.csv')
    #计算测试集正确率
    accur = test(train_data_arr, train_label_arr, test_data_arr, test_label_arr, 25)
    #打印正确率
    print('accur is:%s'%(str(accur * 100)), '%')

    end = time.time()
    #显示花费时间
    print('time span:', end - start)