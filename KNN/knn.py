# -*— coding: utf-8 -*-
import numpy as np

import operator


'''
函数说明:knn算法，分类器

parameters:
    inX - 用于分类的数据(测试集)
    dataset - 用于训练的数据(训练集)
    labels - 分类标签
    k - knn算法参数, 选择距离最小的k个点

returns:
    sortedClassCount[0][0] - 分类结果

'''

def classfy0(inX, dataSet, labels, k):
    # numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    # 在列向量方向上重复inX共一次(横向), 行向量方向上重复inX共dataSetSize次(纵向)
    diff = np.tile(inX,(dataSetSize,1)) - dataSet
    # 二维特征相减后平方
    sqDiffMat = diff ** 2
    # sum()所有元素相加, sum(0)列相加, sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方, 计算出距离
    distances = sqDistances ** 0.5
    # 返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    # 定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        voteLabel = labels[sortedDistIndices[i]]
        # 计算类别的次数
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    # key = operator.itemgetter(1)根据字典的值进行排序
    # key = operator.itemgetter(0)根据字典的键进行排序
    # reverse = True 降序排序字典
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别, 即所要分类的类别
    return sortedClassCount[0][0]


def createDataset():
    # 四组二位特征
    group = np.array([[1,101], [5,89], [108,5], [115,8]])
    # 四组特征标签
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels

if __name__ == '__main__':
    # 创建数据集
    group, labels = createDataset()
    # 测试集
    test = [101, 20]
    # KNN分类
    test_class = classfy0(test, group, labels, 3)
    # 打印分类结果
    # print("test_class[0]: ",test_class[0])
    # print("test_class[0][0]: ",test_class[0][0])
    # print("test_class[0][1]: ",test_class[0][1])
    print(test_class)


"""
总结
KNN算法的优缺点:
    优点:
        简单好用, 容易理解, 精度高, 理论成熟, 既可以用来做分类也可以用来做回归
        可用于数值型数据和离散型数据
        训练时间复杂度为O(n), 无数据输入假定
        对异常值不敏感
    
    缺点:
        计算复杂性高, 空间复杂性高
        样本不平衡问题(有些类别的样本数量很多，而有些样本的数量很少)
        一般数值很大的时候不用这个, 计算量太大，但是单个样本又不能太少，否则容易发生误分
        最大缺点是无法给出数据的内在含义

"""
