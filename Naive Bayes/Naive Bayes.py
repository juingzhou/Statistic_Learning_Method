# -*- coding: utf-8 -*-
# author:juigzhou
# email:juingzhou@163.com
'''
步骤如下：
1. 计算先验概率(yes和no)
2. 计算每个属性的特征集合对应yes和no概率
3. 计算预测概率
'''



data_list = [
    ['sunny', 'hot', 'high', 'FALSE', 'no'],
    ['sunny', 'hot', 'high', 'TRUE', 'no'],
    ['overcast', 'hot', 'high', 'FALSE', 'yes'],
    ['rainy', 'mild', 'high', 'FALSE', 'yes'],
    ['rainy', 'cool', 'normal', 'FALSE', 'yes'],
    ['rainy', 'cool', 'normal', 'TRUE', 'no'],
    ['overcast', 'cool', 'normal', 'TRUE', 'yes'],
    ['sunny', 'mild', 'high', 'FALSE', 'no'],
    ['sunny', 'cool', 'normal', 'FALSE', 'yes'],
    ['rainy', 'mild', 'normal', 'FALSE', 'yes'],
    ['sunny', 'mild', 'normal', 'TRUE', 'yes'],
    ['overcast', 'mild', 'high', 'TRUE', 'yes'],
    ['overcast', 'hot', 'normal', 'FALSE', 'yes'],
    ['rainy', 'mild', 'high', 'TRUE', 'no']
]


# 计算出现次数

def get_count(index, attrs):
    '''
    :param index: 待比较的索引列表
    :param attrs: 待比较的属性列表
    :return: 出现次数
    '''

    count = 0
    for i in data_list:
        # index是一个list, attrs也是一个list
        if len(index) == 1 and i[index[0]] == attrs[0]:
            count += 1
        else:
            flag = True
            for j in range(len(index)):
                if i[index[j]] != attrs[j]:
                    flag = False

            if flag:
                count += 1

    return count


yes_count = get_count([4], ['yes'])
P_yes, P_no = yes_count / len(data_list), (len(data_list) - yes_count) / len(data_list)
print('先验概率yes:%f,no:%f' % (P_yes, P_no))

# 计算每个属性的特征集合
attr_set_list = []
for i in range(4):
    attr_set = []
    for j in data_list:
        attr_set.append(j[i])

    attr_set_list.append(list(set(attr_set)))
print(attr_set_list)

# 计算每个属性的特征集合对应yes和no概率
predict_dict = {}
for i in range(len(attr_set_list)):
    for j in attr_set_list[i]:
        predict_dict[j] = [(get_count([i, 4], [j, 'yes']) + 1) / (yes_count + len(attr_set_list[i]) - 1),
                           (get_count([i, 4], [j, 'no']) + 1) / (
                                   len(data_list) - yes_count + 1 + len(attr_set_list[i]))]
print(predict_dict)


# 计算预测概率
def get_predict_p(predict_features):
    p_yes, p_no = P_yes, P_no
    for i in predict_features:
        p_yes *= predict_dict[i][0]
        p_no *= predict_dict[i][1]
    return p_yes, p_no


predict_features = ['sunny', 'cool', 'high', 'TRUE']
print("predict:", get_predict_p(predict_features))
