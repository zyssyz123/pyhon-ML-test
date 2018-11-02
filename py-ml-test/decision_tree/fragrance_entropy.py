# -*- coding: UTF-8 -*-
from math import log


def creatData():
    data = [[0, 0, 0, 0, 'no'],  # 数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['age', 'works', 'houses', 'Credit']  # 分类属性
    return data, labels


def calcShannonEnt(data):  # 计算香浓熵
    num = len(data)
    labelscount = {}
    for feature in data:
        currentLabel = feature[-1]
        if currentLabel not in labelscount.keys():
            labelscount[currentLabel] = 0
        labelscount[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelscount:
        prob = float(labelscount[key]) / num
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt





if __name__ == '__main__':
    data, features = creatData()
    print(data)
    print(calcShannonEnt(data))
