# -*- encoding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import csv

# 训练集读取
raw_data = np.genfromtxt("train.csv", delimiter=',')  # 读取训练集数据，以矩阵存储
data = raw_data[1:, 3:]  # 去掉无关信息
where_are_NaNs = np.isnan(data)  # 得到True/False的矩阵，判断哪些地方为空值
data[where_are_NaNs] = 0  # 将空值置为0

# 处理训练集，提取特征
month_to_data = {}  # 字典，key:月份，value: 18x480的数据，18的行代表气体颗粒的种类，480代表一个月前20天24小时
for month in range(12):
    sample = np.empty(shape=(18, 480))
    for day in range(20):
        for hour in range(24):
            sample[:, day*24+hour] = data[18*(month*20+day):18*(month*20+day+1), hour]
    month_to_data[month] = sample

# 再次预处理数据
x = np.empty(shape=(12 * 471, 18 * 9), dtype=float)  # x保存特征
y = np.empty(shape=(12 * 471, 1), dtype=float)       # y保存PM2.5结果
# 每10小时为1笔资料，总共有12x471笔
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month*471+day*24+hour, :] = month_to_data[month][:, day*24+hour:day*24+hour+9].reshape(1, -1)
            y[month*471+day*24+hour, 0] = month_to_data[month][9, day*24+hour+9]

# 数据的normalization
mean = np.mean(x, axis=0)  # 纵轴方向
std = np.std(x, axis=0)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if not std[j] == 0:
            x[i][j] = (x[i][j] - mean[j]) / std[j]

# Gradient descent
dim = x.shape[1] + 1
w = np.zeros(shape=(dim, 1))
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1).astype(float)
learning_rate = np.array([[200]] * dim)
adagrad_sum = np.zeros(shape=(dim, 1))

for T in range(10000): # 迭代次数
    if T % 500 == 0:
        print("T=", T)
        print("Loss:", np.power(np.sum(np.power(x.dot(w) - y, 2)) / x.shape[0], 0.5))
    gradient = (-2) * np.transpose(x).dot(y - x.dot(w))  # 计算各个特征的导数
    adagrad_sum += gradient ** 2   # adagrad
    w = w - learning_rate * gradient / (np.sqrt(adagrad_sum) + 0.0005)

np.save('weight.npy', w)

# 测试集读取
w = np.load('weight.npy')
test_raw_data = np.genfromtxt("test.csv", delimiter=',')
test_data = test_raw_data[:, 2:]
where_are_NaNs = np.isnan(test_data)
test_data[where_are_NaNs] = 0

test_x = np.empty(shape=(240, 18 * 9), dtype=float) # 总共有240笔资料
for i in range(240):
    test_x[i, :] = test_data[18 * i:18 * (i + 1), :].reshape(1, -1)

for i in range(test_x.shape[0]):
    for j in range(test_x.shape[1]):
        if not std[j] == 0:
            test_x[i][j] = (test_x[i][j] - mean[j]) / std[j]

test_x = np.concatenate((np.ones(shape=(test_x.shape[0], 1)), test_x), axis=1).astype(float)
answer = test_x.dot(w)


# Write file
f = open("answer.csv", "w", newline='')
w = csv.writer(f)
title = ['id', 'value']
w.writerow(title)
for i in range(240):
    content = ['id_'+str(i), answer[i][0]]
    w.writerow(content)