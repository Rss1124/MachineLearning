from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import functools
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from six.moves import urllib

import tensorflow._api.v2.feature_column as fc
import tensorflow as tf
import tensorflow_datasets as tfds


"""
一元线性回归函数的简单例子
"""
# x = [1, 2, 2.5, 3, 4]
# y = [1, 4, 7, 9, 15]
# plt.plot(x, y, "ro")
# plt.axis([0, 6, 0, 20])
# plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
# plt.show()

"""
两种获取dataset的方法
"""

# 方法一:
# 将csv数据存放在C:\Users\RuoS\.keras\datasets下面
# TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
# TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"
# train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
# test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

# 方法二:
# 直接获取csv文件dftrain和dfeval
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

"""
对数据的一些简单操作
"""

# 直接输出dftrain看里面有些什么数据
# print(dftrain)

# pop()函数:剔除文件中的survived数据
# y_train = dftrain.pop('survived')
# print(dftrain)
# y_eval = dfeval.pop('survived')
# # print(y_eval)

# loc[索引]来查找具体的某一组数据
# print(dftrain.loc[0])

# dftrain["xx"]来查找"xx"这一列的信息
# print(dftrain["sex"])

"""
数据分析
"""

# 年龄的分布情况,绘制成直方图
# dftrain["age"]里面的数据都是int型,可以直接拿来绘制成图形
# plt.hist(dftrain["age"], bins=20, edgecolor="black")
# plt.show()

# 性别的分布情况,绘制成水平条形图
# 因为dftrain["sex"]里面的数据都是str型,所以要用value_counts()将其进行整理,female有多少人,male有多少人,最后绘制成图形
# 绘制水平条形图barh最少要两个参数,第一个参数是y轴的值,第二参数就是数据,用来做水平条形图的x轴
# plt.barh(range(2), dftrain["sex"].value_counts())
# plt.yticks(range(2), ["male", "female"])
# plt.show()

# 优先级的分布情况,绘制成直方图
# height参数用来设置条的宽度
# plt.barh(range(3), dftrain["class"].value_counts(), height=0.3)
# plt.yticks(range(3), ["First", "Second", "Third"])
# plt.show()


# 性别不同时的不同生存率,绘制成水平条形图
grouped = dftrain.groupby("sex")
# 此时grouped是个obj对象,无法直接观察,要想直接观察数据则需要加上.groups
# print(grouped.groups)
# 选择要处理的某一列数据
grouped = grouped["survived"]
# value_counts()会把你选择好的某一列数据再次进行一次分类并进行统计,效果如下:
print(grouped.value_counts())
# mean函数:对你要处理的某一列数据求平均值,在这里[survived]平均值等同于幸存率
plt.barh(range(2), grouped.mean(), height=0.5)
plt.yticks(range(2), ["Female", "Male"])
plt.xlabel("% survive")
plt.show()