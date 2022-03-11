import tensorflow as tf
import numpy as np

# Example of constants
a = tf.constant(3)
b = tf.constant(4)
c = (a+b)/(1-b)
# print(c)

# Example of variable
string = tf.Variable("this is a string", tf.string)
number = tf.Variable(123, tf.int32)
floating = tf.Variable(3.1415926, tf.float64)
# print(string)
# print(number)
# print(floating)

# Example of Rank
rank1_tensor = tf.Variable(["test"], tf.string)
rank2_tensor = tf.Variable([["test", "ok", "3q"], ["test", "yes", "ok"]], tf.string)
rank3_tensor = tf.Variable([["test", "ok", "3q"], ["test", "yes", "ok"], ["test", "no", "ok"]], tf.string)
# print(rank2_tensor)
# print(rank2_tensor.shape)

# change tensor

"""
ones(x, y, z)
x：数组的个数
y: 数组的行数
z: 数组的列数
ones(2, 2, 3)表示生成2个行为2，列为3的数组

reshape(tensor, shape)
a.shap参数只有2个：reshape(tensor, [x, y])表示把tensor更改为x行y列的矩阵
    如果x或y其中一个为-1,那么它会根据tensor中的元素数量自动算出相应值
    比如说:tensor中有12个元素,如果shape=[-1, 4],那么x最后就会自动得到12/4=3,最后tensor会改变成一个3行4列的矩阵
b.shap参数有3个：reshape(tensor, [n, x, y])表示把tensor更改为n个x行y列的矩阵
    同样的x,y,n任一个都有可能是-1,但有且仅有一个是-1
    比如说:tensor中有12个元素,如果shape=[-1, 2, 2],那么n最后就会自动得到12/2/2=3,最后tensor会变成3个2行2列的矩阵集合
"""
tensor = tf.ones(1, 2, 3)
tensor1 = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], tf.int32)
tensor2 = tf.reshape(tensor1, [-1, 4])
tensor3 = tf.reshape(tensor2, [-1, 2, 2])
print(tensor3)





