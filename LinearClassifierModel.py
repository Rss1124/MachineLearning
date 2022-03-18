from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.core.display import clear_output

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')  # y_train就是标签,代表的是"幸存情况"
y_eval = dfeval.pop('survived')

# 将待处理的数据划分为'非数值数据'和'数值数据'
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

# 获取分类后的每项数据可以取到的唯一值,将其保存到feature_columns
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

# 获得用于训练模型的数据集
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # 创建一个可以用于训练的数据集(tf.data.Dataset object)
        ds = ds.cache('E:\Projects\PythonProjects\MachineLearning\cache\TempFile')
        if shuffle:
            ds = ds.shuffle(1000)  # 将数据集顺序打乱
        ds = ds.batch(batch_size).repeat(num_epochs)  # 将数据集撕裂成32批(防止一次性输入到模型的数据过大,内存承受不住)
        return ds  # 返回32批数据集
    return input_function()

train_input_fn = lambda: make_input_fn(dftrain, y_train)
eval_input_fn = lambda: make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# 开始训练模型
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear_output()
print("accuracy: " + str(result['accuracy']))
result = list(linear_est.predict(eval_input_fn))
print("测试数据: " + str(dfeval.loc[3]))
print("测试对象实际是否存活: " + str(y_eval.loc[3]))
print("预测存活率: " + str(result[3]["probabilities"][1]))