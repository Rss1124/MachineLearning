# 预测多元函数的线性回归模型

import pandas as pd
import numpy as np
import math
from sklearn import linear_model

""" 导入训练数据集 """
df = pd.read_csv("E:\Projects\PythonProjects\MachineLearning\SupervisedMachineLearning\Tutorial_LinearRegressionTutorial\DataSet\homeprice_complex_data.csv")
# 处理数据集中的无效数据(bedroom中的nan):使用bedroom的中位数进行填补
median_bedrooms = math.floor(df.bedrooms.median())
df.bedrooms = df.bedrooms.fillna(median_bedrooms)

""" 训练模型 """
reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df.price)
# 进行预测
print(reg.predict([[3000, 3, 40]]))