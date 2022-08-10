# 先处理数据,比如预测"人们是否会买保险",预测的结果是"yes或no",所以先把"yes"和"no"数字化成1和0.这里已经把数据处理好,直接导入数据就行

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# 使用另一个线性回归模型LogisticRegression
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("E:\Projects\PythonProjects\MachineLearning\ LogisticRegression\DataSet\insurance_data.csv")
plt.scatter(df.age, df.bought_insurance, marker='+', color='red')
plt.show()

# 将数据集分成"训练数据集"和"测试数据集"
x_train, x_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, test_size=0.1)
model = LogisticRegression()
model.fit(x_train, y_train)
print("预测x_test:")
print(model.predict(x_test))
print("更加详细的预测结果:")
print(model.predict_proba(x_test))
print("y_test:")
print(y_test)