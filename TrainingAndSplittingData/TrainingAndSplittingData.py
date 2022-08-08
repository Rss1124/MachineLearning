# 将数据集分成训练数据集和测试数据集两部分,然后进行训练

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv("E:\Projects\PythonProjects\MachineLearning\TrainingAndSplittingData\DataSet\carprices.csv")
plt.scatter(df['Mileage'], df['Sell Price'])
plt.show()
plt.scatter(df['Age'], df['Sell Price'])
plt.show()
x = df[['Mileage', 'Age']]
y = df['Sell Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  # 将x,y拆成4个部分,其中test_size表示你要提取n%的数据作为测试数据,在这里x_train和y_train各有16个数据,x_test和y_test各有4个数据
clf = LinearRegression()
clf.fit(x_train, y_train)
print("预测数据如下:")
print(clf.predict(x_test))
print("y_test:")
print(y_test)
