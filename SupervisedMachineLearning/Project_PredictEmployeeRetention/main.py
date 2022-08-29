import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression  # 因为要预测的数据是"是否离开公司",所以本次采用逻辑回归

df = pd.read_csv("E:\Projects\PythonProjects\MachineLearning\SupervisedMachineLearning\Project_PredictEmployeeRetention\DataSet\HR_comma_sep.csv")

""" 处理数据集 """
# 使用LabelEncoder处理文字变量
le_sales = LabelEncoder()
le_salary = LabelEncoder()
# 在数据集后面手动添加两列虚拟变量
df['sales_n'] = le_sales.fit_transform(df['sales'])
df['salary_n'] = le_sales.fit_transform(df['salary'])
# 删除多余的"文字变量"列
final_inputs = df.drop(['left', 'sales', 'salary'], axis='columns')
target = df['left']
# 将数据分成"训练数据"和"测试数据"
x_train, x_test, y_train, y_test = train_test_split(final_inputs, target, test_size=0.1)

""" 绘制出"薪水等级"与"离职"相关的bar图 """
# 统计不同薪水的人的离职数据
left_high = df.loc[(df['salary'] == 'high') & (df['left'] == 1)].count()['salary']  # 薪水为"high"且"离职"的人数总和
left_medium = df.loc[(df['salary'] == 'medium') & (df['left'] == 1)].count()['salary']  # 薪水为"medium"且"离职"的人数总和
left_low = df.loc[(df['salary'] == 'low') & (df['left'] == 1)].count()['salary']  # 薪水为"low"且"离职"的人数总和
# 画图
plt.figure(figsize=(10, 7))
values = (left_low, left_medium, left_high)
bar = plt.bar(np.arange(3), values, width=0.35, color="#87CEFA")
plt.xlabel('salary_level')
plt.ylabel('left_count')
plt.xticks(np.arange(3), ('low', 'medium', 'high'))
plt.yticks(np.arange(0, 2200, 100))
plt.show()

""" 制作逻辑回归模型 """
model = LogisticRegression()
model.fit(x_train, y_train)

""" 将预测过后的数据与y_test一起放在csv里面 """
dataframe = pd.DataFrame({'predict_x_test': model.predict(x_test),
                          'y_test': y_test})
dataframe.to_csv("predict.csv", index=False)
