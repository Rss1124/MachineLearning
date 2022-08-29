import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

df = pd.read_csv("E:\Projects\PythonProjects\MachineLearning\SupervisedMachineLearning\Tutorial_DecisionTree\DataSet\salaries.csv")

""" 处理数据 """
inputs = df.drop('salary_more_then_100k', axis='columns')
target = df['salary_more_then_100k']
# 因为数据中有文字型变量,所以要用虚拟变量进行处理
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()
# 在数据集后面手动添加三列虚拟变量
inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_company.fit_transform(inputs['job'])
inputs['degree_n'] = le_company.fit_transform(inputs['degree'])
# 删除多余的"文字变量"列
inputs_n = inputs.drop(['company', 'job', 'degree'], axis='columns')

""" 训练模型 """
model = tree.DecisionTreeClassifier()
model.fit(inputs_n, target)