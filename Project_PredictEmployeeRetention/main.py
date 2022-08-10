import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # 因为要预测的数据是"是否离开公司",所以本次采用逻辑回归

df = pd.read_csv("E:\Projects\PythonProjects\MachineLearning\Project_PredictEmployeeRetention\DataSet\HR_comma_sep.csv")

""" 处理数据集 """
dummies1 = pd.get_dummies(df.sales)  # 获取虚拟变量
dummies2 = pd.get_dummies(df.salary)  # 获取虚拟变量
merged = pd.concat([df, dummies1, dummies2], axis='columns')  # 将虚拟变量与原生数据集合并
final = merged.drop(['sales', 'salary', 'sales', 'low'], axis='columns')  # 获取最后的数据集

x = final[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident',
        'promotion_last_5years', 'IT', 'RandD', 'accounting', 'hr', 'management', 'marketing', 'product_mng', 'support', 'technical', 'high', 'medium']]
y = final['left']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)  # 将数据分成"训练数据"和"测试数据"

""" 制作逻辑回归模型 """
model = LogisticRegression()
model.fit(x_train, y_train)

""" 将预测过后的数据与y_test一起放在csv里面 """
dataframe = pd.DataFrame({'predict_x_test': model.predict(x_test),
                          'y_test': y_test})
dataframe.to_csv("predict.csv", index=False)
