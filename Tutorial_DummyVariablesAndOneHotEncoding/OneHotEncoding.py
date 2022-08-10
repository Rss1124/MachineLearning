# Topic: 虚拟变量与独热编码
# Description:
# Q: 在制作线性回归模型时遇到了文字型变量(Nominal)该怎么办? A: 使用pandas的虚拟变量或者sklearn库的独热编码
# Q: pandas的虚拟变量是什么? A: pandas会根据原生文字型变量的种类数(n),在原有的数据集上,新增n列数据,用来对应各个文字型变量,用0,1进行赋值
# Q: sklearn库的独热编码是什么? A: sklearn会将原生文字型变量重新以数字赋值(0, 1, 2 ... n),每个种类对应一个数字

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
dfle = pd.read_csv("E:\Projects\PythonProjects\MachineLearning\Tutorial_DummyVariablesAndOneHotEncoding\DataSet\homeprice_nominal_data.csv")
le = LabelEncoder()
print(le.fit_transform(dfle.town))  # 文字变量重新赋值
dfle.town = le.fit_transform(dfle.town)
x = dfle[['town', 'area']].values
y = dfle.price
ohe = ColumnTransformer(
    [('Ohe', OneHotEncoder(), [0])],
    remainder='passthrough'
)
x = ohe.fit_transform(x)  # 再次生成虚拟变量

x = x[:, 1:]  # 随便删除三列虚拟变量的其中一列,这里删除的第一列,下标为0

model = LinearRegression()
model.fit(x, y)
print(model.predict([[1, 0, 2800]]))  # 预测town=robinsville area=2800的price