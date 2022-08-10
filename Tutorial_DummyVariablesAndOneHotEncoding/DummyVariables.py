# Topic: 虚拟变量与独热编码
# Description:
# Q: 在制作线性回归模型时遇到了文字型变量(Nominal)该怎么办? A: 使用pandas的虚拟变量或者sklearn库的独热编码
# Q: pandas的虚拟变量是什么? A: pandas会根据原生文字型变量的种类数(n),在原有的数据集上,新增n列数据,用来对应各个文字型变量,用0,1进行赋值
# Q: sklearn库的独热编码是什么? A: sklearn会将原生文字型变量重新以数字赋值(0, 1, 2 ... n)

from sklearn.linear_model import LinearRegression
import pandas as pd
df = pd.read_csv("E:\Projects\PythonProjects\MachineLearning\Tutorial_DummyVariablesAndOneHotEncoding\DataSet\homeprice_nominal_data.csv")
dummies = pd.get_dummies(df.town)  # 获取虚拟变量
merged = pd.concat([df, dummies], axis='columns')  # 将虚拟变量与原生数据集合并

# Q: 为什么要将"town"和"west windsor"两列数据删掉
# A: "town"数据是文字型数据,本来就要将其删掉,对于"west windsor","monroe township","robinsville"这三个虚拟变量则是要随便选取一个将其删除(对最终的预测也不会有影响)
#    因为多一个变量就意味着会多大量的计算,最终可能会影响模型的准确度
final = merged.drop(['town', 'west windsor'], axis='columns')  # 获取最后的数据集

model = LinearRegression()
x = final.drop('price', axis='columns')  # 获取自变量
y = final.price
model.fit(x, y)
print("虚拟变量: ")
print(model.predict([[2800, 0, 0]]))  # 预测area=2800,town=west windsor的price