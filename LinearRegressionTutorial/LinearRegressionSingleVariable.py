import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('E:\Projects\PythonProjects\MachineLearning\LinearRegressionTutorial\DataSet\homeprice.csv')
""" 画出数据集对应的散点图 """
plt.xlabel('area(sqr ft)')
plt.ylabel('price(RS$)')
plt.scatter(df.area, df.price, color='red', marker='+')

# 获得一元的线性回归方程: y = kx + b
plt.show()

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

# 对线性回归模型进行数据预测:输入"房屋面积(3300)"得到一个"房屋价格"
print(reg.predict([[3300]]))
# 输出模型的"斜率k"
print(reg.coef_)
# 输出模型的"偏离值b"
print(reg.intercept_)

""" 对测试数据集进行预测 """
# 导入测试数据集
df_test = pd.read_csv("E:\Projects\PythonProjects\MachineLearning\LinearRegressionTutorial\DataSet\homeprice_test.csv")
# 将数据集放入线性回归模型进行预测
df_test_predict = reg.predict(df_test)
df_test['prices'] = df_test_predict
df_test.to_csv('homeprice_test_predictions.csv', index=False)

""" 绘制线性回归函数 """
plt.xlabel('area', fontsize=20)
plt.ylabel('price', fontsize=20)
plt.scatter(df.area, df.price, color='red', marker='+')
plt.plot(df.area, reg.predict(df[['area']]), color='blue')
plt.show()

""" 保存训练好的模型,并且直接使用它 """
# 此时会在该路径下生成一个二进制文件
with open('model_pickle', 'wb') as f:
    pickle.dump(reg, f)
# 接下来直接使用模型进行预测
with open('model_pickle', 'rb') as f:
    mp = pickle.load(f)

print(mp.predict([[5000]]))
