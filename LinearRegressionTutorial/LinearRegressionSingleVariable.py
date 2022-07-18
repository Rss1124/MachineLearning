import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('E:\Projects\PythonProjects\MachineLearning\LinearRegressionTutorial\DataSet\homeprice.csv')
plt.xlabel('area(sqr ft)')
plt.ylabel('price(RS$)')
plt.scatter(df.area, df.price, color='red', marker='+')
plt.show()

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)
print(reg.predict([[3300]]))