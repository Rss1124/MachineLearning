# 使用support vector machine对鸢尾植物进行预测
import pandas as pd
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris = load_iris()
print(dir(iris))  # 查看iris数据集里面有什么类型的数据
df = pd.DataFrame(iris.data, columns=iris.feature_names)  # 提取iris里面的feature_names拿来用作实验数据集,并且将其转化为pd数据(二维数组)
df['target'] = iris.target  # 将target也录入实验数据集

""" 介绍几个处理df数据的方法 """
print("筛选出target为'0'的数据")
print(df[df.target==0].head())
print("根据iris里面的target_names给df数据添加新的一列'flower_names'")
df['flower_names'] = df.target.apply(lambda x: iris.target_names[x])  # 因为"target"与"target_names的下标"是对应的,target_names[0]=setosa <=> target=0
print(df.head())

""" 绘制feature_names与target的关系图 """
# 先将不同target的数据分开
iris0 = df[df.target==0]
iris1 = df[df.target==1]
iris2 = df[df.target==2]
# 绘制iris0("SepalLength"与"SepalWidth")和iris1("SepalLength"与"SepalWidth"))的关系图
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(iris0['sepal length (cm)'], iris0['sepal width (cm)'], color='green', marker='+')
plt.scatter(iris1['sepal length (cm)'], iris1['sepal width (cm)'], color='blue', marker='.')
plt.show()

""" 训练SVM模型 """
x = df.drop(['target', 'flower_names'], axis='columns')
y = df.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# 默认SVC里面的kernel参数是rbf,可以改成linear,会造成正确率变更
model = SVC(kernel='linear')
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
