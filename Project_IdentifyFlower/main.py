# 识别花的类型
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn

""" 了解数据集(flower) """
flower = load_iris()
print(dir(flower))
print("查看第一朵花的4个属性")
print(flower.data[0])  # data数据里面有4个属性,他们分别代表: sepal length(萼片长度), sepal width(萼片宽度), petal length(花瓣长度), petal width(花瓣宽度),这4个属性用来对花朵进行预测
print("查看第一朵花的名字")
print(flower.target_names[0])

""" 处理数据集 """
x_train, x_test, y_train, y_test = train_test_split(flower.data, flower.target, test_size=0.2)

""" 训练模型 """
model = LogisticRegression()
model.fit(x_train, y_train)
print("模型准确率")
print(model.score(x_test, y_test))

""" 测试模型 """
print("第67朵花的target")
print(flower.target[67])
print("识别第67朵花的target")
print(model.predict([flower.data[67]]))

""" 打印出模型的混淆矩阵ConfusionMatrix """
# Q:什么是混淆矩阵
# A:混淆矩阵是对模型训练后的总结,用一个矩阵来表现
y_predicted = model.predict(x_test)
cm = confusion_matrix(y_test, y_predicted)
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel("PredictedData")
plt.ylabel("RealData")
plt.title("ConfusionMatrix")
plt.show()