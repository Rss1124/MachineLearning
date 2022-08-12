# 多重分类: 识别手写数字
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits  # 从sklearn库导入数据
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn

""" 了解数据集 """
digits = load_digits()  # 数据集

print("可以看到每个数据包含什么")
print(dir(digits))  # (data)一维数组,用来表示8*8的像素格,(images)手写数字,(target)每个图像的标签
print("\n")

print("查看第一个数据digits[0]的data")
print(digits.data[0])

plt.gray()
plt.matshow(digits.images[0])  # 将一维数组显示为矩阵
plt.title("first digits[0]'s image")
plt.show()
print("\n")

print("查看第一个数据digits[0]的target(标签)")
print(digits.target[0])
print("\n")

""" 处理数据集 """
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

""" 训练模型 """
model = LogisticRegression()
model.fit(x_train, y_train)
print("模型准确率")
print(model.score(x_test, y_test))
print("\n")

""" 测试模型 """
plt.matshow(digits.images[67])
plt.title("digits.images[67]")
plt.show()
print("第67个数据的target")
print(digits.target[67])
print("\n")

print("识别第67个图像是数字几")
print(model.predict([digits.data[67]]))

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

# Q:如何分析混淆矩阵?
# A:行数代表: "y_test"代表的target数字
#   列数代表: "通过"x_test"识别得到的预测数字
#   矩阵里每个单元格的元素代表: 出现的次数
#   所以主对角线上的元素代表: 预测成功的次数
#   其他的非0元素[7][3]代表: 模型识别出来的数是3,实际上这个数是7,这个错误出现了2次