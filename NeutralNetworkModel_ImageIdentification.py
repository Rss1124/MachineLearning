import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

"""
获取数据集DataSet
train_images:60000个28*28的矩阵,每个矩阵表示一个图像
train_label:60000个标签对应60000个图像
test_images:10000个28*28的矩阵,用来测试模型的准确度
test_label:10000个标签对应10000个图像
"""
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_label), (test_images, test_label) = fashion_mnist.load_data()
# 用plt绘画出第二个图像
# plt.figure()
# plt.imshow(train_images[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()


"""
数据预处理preprocessing
因为在28*28的矩阵(灰度图像)里面,每一个单元格都是由一个数值表示(0~255),数值越接近0表示这个单元格的颜色越接近黑色,数值越接近255表示这个单元格的颜色越接近白色
所以预处理就是将0~255的数值压缩成0~1
预处理后,数值越接近0表示这个单元格的颜色越接近黑色,数值越接近1表示这个单元格的颜色越接近白色
"""
train_images = train_images / 255.0
test_images = test_images / 255.0


"""
通过调用kerasAPI来创建一个神经元网络模型Model
"""
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer
    keras.layers.Dense(128, activation='relu'),  # hidden layer
    keras.layers.Dense(10, activation='softmax')  # output layer, softmax用来计算每个类的概率分布
])


"""
编译模型的方法
会定义损失函数、优化器和我们想要的得到的指标: loss function, optimizer, metrics
"""
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


"""
训练模型:Training the Model
"""
model.fit(train_images, train_label, epochs=10)


"""
测试模型:Evaluating the Model
"""
test_loss, test_acc = model.evaluate(test_images, test_label, verbose=1)
print('Test accuracy: ', test_acc)


"""
对数据进行预测:Making Predictions
"""
# predictions = model.predict(test_images)
# predictions[0]  # predictions[0]->9
# test_label[0]  # test_label[0]=9


"""
Verifying Predictions: 证明预测结果
"""
COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR


def predict(model, image, correct_label):
    class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  # 将标签用中文记录下来
    prediction = model.predict(np.array([image]))
    prediction_class = class_name[np.argmax(prediction)]
    show_image(image, class_name[correct_label], prediction_class)


def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    print("Expected: " + str(label))
    print("Guess: " + str(guess))
    plt.show()


def get_number():
    while True:
        num = input("Pick a number: ")
        if num.isdigit():
            num = int(num)
            if 0 <= num <= 1000:
                return int(num)
        else:
            print("try again")


num = get_number()
image = test_images[num]
label = test_label[num]
predict(model, image, label)