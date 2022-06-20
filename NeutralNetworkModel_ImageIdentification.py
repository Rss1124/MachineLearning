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
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

"""
数据预处理preprocessing
因为在28*28的矩阵(灰度图像)里面,每一个单元格都是由一个数值表示(0~255),数值越接近0表示这个单元格的颜色越接近黑色,数值越接近255表示这个单元格的颜色越接近白色
所以预处理就是将0~255的数值压缩成0~1
预处理后,数值越接近0表示这个单元格的颜色越接近黑色,数值越接近1表示这个单元格的颜色越接近白色
"""
train_images = train_images / 255.0
test_images = test_images / 255.0


"""
设置层,通过调用kerasAPI来创建一个神经元网络模型Model
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
test_loss, test_acc = model.evaluate(test_images, test_label, verbose=2)
print('Test accuracy: ', test_acc)


"""
进行预测:
编写预测函数
在模型经过训练后,您可以使用它对一些图像进行预测.模型具有线性输出,即"logits"您可以附加一个softmax层,将"logits"转换成更容易理解的概率.
"""
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
# print(np.argmax(predictions[0]))
# print(test_label[0])

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]), color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

"""
Verifying Predictions: 证明预测结果
"""

# i = 0
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions[i], test_label, test_images)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions[i],  test_label)
# plt.show()
#
# i = 12
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions[i], test_label, test_images)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions[i],  test_label)
# plt.show()

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_label, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_label)
plt.tight_layout
plt.show()


"""
最后，使用训练好的模型对单个图像进行预测
"""
img = test_images[1]
#tf.keras 模型经过了优化，可同时对一个批或一组样本进行预测。因此，即便您只使用一个图像，您也需要将其添加到列表中：
img = (np.expand_dims(img, 0))
predictions_single = probability_model.predict(img)
print(predictions_single)
plot_value_array(1, predictions_single[0], test_label)
_ = plt.xticks(range(10), class_names, rotation=45)
print(np.argmax(predictions_single[0]))