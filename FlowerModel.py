from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import tensorflow as tf

"""
获取数据
"""

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
)
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
)

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)  # header=0 means row zero is the header
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
# print(train)

# train:等会要用到的训练集 train_y:用于训练的标签
train_y = train.pop('Species')
test_y = test.pop('Species')


"""
将数据进行整理,最后得到一个整理后的数据集
"""
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
# print(my_feature_columns)


"""
将数据集转换为dataset,用于后面拿来喂养模型
"""
def input_fn(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.cache('E:\Projects\PythonProjects\MachineLearning\cache\TempFile_Flower')

    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

"""
创建一个具有两层hiddenlayer的深度神经网络用来跑数据
"""
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # two hidden layers of 30 and 10 nodes respectively
    # 该神经网络有两层隐藏层,第一层有30个神经元,第二层有10个神经元
    hidden_units=[30, 10],
    n_classes=3
)

"""
开始训练
"""
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000
)

"""
测试模型的准确度
"""
eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

"""
对模型放入数据进行预测
"""
# 预测的时候,不会将标签数据(label)传入,因为这是模型要给出的答案
def test_input_fn(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted")
for feature in features:
    valid = True
    while valid:
        val = input(feature + ": ")
        if not val.isdigit():
            valid = False
        predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: test_input_fn(predict))

for pred_dict in predictions:
    print(pred_dict)
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    # print(class_id)

    print('Prediction is "{}" ({}%)'.format(SPECIES[class_id], 100 * probability))