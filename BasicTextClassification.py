import tensorflow as tf
from tensorflow import keras
import numpy as np

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

"""
评论文本被转换为整数值,其中每个整数代表词典中的一个单词
这里我们将创建一个辅助函数来查询一个包含了整数到字符串映射的字典对象
"""
# 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()

#保留第一个索引
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# 显示首条评论的文本
print(decode_review(train_data[0]))

"""
影评——即整数数组必须在输入神经网络之前转换为张量
我们可以填充数组来保证输入数据具有相同的长度,然后创建一个大小为max_length * num_reviews 的整型张量.我们可以使用能够处理此形状数据的嵌入层作为网络中的第一层.
由于电影评论长度必须相同,我们将使用 pad_sequences 函数来使长度标准化:
"""
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

"""
构建模型:
神经网络由堆叠的层来构建,这需要从两个主要方面来进行体系结构决策:
a.模型里有多少层?
b.每个层里有多少隐层单元(hidden units)?
"""
# 输入形状是用于电影评论的词汇数目(10,000 词)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

#层按顺序堆叠以构建分类器：
#第一层是嵌入(Embedding)层.该层采用整数编码的词汇表,并查找每个词索引的嵌入向量(embedding vector).这些向量是通过模型训练学习到的.向量向输出数组增加了一个维度.得到的维度为:(batch, sequence, embedding).
#接下来,GlobalAveragePooling1D 将通过对序列维度求平均值来为每个样本返回一个定长输出向量.这允许模型以尽可能最简单的方式处理变长输入.
#该定长输出向量通过一个有 16 个隐层单元的全连接(Dense)层传输.
#最后一层与单个输出结点密集连接.使用 Sigmoid 激活函数,其函数值为介于 0 与 1 之间的浮点数,表示概率或置信度.
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])