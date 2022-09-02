import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("E:\Projects\PythonProjects\MachineLearning\\UnSupervisedMachineLearning\Tutorial_NaiveBayesClassifierAlgorithm\DataSet\spam.csv")
print(df.groupby('label').describe())
df.email = df.email.fillna("invalid")  # 处理无效数据
df.label = df.label.fillna(0)
x_train, x_test, y_train, y_test = train_test_split(df.email, df.label, test_size=0.2)
v = CountVectorizer()
x_train_count = v.fit_transform(x_train.values)  # 处理文字数据

model = MultinomialNB()
model.fit(x_train_count, y_train)
x_test_count = v.transform(x_test)
print(model.score(x_test_count, y_test))
