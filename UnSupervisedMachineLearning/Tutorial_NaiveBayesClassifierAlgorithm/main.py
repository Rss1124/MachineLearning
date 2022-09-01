import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("E:\Projects\PythonProjects\MachineLearning\\UnSupervisedMachineLearning\Tutorial_NaiveBayesClassifierAlgorithm\DataSet\\titanic.csv")
df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)
target = df.Survived
inputs = df.drop('Survived', axis='columns')
dummies = pd.get_dummies(inputs.Sex)  # 处理文字变量
inputs = pd.concat([inputs, dummies], axis='columns')
inputs.Age = inputs.Age.fillna(inputs.Age.mean())  # 处理错误数据
inputs.Fare = inputs.Fare.fillna(inputs.Fare.mean())  # 处理错误数据
inputs = inputs.drop('Sex', axis='columns')
x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2)
model = GaussianNB()
model.fit(x_train, y_train)
print("准确度: " + str(model.score(x_test, y_test)))
print(y_test[: 10])
print(model.predict(x_test[: 10]))