import pandas as pd
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

digits = load_digits()
print(dir(digits))
df = pd.DataFrame(digits.data, columns=digits.feature_names)
df['target'] = digits.target

x = df.drop(['target'], axis='columns')
y = df.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = SVC(kernel='linear', gamma=100)
model.fit(x_train, y_train)
print(model.score(x_test, y_test))