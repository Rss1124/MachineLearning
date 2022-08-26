"""
K-Fold 交叉验证
交叉验证的目的： 在实际训练中，模型通常对训练数据好，但是对训练数据之外的数据拟合程度差。用于评价模型的泛化能力，从而进行模型选择。
交叉验证的基本思想： 把在某种意义下将原始数据(dataset)进行分组,一部分做为训练集(train set),另一部分做为验证集(validation set or test set),首先用训练集对模型进行训练,再利用验证集来测试模型的泛化误差。另外，现实中数据总是有限的，为了对数据形成重用，从而提出k-折叠交叉验证。
在机器学习建模过程中，通行的做法通常是将数据分为训练集和测试集。测试集是与训练独立的数据，完全不参与训练，用于最终模型的评估。
在训练过程中，经常会出现过拟合的问题，就是模型可以很好的匹配训练数据，却不能很好在预测训练集外的数据。如果此时就使用测试数据来调整模型参数，就相当于在训练时已知部分测试数据的信息，会影响最终评估结果的准确性。通常的做法是在训练数据再中分出一部分做为验证(Validation)数据，用来评估模型的训练效果。
验证数据取自训练数据，但不参与训练，这样可以相对客观的评估模型对于训练集之外数据的匹配程度。
模型在验证数据中的评估常用的是交叉验证，又称循环验证。它将原始数据分成K组(K-Fold)，将每个子集数据分别做一次验证集，其余的K-1组子集数据作为训练集，这样会得到K个模型。
这K个模型分别在验证集中评估结果，最后的误差MSE(Mean Squared Error) 加和平均就得到交叉验证误差。
交叉验证有效利用了有限的数据，并且评估结果能够尽可能接近模型在测试集上的表现，可以做为模型优化的指标使用。
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

digits = load_digits()
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)

kf = KFold(n_splits=3)  # n_splits表示:要处理"n"次数据集,每次处理会将数据集分成"n"份,留一份作为test数据集,其他n-1份作为train数据集
skf = StratifiedKFold(n_splits=3)  # "StratifiedKFold"比"KFold"好在哪儿

# 将[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],拆分3次,每次拆分会分成3份
for train_index, test_index in kf.split([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    print(train_index, test_index)
print("\n")

""" 定义一个函数来训练模型 """
def get_score(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

""" 检测不同预测模型的交叉验证情况 """
scores_l = []
scores_svm = []
scores_rf = []
for train_index, test_index in kf.split(digits.data):
    x_train, x_test, y_train, y_test = digits.data[train_index], digits.data[test_index], digits.target[train_index], digits.target[test_index]
    scores_l.append(get_score(LogisticRegression(), x_train, x_test, y_train, y_test))
    scores_svm.append(get_score(SVC(), x_train, x_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(), x_train, x_test, y_train, y_test))
print(scores_l)
print(scores_svm)
print(scores_rf)

""" 更高效更简单的方法使用cross_val_score """
# cross_val_score 省略了许多步骤
print(cross_val_score(LogisticRegression(), digits.data, digits.target, cv=4))  # cv表示:n_splits=4
