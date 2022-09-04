import numpy as np
import pandas as pd
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['flower'] = iris.target
df['flower'] = df['flower'].apply(lambda x: iris.target_names[x])
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

""" 使用传统方法研究同一个模型不同参数对准确度的的影响(使用K-Fold 交叉验证) """
print("同一个模型不同参数对准确度的的影响:")
# print(cross_val_score(svm.SVC(kernel='linear', C=10, gamma='auto'), iris.data, iris.target, cv=5))
# print(cross_val_score(svm.SVC(kernel='rbf', C=10, gamma='auto'), iris.data, iris.target, cv=5))
# print(cross_val_score(svm.SVC(kernel='rbf', C=20, gamma='auto'), iris.data, iris.target, cv=5))
kernels = ['rbf', 'linear']
C = [1, 10, 20]
avg_scores = {}
for kval in kernels:
    for cval in C:
        cv_scores = cross_val_score(svm.SVC(kernel=kval, C=cval, gamma='auto'), iris.data, iris.target, cv=5)
        avg_scores[kval + '_' + str(cval)] = np.average(cv_scores)
print(avg_scores)
print("\n")

""" 使用GridSearchCV能更快得进行研究(实现的操作跟上面是一样的) """
print("GridSearchCV研究:")
clf = GridSearchCV(svm.SVC(gamma='auto'), {
    'C': [10, 20, 30],
    'kernel': ['rbf', 'linear']
}, cv=5, return_train_score=False)
clf.fit(iris.data, iris.target)
result = pd.DataFrame(clf.cv_results_)
print(result[['param_C', 'param_kernel', 'mean_test_score']])
print("最高的准确度:" + str(clf.best_score_))
print("最佳的参数:" + str(clf.best_params_))
print("\n")

""" 使用RandomizedSearchCV减少计算量 """
print("RandomizedSearchCV减少计算量:")
rs = RandomizedSearchCV(svm.SVC(gamma='auto'), {
    'C': [1, 10, 20],
    'kernel': ['rbf', 'linear']
}, cv=5, return_train_score=False, n_iter=2)  # iter=2表示,只会循环两次
rs.fit(iris.data, iris.target)
rs_result = pd.DataFrame(rs.cv_results_)[['param_C', 'param_kernel', 'mean_test_score']]
print(rs_result)
