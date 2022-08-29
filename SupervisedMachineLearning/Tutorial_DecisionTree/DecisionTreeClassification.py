# 使用pandas和numpy,通过面向对象编程,纯代码实现iris花朵的决策树分类
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
data = pd.read_csv("E:\Projects\PythonProjects\MachineLearning\SupervisedMachineLearning\Tutorial_DecisionTree\DataSet\iris.csv")

""" 使用class定义决策树的节点("决策节点"和"叶子节点") """
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        # 决策节点
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # 叶子节点
        self.value = value


""" 使用class定义决策树 """
class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=2):
        # 初始化树的根节点
        self.root = None

        # 决策条件
        self.min_samples_split = min_samples_split  # min_samples_split=2 表示"最少"要有"两个不同类别"的数据在同一个数据集里才会进行split操作
        self.max_depth = max_depth  # 表示"split后"树的深度

    def build_tree(self, dataset, curr_depth=0):
        # 使用递归函数来构建决策树
        x, y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(x)  # num_samples
        # 用来表示当前数据集里面的"不同类别(type)"数据的个数,比如本例中iris花朵有三种类型,如果当前数据中"只有一种类型"的数据,那么说明这个数据集非常"有序",就不用进行split了
        # num_features用来表示花朵的4个属性('sepal_length', 'sepal_width', 'petal_length', 'petal_width')

        # 如果当前数据集出现了两种及两种以上的数据,就要进行split操作
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # 找最佳的决策条件
            best_split = self.get_best_split(dataset, num_features)
            if best_split["info_gain"] > 0:
                # 向左孩子出发
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                # 向右孩子出发
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                # 递归结束,返回
                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree,
                            best_split["info_gain"])

        leaf_value = self.calculate_leaf_value(y)
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_features):
        # 将最佳决策条件保留下来
        best_split = {}
        max_info_gain = -float("inf")

        # 检查所有的feature
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)  # unique函数去除其中重复的元素，并按元素由大到小返回一个新的无元素重复的元组或者列表
            # 检查每个数据里面的出现的feature values
            for threshold in possible_thresholds:
                # 获取当前的决策条件(split)
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # 检查子树是否为空
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # 计算information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # 更新最好的决策条件
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        return best_split

    @staticmethod
    def split(dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode == "gini":
            gain = self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain

    @staticmethod
    def entropy(y):
        # 计算熵值
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    @staticmethod
    def gini_index(y):
        # 计算基尼系数
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)  # 基尼系数的计算方法
            gini += p_cls ** 2
        return 1 - gini

    @staticmethod
    def calculate_leaf_value(y):
        # 计算叶子节点的值
        y = list(y)
        return max(y, key=y.count)

    def print_tree(self, tree=None, indent=""):
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("x_" + str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % indent, end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % indent, end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, x, y):
        # 训练决策树
        dataset = np.concatenate((x, y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, x):
        predictions = [self.make_prediction(x, self.root) for x in x]
        return predictions

    def make_prediction(self, x, tree):
        if tree.value != None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=41)
classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
classifier.fit(x_train, y_train)
classifier.print_tree()