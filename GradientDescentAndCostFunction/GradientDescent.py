# 根据x, y,通过梯度下降找到他们之间的关系式:y = mx + b
import numpy as np

""" 需要掌握的知识点 """
# 1.Q：如何知道关系式是否符合所给数据？
#   A：损失函数cost,依次对每个测试数据和预测数据做差然后平方,最后除以n,越接近0,说明关系式越符合.表达式:Σ{[测试数据(y)-预估数据(y_predicted)]^2}/n
# 2.Q：如何确定m, b？
#   A：根据梯度下降规则,假设m,b的初值为0,然后根据损失函数式:Σ{[y-(mx + b)]^2}/n,分别求出m,b的偏导md,bd表达式,再将x,y代入算出具体md,bd的值
#      接着设置一个learning_rate,用于计算下一个m,b将要移动的步长,learning_rate越小表示m,b的变化量越小,找到满足的关系式的概率越大
""" 分割线 """

def gradient_descent(x, y):
    n = len(x)  # n表示数据个数
    m_curr = b_curr = 0
    iterations = 10000  # iterations表示迭代次数,迭代次数越大,找到满足的关系式的概率越大
    learning_rate = 0.001

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y - y_predicted)])
        md = -(2/n)*sum(x * (y - y_predicted))  # m的偏导表达式
        bd = -(2/n)*sum(y - y_predicted)  # b的偏导表达式
        m_curr = m_curr - learning_rate * md  # learning_rate * md: m要改变的值
        b_curr = b_curr - learning_rate * bd  # learning_rate * bd: b要改变的值
        print("m: {}, b: {}, cost: {}, iteration: {}".format(m_curr, b_curr, cost,  i))

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

gradient_descent(x, y)