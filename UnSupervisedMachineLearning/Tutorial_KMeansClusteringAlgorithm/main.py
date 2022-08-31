import os
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

os.environ["OMP_NUM_THREADS"] = '1'

# 注意:\U会被转译成其他东西,所以表示地址的时候用\\
df = pd.read_csv("E:\Projects\PythonProjects\MachineLearning\\UnSupervisedMachineLearning\Tutorial_KMeansClusteringAlgorithm\DataSet\income.csv")
plt.scatter(df['Age'], df['Income($)'])
plt.show()

""" 使用KMeans库进行聚类操作 """
km = KMeans(n_clusters=3)  # 这里用作实例,直接设K=3
y_predicted = km.fit_predict(df[['Age', 'Income($)']])
df['cluter'] = y_predicted

""" 绘制聚类后的散点图 """
df1 = df[df.cluter==0]
df2 = df[df.cluter==1]
df3 = df[df.cluter==2]

plt.scatter(df1['Age'], df1['Income($)'], color='green')
plt.scatter(df2['Age'], df2['Income($)'], color='red')
plt.scatter(df3['Age'], df3['Income($)'], color='black')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker="*")  # 画出质心的位置

plt.xlabel("Age")
plt.ylabel('Income($)')
plt.show()

# 我们可以发现聚类的情况并不是很好,这是因为我们的数据数值太大了income是6位数,age是2位数.所以在聚类之前,我们要对数据进行预处理,让他们的数值在0-1之间


""" 使用MinMaxScaler对数据进行预处理 """
scaler = MinMaxScaler()
scaler.fit(df[['Income($)']])  # 参数是个二维数组[[]]
df['Income($)'] = scaler.transform(df[['Income($)']])
scaler.fit(df[['Age']])
df.Age = scaler.transform(df[['Age']])

km = KMeans(n_clusters=3)  # 这里用作实例,直接设K=3
y_predicted = km.fit_predict(df[['Age', 'Income($)']])
df['cluter'] = y_predicted

df1 = df[df.cluter==0]
df2 = df[df.cluter==1]
df3 = df[df.cluter==2]

plt.scatter(df1['Age'], df1['Income($)'], color='green')
plt.scatter(df2['Age'], df2['Income($)'], color='red')
plt.scatter(df3['Age'], df3['Income($)'], color='black')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker="*")  # 画出质心的位置

plt.xlabel("Age")
plt.ylabel('Income($)')
plt.show()

""" 使用ElbowTechnique找到最佳的K """
k_rng = range(1, 10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)  # 这里用作实例,直接设K=3
    km.fit_predict(df[['Age', 'Income($)']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(k_rng, sse)
plt.show()