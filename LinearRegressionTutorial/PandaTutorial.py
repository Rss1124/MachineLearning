import pandas as pd

"""
将数据集导入pandas
"""
df = pd.read_csv('E:\Projects\PythonProjects\MachineLearning\LinearRegressionTutorial\DataSet\pokemon_data.csv')
print(df)

"""
在panda中获取指定数据
"""

# 读取每一列数据
print(df[['Name', 'Type 1', 'Type 2']][0:5])

# 读取每一行数据
# print(df.iloc[0:4])
for index, row in df.iterrows():
    print(index, row)

# 读取矩阵里的特定元素
print("第二行第一列的元素是:\n" + df.iloc[2, 1])
print("Type 1为Fire的元素如下:\n")
print(df.loc[df['Type 1'] == 'Fire'])

# 对数据进行汇总以及对数据进行排序
print("数据汇总如下:\n")
print(df.describe())
print("对Name进行降序排序后如下:\n")
print(df.sort_values('Name', ascending=False))
print("对Type 1和Hp进行联合升序排序后如下:\n")
print(df.sort_values(['Type 1', 'HP'], ascending=[1, 0]))

# 新增一列数据

# tip: 这种方法不是永久性的修改,因为这只是将数据存入内存中,再次运行时,仍然会使用原来的数据集
df['Total'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] + df['Sp. Def'] + df['Speed']
print("对数据集新增一列'Total'数据后如下:\n")
print(df.head(5))


# 删除一列数据
df = df.drop(columns=['Total'])
print("删除'Total'数据后如下:\n")
print(df.head(5))
