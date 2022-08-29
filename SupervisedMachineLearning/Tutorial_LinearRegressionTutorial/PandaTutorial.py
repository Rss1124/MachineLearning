import pandas as pd

"""
将数据集导入pandas
"""
df = pd.read_csv('E:\Projects\PythonProjects\MachineLearning\SupervisedMachineLearning\Tutorial_LinearRegressionTutorial\DataSet\pokemon_data.csv')
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
# df['Total'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] + df['Sp. Def'] + df['Speed']
# tip: 这种方法不是永久性的修改,因为这只是将数据存入内存中,如果不保存! 再次运行时,仍然会使用原来的数据集

# 另一种更实用的方法新增一列数据
df['Total'] = df.iloc[:, 4:10].sum(axis=1)
# tip: iloc的第二个参数4:10是要求和的范围,并且这个范围是左闭右开的[4,10)

print("对数据集新增一列'Total'数据后如下:\n")
print(df.head(5))

# 交换某列数据的位置
cols = list(df.columns.values)
df = df[cols[0:4] + [cols[-1]] + cols[4:12]]
# tip: 参数x:y是范围,并且这个范围是左闭右开的[4,10)
print("将最后一列数据'Total'交换到第5列后如下:\n")
print(df.head(5))

# 保存新的csv文件
print("新的csv文件已经保存到当前文件夹下,请自行前往查看,同时记得删除它,因为它只是一个演示样例\n")
df.to_csv('modified.csv', index=False)
# tip:有多种函数,以便存储多种文件,比如:to_excel

# 删除一列数据
df = df.drop(columns=['Total'])
print("删除'Total'数据后如下:\n")
print(df.head(5))

# 过滤数据
new_df = df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison')]
# tip: 可以试试把&换成|试试运行结果是什么
print("Type 1为Grass以及Type 2为Poison的元素如下:\n")
print(new_df.head(5))
print("将筛选后的数据另存为一个新的csv文件,记得删除它\n")
new_df.to_csv('filtered.csv')
df = df.loc[~df['Name'].str.contains('Mega')]
# tip: ~df['Name'].str.contains('Mega')中~表示否的意思,如果用!则会报错
# tip: 可以import re,里面的方法更多
print("将Name中带有mega关键字的数据过滤掉后的情况如下:\n")
print(df.head(5))

# 更改数据
df.loc[df['Type 1'] == 'Fire', 'Type 1'] = 'Flamer'
print("将'Type 1'为'Fire'的对象里的'Type 1'更改为'Flamer'\n")
print(df.head(8))
df.loc[df['Type 1'] == 'Flamer', 'Legendary'] = 'True'
print("将'Type 1'为'Flamer'的对象里的'Legendary'更改为'True'\n")
print(df.head(8))

# 数据统计
print("根据HP的平均值,从高到低进行排序,用Type 1进行显示\n")
print(df.groupby(['Type 1']).mean().sort_values('HP', ascending=False))
print("统计发现:Type1为Dragon的平均HP最高")

print("对同一个Type 1的对象的每一个数据进行求和\n")
print(df.groupby(['Type 1']).sum())

print("根据Type 1, Type 2来统计不同对象的个数\n")
df['count'] = 1
print(df.groupby(['Type 1', 'Type 2']).count()['count'])

# 如何快速处理大量数据
for df in pd.read_csv('E:\Projects\PythonProjects\MachineLearning\SupervisedMachineLearning\Tutorial_LinearRegressionTutorial\DataSet\pokemon_data.csv', chunksize=5):
    print("CHUNK DF")
    print(df)
# tips: 参数chunksize 5代表同时处理5行数据,这个数字大小取决于你的计算机能力如何,可以很大,也可以为1