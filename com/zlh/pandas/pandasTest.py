import numpy as np
import pandas as pd
from pylab import *

def main():
    ##1--Data Structure
    s = pd.Series([i * 2 for i in range(1, 11)])
    print(type(s))
    dates = pd.date_range("20190101", periods = 8)
    print(dates)
    df = pd.DataFrame(np.random.randn(8, 5), index = dates, columns = list('ABCDE'))
    print(df)

#     df = pd.DataFrame({'A':1, 'B':pd.Timestamp('20170301'), 'C':pd.Series(1, index = list(range(4)),dtype = 'float32'),
#                     'D':np.array([3] * 4, dtype = 'float32'), 'E':pd.Categorical(['police', 'student', 'teacher', 'doctor'])})
#     print(df)
    
    #2--Basic
    print(df.head(3)) #头三行
    print(df.tail(3)) #尾三行
    print(df.index) #index
    print(df.values)    
    print(df.T) #转置
#     print(df.sort(columns = 'C')) #C列排序
    print(df.sort_values(by = ['C']))#C列排序
    print(df.sort_index(axis = 1, ascending = False)) #按索引降序
    print(df.describe()) #描述
    
    #Select
    print(df['A']) #切片，A列
    print(type(df['A']))
    print(df[:3]) #0，1，2行
    print(df['20190101':'20190104']) #20190101 至 20190104行的数据
    print(df.loc[dates[0]]) #第0列
    print(df.loc['20190101':'20190104',['B','D']]) #20190101 至 20190104行，B和D列，的数据
    print(df.at[dates[0],'C']) #第0行，‘C’列的数据
    print(df.iloc[1:3, 2:4]) #第1行3列  至 2行4列 的数据
    print(df.iloc[1, 4]) #第1行4列的数据
    print(df.iat[1,4]) #同上
    print(df[df.B > 0][df.A < 0]) #类似sql，满足在B列上大于0，A列上小于0的数
    print(df[df>0]) #只显示大于0的值
    print(df[df['E'].isin([1, 2])]) #类似sql的in
    
    #Set,以下为赋值
    s1 = pd.Series(list(range(10, 18)), index =pd.date_range('20190101', periods=8) )
    df['F'] = s1
    print(df)
    df.at[dates[0], 'A'] = 0 
    df.iat[1,1] = 1
    df.loc[:, 'D'] = np.array([4] * len(df))
    df2 = df.copy()
    df2[df2 > 0] = -df2 #所有正的都变成负数
    print(df2)
    
    ##3--Missing Values
    df1 = df.reindex(index = dates[:4], columns = list('ABCD')+['G']) #取df的0到3行，A到D列，并添加G列，目前G列缺少值
    print(df1)
    df1.loc[dates[0]:dates[1],'G'] = 1 #G列的第0到1个赋为1，其他的仍缺少值
    print(df1)
    print(df1.dropna()) #有缺少值的，整行丢弃
    print(df1.fillna(value = 2)) #缺省值设为2
    
    ##4.1--Statistic
    print(df)
    print(df.mean()) #求每列均值
    print(df.var()) #求系列的方差
    s = pd.Series([1,2,4,np.nan,5,7,9,10], index = dates)
    print(s)
    print(s.shift(2)) #移两位
    print(s.diff()) #不填数字，表示一阶，填写数字表示多阶。数据进行某种移动之后与原数据进行比较得出的差异数据，其实先shift() - df
    print(s.value_counts()) #每个值出现的次数
    print(df.apply(np.cumsum)) #对于每一列，下面的值是上面值的累加和
    print(df.apply(lambda x:x.max() - x.min())) #最大值减最小值，即极差
    
    ##4.2--Concat
    pieces = [df[:3], df[-3:]] #前三行，后三行
    print(pd.concat(pieces)) #拼接
    left = pd.DataFrame({'key':['x', 'y'], 'value':[1, 2]})
    right = pd.DataFrame({'key':['x', 'z'], 'value':[3, 4]})
    print('LEFT:',left)
    print('RIGHT:',right)
    print(pd.merge(left,right,on='key',how='left')) #类似sql的left jon
    df3 = pd.DataFrame({'A':['a','b','c','b'],'B':list(range(4))})
    print(df3.groupby('A').sum()) #类似sql的group by
    
    
    ##4.3--Reshape 交叉分析 或 透视
    import datetime
    # 24行
    df4 = pd.DataFrame({'A':['one','one','two','three'] * 6,
                        'B':['a','b','c'] * 8,
                        'C':['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 4,
                        'D':np.random.randn(24),
                        'E':np.random.randn(24),
                        'F':[datetime.datetime(2017, i, 1) for i in range(1, 13)] +
                            [datetime.datetime(2017, i, 15) for i in range(1, 13)]})
    print(df4)
    print('------------')
    #TODO:
    print(pd.pivot_table(df4, values='D',index=['A','B'],columns=['C'])) #values:输出值，index:输出项，columns：列值
    
    ##5.1 Time Series
    t_exam = pd.date_range('20190101', periods=10, freq='S') #S表示秒
    
    ##5.2 Graph
    ts = pd.Series(np.random.randn(1000), index = pd.date_range('20190101', periods=1000))
    ts = ts.cumsum() #累加
#     from pylab import *
    ts.plot()
    show()
    
    ##5.2 File
    df6 = pd.read_csv('./data/test.csv')
    print(df6)
    df7 = pd.read_excel('./data/test.xlsx', 'Sheet1') 
    print('Excel', df7)
    df6.to_csv('./data/test2.csv')
    df7.to_excel('./data/test2.xlsx')
if __name__ == '__main__':
    main()