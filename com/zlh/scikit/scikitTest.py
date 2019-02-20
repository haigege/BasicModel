import numpy as np
import pandas as pd

def main():
    #通过花萼长度、花萼宽度、花瓣长度、花瓣宽度四个属性值，判断鸢尾属于哪种（Iris Setosa （ 山 鸢 尾 ） 、Iris Versicolour （ 杂 色 鸢 尾 ）、Iris  Virginica （ 维 吉 尼 亚 鸢 尾 ））
    
    #Pre-processing 数据预处理
    from sklearn.datasets import load_iris 
    iris = load_iris() #鸢尾花样例。 data属性：花萼长度、花萼宽度、花瓣长度、花瓣宽度四个属性值； target:标注，0、1、2三类
    print(iris)
    print(len(iris['data']))
    
    #from sklearn.cross_validation import train_test_split
    from sklearn.model_selection import train_test_split
    #test_size = 0.2:验证数据占总体的20%，random_state = 1:随要的选择数据
    train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target, test_size = 0.2, random_state = 1)
    
    #Model 建模
    from sklearn import tree #引入决策树
    clf = tree.DecisionTreeClassifier(criterion='entropy') #选择分类器非回归器，criterion='enropy'表示熵增益
    clf.fit(train_data, train_target) #训练集进行训练,通过train_data和 train_target建立了决策树的关系
    y_pred = clf.predict(test_data) #用test_data（验证集）进行预测
    
    #Verify 通过准确率和混淆矩阵两种方式验证
    from sklearn import metrics
    print(metrics.accuracy_score(y_true = test_target, y_pred = y_pred)) #准确率验证：y_true = test_target验证target,为真实值；y_pred = y_pred为预测值
    print(metrics.confusion_matrix(y_true=test_target, y_pred=y_pred)) #混淆矩阵验证
    
    #输出
    with open('./tree.dot', 'w') as fw:
        tree.export_graphviz(clf, out_file=fw)
    
if __name__ == '__main__':
    main()