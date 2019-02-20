from keras.models import Sequential #神经网络各层容器
from keras.layers import Dense, Activation #Dense:加权求和层，Activation：激函数
from keras.optimizers import SGD #随机梯度下降算法

def main():
#     pass
    #运行一下，用报错：ModuleNotFoundError: No module named 'tensorflow'
    #此时，在C盘用户->自己的账号->.keras->keras.json文件
    #先备份，然后打开文件keras.json，里面的"backend": "tensorflow"
    from sklearn.datasets import load_iris
    iris = load_iris()
    print(iris["target"])
    from sklearn.preprocessing import LabelBinarizer 
    print(LabelBinarizer().fit_transform(iris['target'])) #标签化
    from sklearn.model_selection import train_test_split
    train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target, test_size = 0.2, random_state = 1)
    labels_train = LabelBinarizer().fit_transform(train_target)
    labels_test = LabelBinarizer().fit_transform(test_target)
 
    #构建神经网络层
    #填入网络结构
    model = Sequential(
        [
            Dense(5,input_dim=4),
            Activation('relu'),
            Dense(3),
            Activation('sigmoid'),
            ])
    #也可用下面的方式：
    #model = Sequential()
    #model.add(Dense(5), input = 4)
     
    sgd = SGD(lr=0.01, decay=1e-6,momentum=0.9,nesterov=True)  #定义随机梯度下降的优化器
    model.compile(optimizer=sgd,loss='categorical_crossentropy')
    model.fit(train_data,labels_train,nb_epoch=200,batch_size=40)
    print(model.predict_classes(test_data))
     
    model.save_weights('./data/w')
    model.load_weights('./data/w')
    
if __name__ == '__main__':
    main()