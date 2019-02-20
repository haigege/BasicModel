import numpy as np
from numpy.core.function_base import linspace
from scipy.constants.constants import alpha

def main():
    import matplotlib.pyplot as plt
    
    ##scatter 散点
    fig = plt.figure()
    ax = fig.add_subplot(3, 3, 1)
    n = 128
    X = np.random.normal(0, 1, n) #随机数
    Y = np.random.normal(0, 1, n)
    T = np.arctan2(Y, X) #上色，Y除以X
    #plt.axes([0.025, 0.025, 0.95, 0.95]) #指定显示范围
    ax.scatter(X, Y, s = 75, c=T, alpha = .5) #散点，s为size，c为color
    plt.xlim(-1.5, 1.5), plt.xticks([]) #
    plt.ylim(-1.5, 1.5), plt.yticks([]) #
    plt.axis()
    plt.title('scatter')
    plt.xlabel('x')
    plt.ylabel('y')
    
    ##bar 柱状图
    ax = fig.add_subplot(332)
    n = 10
    X = np.arange(n)
    Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n) #随机数
    Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
    
    ax.bar(X, +Y1, facecolor = '#9999ff', edgecolor = 'white') #画出来，+Y1表示画在上面
    ax.bar(X, -Y2, facecolor = '#ff9999', edgecolor = 'white') #画出来，-Y2表示画在下面
    for x, y in zip(X, Y1):
        plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha = 'center', va = 'bottom') #Y1进行注释，x+0.4表示延x轴正方向0.4，y+0.05表示延y轴正方向移0.05，va='bottom'表示图在文字下面
    for x, y in zip(X, Y2):
        plt.text(x + 0.4, -y - 0.05, '%.2f' % y, ha = 'center', va = 'top')
    
    ##pie 饼图
    fig.add_subplot(333)
    n = 20
    Z = np.ones(n) #全域的变量，为1
    Z[-1] *= 2 #最后一个变量设为2
    #Z表示每块儿的值，explode = Z * .05表示每个扇形离中心的距离，labels为显示的值，colors显示的颜色（以灰度形式）
    plt.pie(Z, explode = Z * .05, labels = ['%.2f' % (i / float(n)) for i in range(n)], colors = ['%f' % (i / float(n)) for i in range(n)])
    plt.gca().set_aspect('equal') #设为正的圆形
    plt.xticks([]), plt.yticks([])
    
    
    ##polar 极坐标
#     fig.add_subplot(334)
    fig.add_subplot(334, polar = True) #极坐标形式显示
    n = 20
    theta = np.arange(0.0, 2 * np.pi, 2 * np.pi / n)
    radii = 10 * np.random.rand(n) #半径
    #plt.plot(theta, radii)
    plt.polar(theta, radii)

    
    ##heatmap 热图
    from matplotlib import cm #colormap，上色
    fig.add_subplot(335)
    data = np.random.rand(3, 3)
    cmap = cm.Blues
    #interpolation = 'nearest':离最近的差值
    map = plt.imshow(data, cmap = cmap, aspect = 'auto', interpolation = 'nearest', vmin = 0, vmax = 1)
    
    ##3D
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(336, projection = '3d')
    ax.scatter(1, 1, 3, s = 100) #坐标（1，1，3），size为100
    
    ##hot map
    fig.add_subplot(313)
    def f(x, y):
        return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
    n = 256
    x = linspace(-3, 3, n)
    y = linspace(-3, 3, n)
    X, Y = np.meshgrid(x, y)
    plt.contourf(X, Y, f(X, Y), 8, alpha = .75, cmap = plt.cm.hot)
    #plt.savefig("./fig.png")
    
    plt.show()
    
if __name__ == '__main__':
    main()