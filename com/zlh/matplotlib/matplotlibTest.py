import numpy as np

def main():
    import matplotlib.pyplot as plt
    x = np.linspace(-np.pi, np.pi, 1024, endpoint=True) #横轴：-兀 到 兀之间，有256个点，包括最后一个点 
    c, s = np.cos(x), np.sin(x) #定义余弦、正弦两个函数
    plt.figure(1) #第一个图
    plt.plot(x, c, color = 'blue', linewidth = 1.0, linestyle = '-', label = 'COS', alpha = 0.5) #绘制,颜色、线宽、线型、label、透明度
    plt.plot(x, s, 'r*', label = 'sin') #绘制 r:红色，线型：*
    plt.title("COS & SIN")
    
    ax = plt.gca() #轴编辑器
    ax.spines['right'].set_color('none') #隐藏右边的线
    ax.spines['top'].set_color('none') #隐藏上边的线
    ax.spines['left'].set_position(('data', 0)) #数据域0的位置，也就是中间
    ax.spines['bottom'].set_position(('data', 0)) #数据域0的位置，也就是中间
    
    ax.xaxis.set_ticks_position("bottom") #x轴数据，置于下方
    ax.yaxis.set_ticks_position("left") #y轴数据，置于左边
    
    plt.xticks([-np.pi, -np.pi / 2.0, 0, np.pi / 2, np.pi],[r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$']) #x轴数据的位置、内容
    plt.yticks(np.linspace(-1, 1, 5, endpoint = True)) #纵轴数据的内容
    for label in ax.get_xticklabels() + ax.get_yticklabels(): #获取x轴和y轴的lable
        label.set_fontsize(16) #字号
        label.set_bbox(dict(facecolor = 'b', edgecolor = 'None', alpha = 0.2)) #label小方框的内容，背景颜色：白，不要边的颜色，透明度为0.2

    plt.legend(loc = 'upper left') #图例
    #plt.axis([-1, 1, -0.5, 1]) #显示范围：横轴起、横轴止、纵轴起、纵轴止
    
#     plt.fill_between(x, np.abs(x) < 0.5, c, c > 0.5, color='green', alpha = 0.25)
    plt.fill_between(x, s, c, color='green', alpha = 0.25) #填充????
    
    t = 1
    plt.plot([t, t], [0,np.cos(t)], 'y', linewidth = 3, linestyle = '--') #(t, 0),(t, np.cos(t))两点连线，y：黄色
    plt.annotate('cos(1)', xy = (t,np.cos(1)), xytext = (+10, +30),
    textcoords = 'offset points', arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3, rad=.2'))
    #标注，内容“cos(1)”,从xytext（xytext，以textcoords的形式，从xy移动（+10, =30））指向点xy,样式为arrowprops:弧度为0.2
    
    plt.grid() #显示网格
    
    
    
    plt.show() #展示
    
if __name__ == '__main__':
    main()