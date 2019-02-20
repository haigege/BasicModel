import numpy as np
from scipy.optimize.optimize import rosen
from pylab import *

def main():
    ##1-- Integral
    from scipy.integrate import quad, dblquad, nquad #dblquad为二元积分, nquad为n维积分
    print(quad(lambda x : np.exp(-x), 0, np.inf)) #0到+无穷的积分，结果：(1.0000000000000002, 5.842606703608969e-11),值 和 误差
    print(dblquad(lambda t, x : np.exp(-x * t) / t ** 3, 0, np.inf, lambda x : 1, lambda x : np.inf))
    
    def f(x, y):
        return x * y
    def bound_y():
        return [0, 0.5] #定义边界
    def bound_x(y):
        return [0, 1-2 * y]
    print(nquad(f, [bound_x,bound_y]))
    
    
    ##2--Optimizer
    from scipy.optimize import minimize #最小值模块
    def rosen(x):
        return sum(100.0 * (x[1:] - x[:-1] ** 2.0) + (1 - x[:-1]) ** 2.0)
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    # nelder-mead单纯形法,是一类直接搜索算法
    res = minimize(rosen, x0, method = 'nelder-mead', options = {'xtol': 1e-8, 'disp': True})
#     print('ROSE MINI: ', res.x)
    print('ROSE MINI: ', res)
    
    def func(x):
        return -(2 * x[0] * x[1] + 2 * x[0] - x[0] ** 2 -2 * x[1] ** 2)
    def func_deriv(x): #为了计算更快速，提前写出偏导数
        dfdx0 = -(-2 * x[0] + 2 * x[1] + 2)
        dfdx1 = -(2 * x[0] - 4 * x[1])
        return np.array([dfdx0, dfdx1])
    # 约束条件，“jac”:雅可比行列式
    cons = ({'type':'eq', 'fun':lambda x:np.array([x[0]**3 - x[1]]), 'jac':lambda x:np.array([3.0*(x[0]**2.0), -1.0])},
            {'type':'ineq', 'fun':lambda x:np.array([x[1]-1]), 'jac':lambda x:np.array([0.0, 1.0])})
    res = minimize(func, [-1.0, 1.0], jac = func_deriv, constraints = cons, method = 'SLSQP', options = {'disp':True})
    print('RESTRICT:',res)
    
    from scipy.optimize import root #求根
    def fun(x):
        return x + 2 * np.cos(x)
    sol = root(fun, 10)
    print('ROOT:',sol.x,', FUN:',sol.fun)
    
    
    ##3--Interpolation 
#     from pylab import * #置于模块前
    x = np.linspace(0, 1, 10)
    y = np.sin(2 * np.pi * x)
    from scipy.interpolate import interp1d #引入插值算法模块，interp1d为一维插值
    li = interp1d(x, y, kind = 'cubic')
    x_new = np.linspace(0, 1, 50)
    y_new = li(x_new)
    figure()
    plot(x, y, 'r') #红色表示原数据
    plot(x_new, y_new, 'k') #黑色表示新数据
    print(y_new)
    show()
    
    ##4--
    from scipy import linalg as lg
    arr = np.array([[1, 2], [3, 4]])
    print('Det:', lg.det(arr)) #行列式
    print('Inv:', lg.inv(arr)) #逆矩阵
    b = np.array([6, 14])
    print('Sol:', lg.solve(arr, b)) #解方程
    print('Eig:', lg.eig(arr)) #特征值 和 特征向量
    
    #矩阵的四种分解
    print('LU:', lg.lu(arr))
    print('QR:', lg.qr(arr))
    print('SVD:', lg.svd(arr))
    print('Schur:', lg.schur(arr))
    
if __name__ == '__main__':
    main()