# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 08:53:13 2025

@author: Herman Plank
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def form(params, x):
    a,b = params
    return a * x + b

def square_fcn(params, x, y):
    return np.sum((y - form(params,x)) ** 2)

def least_square(x, y):
    A_t = np.ones((2, len(x)))
    A_t[0] = x
    A = np.transpose(A_t)
    A_tA_inv = np.linalg.inv(np.linalg.matmul(A_t, A))
    ab = np.linalg.matmul(np.linalg.matmul(A_tA_inv, A_t), y)
    return ab

def plot(ab, x, y):
    X = np.linspace(min(x)-2, max(x)+2, 100)
    Y = ab[0]*X + ab[1]
    plt.axis('equal')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x, y, c='red')
    plt.title(f'a = {ab[0]:.3f}, b = {ab[1]:.3f}')
    plt.plot(X, Y, 'blue',)
    
def surfaceplot(ab, x, y):
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel(r'$|\hat{y}-y|$')
    a = np.linspace(ab[0]-5, ab[0]+5, 41)
    b = np.linspace(ab[1]-10, ab[1]+10, 41)
    z = np.zeros((len(a), len(b)))
    zmax = 0
    for i in range(len(a)):
        for j in range(len(b)):
            z[i, j] = np.sqrt(np.sum((y - a[i]*x - b[j]) ** 2))
            if z[i, j] > zmax:
                zmax = z[i, j]
    a, b = np.meshgrid(a, b)
    A = ab[0]
    B = ab[1]
    Z = np.linspace(0, zmax, 2)
    plt.subplot(1,2,1)
    ax.plot_surface(b, a, z, cmap=cm.Reds, linewidth=0, antialiased=False, label=f'a = {A:.3f}, b = {B:.3f}')
    ax.plot3D(A, B, Z, c='black', linewidth=3)
    ax.scatter(ab[0], ab[1], np.sum((y - ab[0]*x - ab[1]) ** 2), c='black', s=100)
    ax.legend()
    plt.subplot(1,2,2)
    plt.pcolormesh(a,b,z, cmap=cm.Reds)
    plt.colorbar(location='top')
    plt.scatter(ab[0], ab[1], c='k', marker='*', label=f'a = {A:.3f}, b = {B:.3f}')
    plt.legend()
    
def P(x, y):
    x_t = np.ones((2, len(x)))
    x_t[0] = x
    x = np.transpose(x_t)  
    P = np.linalg.matmul(x, np.linalg.matmul(np.linalg.inv(np.linalg.matmul(x_t, x)), x_t))
    return P

def yhat(x, y):
    ab = least_square(x1, y1)
    A_t = np.ones((2, len(x)))
    A_t[0] = x
    A = np.transpose(A_t)
    yhat = np.linalg.matmul(A,ab)
    return yhat
    
    
    

data1 = pd.read_csv('regression_1.dat',sep=r'\s*,\s*')
data2 = pd.read_csv('regression_2.dat',sep=r'\s*,\s*')

x1 = np.array(data1['x'])
y1 = np.array(data1['y'])
x2 = np.array(data2['x'])
y2 = np.array([data2['y1'], data2['y2'], data2['y3']])


result_scipy = []
result_scipy.append(fmin(square_fcn, [1.,0.], (x1,y1)))
result_scipy.append(fmin(square_fcn, [1.,0.], (x2,y2[0])))
result_scipy.append(fmin(square_fcn, [1.,0.], (x2,y2[1])))
result_scipy.append(fmin(square_fcn, [1.,0.], (x2,y2[2])))

result_square = []
result_square.append(least_square(x1, y1))
result_square.append(least_square(x2, y2[0]))
result_square.append(least_square(x2, y2[1]))
result_square.append(least_square(x2, y2[2]))


plt.subplot(2,2,1)
plot(result_square[0], x1, y1)
#plt.scatter(x1, yhat)
plt.subplot(2,2,2)
plot(result_square[1], x2, y2[0])
plt.subplot(2,2,3)
plot(result_square[2], x2, y2[1])
plt.subplot(2,2,4)
plot(result_square[3], x2, y2[2])
plt.show()

#surfaceplot(result_square[0], x1, y1)
#surfaceplot(result_square[1], x2, y2[0])
#surfaceplot(result_square[2], x2, y2[1])
#surfaceplot(result_square[3], x2, y2[2])
a= np.linalg.matmul(P(x1, y1), y1)
b= yhat(x1, y1)
plt.subplot(2,2,1)
plt.scatter(x1, a, color='black', marker='*')
plt.scatter(x1, b, color='g')
plt.show()
a-b