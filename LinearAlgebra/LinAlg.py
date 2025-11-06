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
    
def yhat(x, y): #Task 4
    ab = least_square(x, y)
    A_t = np.ones((2, len(x)))
    A_t[0] = x
    A = np.transpose(A_t)
    yhat = np.linalg.matmul(A, ab)
    return yhat

def projection(x, y): #Task 4
    v = yhat(x, y)[:, np.newaxis]
    v_t = np.transpose(v)
    P = np.linalg.matmul(v, v_t)/np.linalg.matmul(v_t, v)
    return P

def projection_test(x, y): #Task 5
    P = projection(x, y)
    P_t = np.transpose(P)
    P2 = np.linalg.matmul(P, P)
    
    idempotent, symmetric = True, True
    allowance = 10**(-16)
    
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if (P2 - P)[i,j] > allowance:
                idempotent = False
            elif (P_t - P)[i,j] > allowance:
                symmetric = False
    
    print("Idempotent: {} \nSymmetric: {}".format(idempotent, symmetric))
    
def ort_proj_check(proj, orig):
    allowance = 10**(-8)
    if np.dot(proj, orig-proj) > allowance:
        print("The vector is not the the result of an orthogonal projection of the original vector")
    else: 
        print("The vector is the the result of an orthogonal projection of the original vector")
        
def reflection(x, y):
    P = projection(x, y)
    I = np.identity(P.shape[0])
    R = 2*P - I
    return R
    

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

#%%
plt.subplot(2,2,1)
plot(result_square[0], x1, y1)
plt.scatter(x1, yhat(x1,y1))
plt.scatter(x1, np.matmul(reflection(x1, y1), y1))
plt.subplot(2,2,2)
plot(result_square[1], x2, y2[0])
plt.scatter(x2, yhat(x2,y2[0]))
plt.scatter(x2, np.matmul(reflection(x2, y2[0]), y2[0]))
plt.subplot(2,2,3)
plot(result_square[2], x2, y2[1])
plt.scatter(x2, yhat(x2,y2[1]))
plt.scatter(x2, np.matmul(reflection(x2, y2[1]), y2[1]))
plt.subplot(2,2,4)
plot(result_square[3], x2, y2[2])
plt.scatter(x2, yhat(x2,y2[2]))
plt.scatter(x2, np.matmul(reflection(x2, y2[2]), y2[2]))
plt.show()


#%%
surfaceplot(result_square[0], x1, y1)
surfaceplot(result_square[1], x2, y2[0])
surfaceplot(result_square[2], x2, y2[1])
surfaceplot(result_square[3], x2, y2[2])

#%%
print("x1, x2:")
projection_test(x1, y1)
print("\nx2, y2[0]:")
projection_test(x2, y2[0])
print("\nx2, y2[1]:")
projection_test(x2, y2[1])
print("\nx2, y2[2]:")
projection_test(x2, y2[2])

#%%
print("x1, y:1")
ort_proj_check(yhat(x1,y1), y1)
print("\nx2, y2[0]:")
ort_proj_check(yhat(x2,y2[0]), y2[0])
print("\nx2, y2[1]:")
ort_proj_check(yhat(x2,y2[1]), y2[1])
print("\nx2, y2[2]:")
ort_proj_check(yhat(x2,y2[2]), y2[2])
