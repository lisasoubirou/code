# -*- coding: utf-8 -*-
"""
Created on March 29
@author: LS276867
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator
import sys
sys.path.append('/mnt/c/muco')
from class_geometry.class_geo import Geometry 
import json

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

#Importing line
file_input='/mnt/c/muco/code/class_geometry/para_RCS_ME.txt'
RCS = Geometry(file_input)

pattern=RCS.pattern 
i_BNC=pattern.index('BNC')
i_BSC=pattern.index('BSC')

h_nc=[]
h_sc=[]
l_sc=[]
l_nc=[]
time=np.linspace(0,1,50)
for t in time:
    length=RCS.L_dip_path(t)
    bending=RCS.hn(t)
    l_nc.append(length[i_BNC])
    l_sc.append(length[i_BSC])
    h_nc.append(bending[i_BNC])
    h_sc.append(bending[i_BSC])

def polynomial_interpolation_coef(x, y, degree):
    """
    Perform polynomial interpolation and return the coefficients.
    Parameters:
        x (array_like): The x-coordinates of the data points.
        y (array_like): The y-coordinates of the data points.
        degree (int): The degree of the polynomial to fit.

    Returns:
        coefficients (ndarray): Coefficients of the polynomial fit.
                               The coefficients are in decreasing order of powers.
    """
    coefficients = np.polyfit(x, y, degree)
    return coefficients

def eval_horner(coefficients, x):
    """
    Evaluate the polynomial defined by its coefficients at the given x-values using Horner's method.
    Parameters:
        coefficients (array_like): Coefficients of the polynomial.
        x (array_like): The x-values at which to evaluate the polynomial.
    Returns:
        y (ndarray): The y-values corresponding to the evaluated polynomial.
    """
    n = len(coefficients)
    y = np.zeros_like(x)
    for coeff in coefficients:
        y = y * x + coeff
    return y

degree_h=4
degree_l=5
coef_h_nc=polynomial_interpolation_coef(time,h_nc,degree_h)
coef_h_sc=polynomial_interpolation_coef(time,h_sc,degree_h)
coef_l_nc=polynomial_interpolation_coef(time,l_nc,degree_l)
coef_l_sc=polynomial_interpolation_coef(time,l_sc,degree_l)
time_test=np.linspace(0,1,40)
h_nc_pol=eval_horner(coef_h_nc,time_test)
h_sc_pol=eval_horner(coef_h_sc,time_test)
l_nc_pol=eval_horner(coef_l_nc,time_test)
l_sc_pol=eval_horner(coef_l_sc,time_test)

plt.figure(1,figsize=(10, 5))
ax1 = plt.subplot(1, 2, 1, xlabel='Time', ylabel='L_SC [m]')
ax2 = plt.subplot(1, 2, 2, xlabel='Time', ylabel='L_NC [m]')
ax1.plot(time, l_sc, color='blue')
ax1.scatter(time_test, l_sc_pol, color='blue', marker='+')
ax2.plot(time, l_nc, color='red')
ax2.scatter(time_test, l_nc_pol, color='red', marker='+')
plt.tight_layout()
plt.show()

plt.figure(2,figsize=(10, 5))
ax1 = plt.subplot(1, 2, 1, xlabel='Time', ylabel='h_SC [1/m]')
ax2 = plt.subplot(1, 2, 2, xlabel='Time', ylabel='h_NC [1/m]')
ax1.plot(time, h_sc, color='blue')
ax1.scatter(time_test, h_sc_pol, color='black', marker='+')
ax2.plot(time, h_nc, color='red')
ax2.scatter(time_test, h_nc_pol, color='black',marker='+')
plt.tight_layout()
plt.show()





