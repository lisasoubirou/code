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


degree_h=4
degree_l=5
degree_path=6

def interpolation_coeffs(name, x, y, degree):
    """Perform polynomial interpolation and return the coefficients.

    Args:
        name (string): Name of the variable to fit.
        x (array_like): x-coordinates of the data points to fit.
        y (array_like): y-coordinates of the data points to fit.
        degree (integer): The degree of the polynomial to fit.
    Retuns:    
        coefficients (ndarray): Coefficients of the polynomial fit.
    """    
    res=np.polyfit(x, y, degree, full=True)
    print('Fit result for: ', name)
    print('polynomial degree:', degree)
    print('residuals:', res[1][0])
    print('rcond:',res[-1])
    return(res[0])


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

def calc_dip_coef(file_input,t_ref):
    RCS = Geometry(file_input)
    pattern=RCS.pattern 
    i_BNC=pattern.index('BNC')
    i_BSC=pattern.index('BSC')

    #Calculate data to fit from RCS
    h_nc=[]
    h_sc=[]
    l_sc=[]
    l_nc=[]
    path_var=[]
    time=np.linspace(0,1,50)
    for t in time:
        length=RCS.L_dip_path(t)
        bending=RCS.hn(t)
        path=2*RCS.nb_cell_arc*(RCS.path_length(t)-RCS.path_length(t_ref))
        l_nc.append(length[i_BNC])
        l_sc.append(length[i_BSC])
        h_nc.append(bending[i_BNC])
        h_sc.append(bending[i_BSC])
        path_var.append(path)

    coef_h_nc=interpolation_coeffs('h_nc',time,h_nc,degree_h)
    coef_h_sc=interpolation_coeffs('h_sc',time,h_sc,degree_h)
    coef_l_nc=interpolation_coeffs('l_nc',time,l_nc,degree_l)
    coef_l_sc=interpolation_coeffs('l_sc',time,l_sc,degree_l)
    coef_path=interpolation_coeffs('path',time,path_var,degree_path)
    return (coef_h_nc,coef_h_sc,coef_l_nc,coef_l_sc,coef_path)

if __name__ == "__main__":
    file_input='/mnt/c/muco/code/class_geometry/para_RCS_ME.txt'
    RCS = Geometry(file_input)
    t_ref=0.5
    pattern=RCS.pattern 
    time_test=np.linspace(0,1,40)
    time=np.linspace(0,1,50)
    h_nc=[]
    h_sc=[]
    l_sc=[]
    l_nc=[]
    path_var=[]
    i_BNC=pattern.index('BNC')
    i_BSC=pattern.index('BSC')
    for t in time:
        length=RCS.L_dip_path(t)
        bending=RCS.hn(t)
        path=2*RCS.nb_cell_arc*(RCS.path_length(t)-RCS.path_length(t_ref))
        l_nc.append(length[i_BNC])
        l_sc.append(length[i_BSC])
        h_nc.append(bending[i_BNC])
        h_sc.append(bending[i_BSC])
        path_var.append(path)
    coef_h_nc,coef_h_sc,coef_l_nc,coef_l_sc,coef_path=calc_dip_coef(file_input,t_ref)
    h_nc_pol=eval_horner(coef_h_nc,time_test)
    h_sc_pol=eval_horner(coef_h_sc,time_test)
    l_nc_pol=eval_horner(coef_l_nc,time_test)
    l_sc_pol=eval_horner(coef_l_sc,time_test)
    path_pol=eval_horner(coef_path,time_test)

    h_nc_geo=[RCS.hn(t)[i_BNC] for t in time_test]
    h_sc_geo=[RCS.hn(t)[i_BSC] for t in time_test]
    l_nc_geo=[RCS.L_dip_path(t)[i_BNC] for t in time_test]
    l_sc_geo=[RCS.L_dip_path(t)[i_BSC] for t in time_test]
    path_geo=[2*RCS.nb_cell_arc*(RCS.path_length(t)-RCS.path_length(t_ref)) for t in time_test]

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

    plt.figure(3,figsize=(10, 5))
    ax1 = plt.subplot(1, 2, 1, xlabel='Time', ylabel='path length diff [m]')
    ax1.plot(time, path_var, color='green')
    ax1.scatter(time_test, path_pol, color='black', marker='+')
    plt.tight_layout()
    plt.show()

    plt.figure(4,figsize=(10, 5))
    ax1 = plt.subplot(1, 2, 1, xlabel='Time', ylabel='h_SC [1/m]')
    ax2 = plt.subplot(1, 2, 2, xlabel='Time', ylabel='h_NC [1/m]')
    ax1.scatter(time_test, h_sc_geo-h_sc_pol, color='black', marker='+')
    ax2.scatter(time_test, h_nc_geo-h_nc_pol, color='black',marker='+')
    plt.tight_layout()
    plt.show()

    plt.figure(5,figsize=(10, 5))
    ax1 = plt.subplot(1, 2, 1, xlabel='Time', ylabel='l_SC [m]')
    ax2 = plt.subplot(1, 2, 2, xlabel='Time', ylabel='l_NC [m]')
    ax1.scatter(time_test, l_sc_geo-l_sc_pol, color='black', marker='+')
    ax2.scatter(time_test, l_nc_geo-l_nc_pol, color='black',marker='+')
    plt.tight_layout()
    plt.show()

    plt.figure(5,figsize=(10, 5))
    ax1 = plt.subplot(1, 2, 1, xlabel='Time', ylabel='path length diff [m]')
    ax1.scatter(time_test, path_geo-path_pol, color='black', marker='+')
    plt.tight_layout()
    plt.show()


