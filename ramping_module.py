# -*- coding: utf-8 -*-
"""
Created on March 28

@author: LS276867
"""

import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt
import xpart as xp
import sys
sys.path.append('/mnt/c/muco/code')
from class_geometry.class_geo import Geometry 
from interpolation import calc_dip_coef,eval_horner

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize

#Constants
muon_mass_kg=sc.physical_constants['muon mass'][0]
e_charge=sc.e
c=sc.c

def ramping(RCS, line, method, t_ref, n_turn_rcs, RF_tuning=True): 
    """ Attach the times varying knobs to the dipoles and edges. 
    EDGES FOR A 3 DIPOLE PATTERN OLY SO FAR !!
    Args:
        RCS (_type_): class Geometry object
        line (_type_): line to ramp on 
        method (_type_): var_k or var_ref
        t_ref (_type_): time reference used during the acc
        n_turn_rcs (_type_): Number of turns done in the rcs (to calculate acc time)
        RF_tuning (bool, optional): True if we want RF tuning, False otherwise. Defaults to True.

    """

    t_acc=RCS.C*n_turn_rcs/c
    tab=line.get_table()
    # line.discard_tracker()
    pattern=RCS.pattern 
    i_BNC=pattern.index('BNC')
    i_BSC=pattern.index('BSC')

    line.vars['t_turn_s']=0.

    coef=calc_dip_coef(RCS,t_ref)
    line.functions.h_nc_pol= lambda x: eval_horner(coef[0],x)
    line.functions.h_sc_pol= lambda x: eval_horner(coef[1],x)
    line.functions.l_nc_pol= lambda x: eval_horner(coef[2],x)
    line.functions.l_sc_pol= lambda x: eval_horner(coef[3],x)
    line.functions.path_diff_1cav= lambda x: -eval_horner(coef[4],x)/2

    line.vars['h_NC']=line.functions.h_nc_pol(line.vars['t_turn_s']/t_acc)
    line.vars['h_SC']=line.functions.h_sc_pol(line.vars['t_turn_s']/t_acc)
    line.vars['l_dip_NC']=line.functions.l_nc_pol(line.vars['t_turn_s']/t_acc)
    line.vars['l_dip_SC']=line.functions.l_sc_pol(line.vars['t_turn_s']/t_acc)
    line.vars['dz_rf']=line.functions.path_diff_1cav(line.vars['t_turn_s']/t_acc)

    if RF_tuning is True:
        line.element_refs['dz_acc'].dzeta=line.vars['dz_rf']
        line.element_refs['dz_acc_1'].dzeta=line.vars['dz_rf']

    h_ref_NC=RCS.hn(t_ref)[i_BNC]
    h_ref_SC=RCS.hn(t_ref)[i_BSC]
    l_dip_ref_NC=RCS.L_dip_path(t_ref)[i_BNC]
    l_dip_ref_SC=RCS.L_dip_path(t_ref)[i_BSC]

    for el in tab.rows[tab.element_type == 'Bend'].name:
        if method == 'var_ref':
            if 'BSC' in el:
                line.element_refs[el].k0=line.vars['h_SC']
                line.element_refs[el].h=line.vars['h_SC']
                line.element_refs[el].length=line.vars['l_dip_SC']
            elif 'BNC' in el:
                line.element_refs[el].k0=line.vars['h_NC']
                line.element_refs[el].h=line.vars['h_NC']
                line.element_refs[el].length=line.vars['l_dip_NC']
        elif method == 'var_k':
            if 'BSC' in el:  
                line.element_refs[el].k0=line.vars['h_SC']
                line.element_refs[el].h=h_ref_SC
                line.element_refs[el].length=l_dip_ref_SC
            elif 'BNC' in el:
                line.element_refs[el].k0=line.vars['h_NC']
                line.element_refs[el].h=h_ref_NC
                line.element_refs[el].length=l_dip_ref_NC
        else:
            raise ValueError("Invalid option: {}".format(method))
        
    #Time dependent knobs on dipole edge strength    
    for el in tab.rows[tab.element_type == 'DipoleEdge'].name:
        if 'en1' or 'ex1' or 'en3' or 'ex3' in el:
                line.element_refs[el].k=line.vars['h_SC']
        elif 'en2' or 'ex2' in el:
                line.element_refs[el].k=line.vars['h_NC']