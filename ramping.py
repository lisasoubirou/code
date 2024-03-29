# -*- coding: utf-8 -*-
"""
Created on March 28

@author: LS276867
"""

import numpy as np
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt
import xpart as xp
import sys
sys.path.append('/mnt/c/muco')
from class_geometry.class_geo import Geometry 
from optics_function import plot_twiss
import json

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

#Importing line
method='var_k'
t_ref=0.5
t_acc=1.09704595

file_input='/mnt/c/muco/code/class_geometry/para_RCS_ME.txt'
RCS = Geometry(file_input)
file_seq = 'lattice_disp_suppr_6d_no_slicing.json'
line=xt.Line.from_json(file_seq)
line.particle_ref = xt.Particles(mass0=xp.MUON_MASS_EV, q0=1.,
                                    energy0=xp.MUON_MASS_EV + RCS.E_inj,
                                 )
tw_6d=line.twiss(method='6d', matrix_stability_tol=5e-3)

tab=line.get_table()
line.discard_tracker()

pattern=RCS.pattern 
i_BNC=pattern.index('BNC')
i_BSC=pattern.index('BSC')

# line.vars['t_turn_s']
# line.vars['h_NC']=0
# line.vars['h_SC']=0
# line.vars['l_dip_NC']=0
# line.vars['l_dip_SC']=0

line._init_var_management()
line.vars['t_turn_s']=0.
def h_geo_nc(x, i=i_BNC, tt=t_acc):
    return RCS.hn(x/tt)[i]
exit

line.functions.h_geo_nc = h_geo_nc
line.vars['h_NC']=line.functions.h_geo_nc(line.vars['t_turn_s'])
line.functions.h_geo_sc = lambda x: RCS.hn(x/t_acc)[i_BSC]
line.vars['h_SC']=line.functions.h_geo_sc(line.vars['t_turn_s'])

# line.vars['h_NC']=RCS.hn(line.vars['t_turn_s'].vv/t_acc)[i_BNC]
# line.vars['h_SC']=RCS.hn(line.vars['t_turn_s'].vv/t_acc)[i_BSC]
line.vars['l_dip_NC']=RCS.L_dip_path(line.vars['t_turn_s'].vv/t_acc)[i_BNC]
line.vars['l_dip_SC']=RCS.L_dip_path(line.vars['t_turn_s'].vv/t_acc)[i_BSC]
line.vars['test']=2*line.vars['t_turn_s']

h_ref_NC=RCS.hn(t_ref)[i_BNC]
h_ref_SC=RCS.hn(t_ref)[i_BSC]
l_dip_ref_NC=RCS.L_dip_path(t_ref)[i_BNC]
l_dip_ref_SC=RCS.L_dip_path(t_ref)[i_BSC]

for el in tab.rows[tab.element_type == 'Bend'].name:
    if 'BSC' in el:
        if method == 'var_ref':
            line.element_refs[el].k0=line.vars['h_SC']
            line.element_refs[el].h=line.vars['h_SC']
            line.element_refs[el].length=line.vars['l_dip_SC']
        elif method == 'var_k':
            line.element_refs[el].k0=line.vars['h_SC']
            line.element_refs[el].h=h_ref_SC
            line.element_refs[el].length=l_dip_ref_SC
        else:
            raise ValueError("Invalid option: {}".format(method))
    elif 'BNC' in el:
        if method == 'var_ref':
                line.element_refs[el].k0=line.vars['h_NC']
                line.element_refs[el].h=line.vars['h_NC']
                line.element_refs[el].length=line.vars['l_dip_NC']
        elif method == 'var_k':
            line.element_refs[el].k0=line.vars['h_NC']
            line.element_refs[el].h=h_ref_NC
            line.element_refs[el].length=l_dip_ref_NC
        else:
            raise ValueError("Invalid option: {}".format(method))

print('BSC',line['BSC'].k0)
print('BNC',line['BNC'].k0)
print('test',line.vars['test']._get_value())
line.vars['t_turn_s']=0.5
print('BSC',line['BSC'].k0)
print('BNC',line['BNC'].k0)
print('test',line.vars['test']._get_value())



line.enable_time_dependent_vars = True
