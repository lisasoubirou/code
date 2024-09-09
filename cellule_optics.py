# -*- coding: utf-8 -*-
"""
Created on Wed July  24  2024

@author: LS276867
"""

import numpy as np
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt
import xpart as xp
import sys
sys.path.append('/mnt/c/muco')
from rcsparameters.geometry.geometry import Geometry
# from ramping_module import ramping
# from track_function import track
import json

plt.rc('axes', labelsize=12)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

file_input='/mnt/c/muco/code/class_geometry/parameter_files/para_RCS_ME.txt'
RCS = Geometry(file_input, dipole_spacing=0.4,LSSS=3.4,nb_cell_arc=4)
time_frac=0
n_slice=0
option='var_ref'

time=time_frac
t_ref=0.5
energy=RCS.E_inj+(RCS.E_ext-RCS.E_inj)*time_frac

#Get data from class_geo
nb_cell_rcs=RCS.nb_cell_rcs
nb_arc=RCS.nb_arc
nb_cell_arc=RCS.nb_cell_arc
pattern=RCS.pattern
Ldd=RCS.dipole_spacing 
Ls=RCS.LSSS #Length of short straight section
Lins_cell=RCS.insertion_length/nb_cell_arc
Lcell=RCS.cell_length
list_hn=RCS.hn(time) #List of bendings
list_hn_ref=RCS.hn(t_ref) #List of reference bendings
theta=RCS.theta(time) #List of bending angle 
theta_ref=RCS.theta(t_ref)
epsilon=RCS.epsilon(time)
L_dip_path=RCS.L_dip_path(time)
L_dip_path_ref=RCS.L_dip_path(t_ref)

#RF data
N_turn_rcs=55
n_turns = N_turn_rcs*nb_cell_rcs
energy_increment_per_cell = (RCS.E_ext-RCS.E_inj)/n_turns
sync_phase=45
phase=180-sync_phase
volt_cell=energy_increment_per_cell/np.sin(phase*np.pi/180)
volt_cav=volt_cell

#Define elements: drifts, quads, sextupoles
mu=np.pi/2 #Phase advance 
f=RCS.cell_length/(4*np.sin(mu/2)) #Focusing strength of quadrupole
drift_dd=xt.Drift(length=Ldd)
drift_ins=xt.Drift(length=Lins_cell/2)

quad_f=xt.Quadrupole(k1=1/f/Ls, length=Ls)
quad_d=xt.Quadrupole(k1=-1/f/Ls, length=Ls)
step_quad=1e-8

#Cavity settings
cavity=xt.Cavity(voltage=volt_cell, frequency=1.3e9, lag=phase)
RF_acc=xt.ReferenceEnergyIncrease(Delta_p0c=energy_increment_per_cell)
dz_acc=xt.ZetaShift(dzeta=0)

beg = xt.Marker()
end = xt.Marker()

#Define dipoles manually
if option == 'var_ref':
    print('Option: var_ref')
    i_BNC=pattern.index('BNC')
    BNC=xt.Bend(k0=list_hn[i_BNC],
                    h=list_hn[i_BNC],
                    length=L_dip_path[i_BNC]
                    # length=length_sc
                    )   
    if len(set(pattern))>1:
        i_BSC=pattern.index('BSC')
        BSC=xt.Bend(k0=list_hn[i_BSC],
                    h=list_hn[i_BSC],
                    length=L_dip_path[i_BSC]
                    # length=length_sc
                    )
elif option == 'var_k':
    print('Option: var_k')
    i_BNC=pattern.index('BNC')
    BNC=xt.Bend(k0=list_hn[i_BNC],
                    h=list_hn_ref[i_BNC],
                    length=L_dip_path_ref[i_BNC]
                    # length=length_sc
                    )   
    if len(set(pattern))>1:
        i_BSC=pattern.index('BSC')
        BSC=xt.Bend(k0=list_hn[i_BSC],
                    h=list_hn_ref[i_BSC],
                    length=L_dip_path_ref[i_BSC]
                    # length=length_sc
                    )
else:
    raise ValueError("Invalid option: {}".format(option))

# If we want edges
# eps=RCS.epsilon(t_ref)
# hn=list_hn
# model='full'
# en1=xt.DipoleEdge(k=hn[0], e1=-eps[0], side='entry',model = model)
# ex1=xt.DipoleEdge(k=hn[0], e1=eps[1], side='exit',model =model)
# en2=xt.DipoleEdge(k=hn[1], e1=-eps[1], side='entry',model =model)
# ex2=xt.DipoleEdge(k=hn[1], e1=eps[2], side='exit',model =model)
# en3=xt.DipoleEdge(k=hn[2], e1=-eps[2], side='entry',model =model)
# ex3=xt.DipoleEdge(k=hn[2], e1=eps[3], side='exit',model =model)

# hcell=[drift_dd]+[en1,BSC,ex1]+ [drift_dd]+ [en2, BNC,ex2]+ [drift_dd]+[en3, BSC,ex3]+[drift_dd]
hcell=[drift_dd, BSC, drift_dd, BNC, drift_dd, BSC, drift_dd] #without edges
# hcell_name=['drift_dd']+['en1','BSC','ex1']+ ['drift_dd']+ ['en2', 'BNC','ex2']+ ['drift_dd']+['en3','BSC','ex3']+['drift_dd']
hcell_name=['drift_dd','BSC','drift_dd','BNC','drift_dd','BSC','drift_dd'] #without edeges

# FODO_elements=[quad_f,drift_s]+cell+[drift_s, quad_d,drift_s]+cell+[drift_s]
# FODO_names=['quad_f','drift_s']+cell_name+['drift_s', 'quad_d','drift_s']+cell_name+['drift_s']
FODO_elements=[beg, quad_f]+hcell+[quad_d]+hcell+[drift_ins, cavity, RF_acc, drift_ins, end]
FODO_names=['marker_beg', 'quad_f']+hcell_name+['quad_d']+hcell_name+['drift_ins',
                                                    'cavity','RF_acc','drift_ins','marker_end']

line_FODO=xt.Line(
    elements=FODO_elements,
    element_names=FODO_names)
line_FODO.particle_ref = xp.Particles(p0c=energy, #eV
                                    q0=1, mass0=xp.MUON_MASS_EV)
if n_slice > 0:
    line_FODO.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Teapot(n_slice))])

tw = line_FODO.twiss(method='6d') 

print('GLOBAL PARAMETER')
print('MCF=', round(tw['momentum_compaction_factor'], 6))
print('Qx=', round(tw['qx'], 5))
print('Qy', round(tw['qy'], 5))
print('dqx', round(tw['dqx'], 2))
print('dqy', round(tw['dqy'], 2))

fig1 = plt.figure(1, figsize=(6.4, 4.8*1.5))
bet = plt.subplot(2,1,1)
disp = plt.subplot(2,1,2, sharex=bet)

bet.plot(tw.s, tw.betx, label=r'$\beta_x$')
bet.plot(tw.s, tw.bety,label=r'$\beta_y$')
bet.set_ylabel(r'$\beta_{x,y}$ [m]')
bet.axvspan(tw['s','drift_ins'], tw['s','marker_end'],
                   color='b', alpha=0.1, linewidth=0)
bet.legend()

# spbet.legend()
disp.plot(tw.s, tw.dx,label=r'$D_x$')
disp.plot(tw.s, tw.dy,label=r'$D_y$')
disp.set_ylabel(r'$D_{x,y}$ [m]')
disp.legend()
disp.axvspan(tw['s','drift_ins'], tw['s','marker_end'],
                   color='b', alpha=0.1, linewidth=0)
fig1.subplots_adjust(left=.15, right=.92, hspace=.27)
plt.show()