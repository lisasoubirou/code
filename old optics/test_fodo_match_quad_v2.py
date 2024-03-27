# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:10:16 2024
@author: LS276867

Old code
Test optics: dispersion suppressor (final fodo design) => NOW USE optics_function.py 
"""

#%% Init
import numpy as np
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt
import xpart as xp
import sys
sys.path.append('C:\muco')
from class_geometry.class_geo import Geometry 
import json

#Call class_geo
file_input='/mnt/c/muco/class_geometry/para_RCS_ME.txt'
RCS = Geometry(file_input) 

#Time and energy of study
time=RCS.phi_inj
energy=RCS.E_inj

#Get data from class_geo
Ldd=RCS.dipole_spacing 
Ls=RCS.LSSS #Length of short straight section
Lqd=RCS.QP_dipole_spacing 
Lins=RCS.insertion_length
Lcell=RCS.cell_length
list_hn=RCS.hn(time) #List of bendings
theta=RCS.theta(time) #List of bending angle 
L_dd_path=RCS.L_dd_path(time) 
L_dip=RCS.L_dip(time)
pattern=RCS.pattern 
nb_cell_tot=RCS.nb_cell_arcs
nb_arc=RCS.nb_arc

N_cell=1
#Define elements
mu=np.pi/2 #Phase advance 
#mu=2*np.pi*(0.25+0.22/nb_cell_tot)  
f=RCS.cell_length/(4*np.sin(mu/2)) #Focusing strength of quadrupole
drift_qd=xt.Drift(length=Lqd)
drift_dd=xt.Drift(length=Ldd)
drift_s=xt.Drift(length=Ls/2)
drift_s_sxt=xt.Drift(length=Ls/2-Lqd)
#drift_r=xt.Drift(length=Lcell/2/2)
drift_r=xt.Drift(length=Lins/2/2)

quad_f=xt.Multipole(knl=[0., 1/f], ksl=[0., 0.])
quad_d=xt.Multipole(knl=[0., -1/f], ksl=[0., 0.])
quad_f1=xt.Multipole(knl=[0., 1/f], ksl=[0., 0.])
quad_d1=xt.Multipole(knl=[0., -1/f], ksl=[0., 0.])
quad_f2=xt.Multipole(knl=[0., 1/f], ksl=[0., 0.])
quad_d2=xt.Multipole(knl=[0., -1/f], ksl=[0., 0.])
quad_f3=xt.Multipole(knl=[0., 1/f], ksl=[0., 0.])
quad_d3=xt.Multipole(knl=[0., -1/f], ksl=[0., 0.])

s_f=1/(f*0.5)
s_d=-1/(f*0.26)
sxt_d=xt.Multipole(knl=[0., 0.,s_d], ksl=[0., 0.,0.])
sxt_f=xt.Multipole(knl=[0., 0.,s_f], ksl=[0., 0.,0.])

V=0
freq=0
phi=0
cavity=xt.Cavity(voltage=V, frequency=freq, lag=phi)

beg = xt.Marker()
end = xt.Marker()

#Define line manually
length_nc=RCS.dipole_families['BNC']['length']
i_BNC=pattern.index('BNC')
BNC=xt.Multipole(length=length_nc, 
                 knl=[theta[i_BNC]], 
                 ksl=[0.0], hxl=theta[i_BNC], hyl=0. )  
BNC2=xt.Multipole(length=length_nc,    #Half angle dipole
                 knl=[theta[i_BNC]/2], 
                 ksl=[0.0], hxl=theta[i_BNC]/2, hyl=0. )  
drift_BNC=xt.Drift(length=length_nc/2)
if len(set(pattern))>1:
    length_sc=RCS.dipole_families['BSC']['length']
    i_BSC=pattern.index('BSC')
    BSC=xt.Multipole(length=length_sc,
                     knl=[theta[i_BSC]], 
                     ksl=[0.0], hxl=theta[i_BSC], hyl=0. ) #,  
    BSC2=xt.Multipole(length=length_sc,  #Half angle dipole
                     knl=[theta[i_BSC]/2], 
                     ksl=[0.0], hxl=theta[i_BSC]/2, hyl=0. ) #,  
    drift_BSC=xt.Drift(length=length_sc/2)

#%% Optics
#FODO cell
cell=[drift_qd, drift_BSC, BSC, drift_BSC, drift_dd, drift_BNC, BNC, drift_BNC, 
      drift_dd,drift_BSC, BSC, drift_BSC, drift_qd]
cell2=[drift_qd, drift_BSC, BSC2, drift_BSC, drift_dd, drift_BNC, BNC2, drift_BNC, 
       drift_dd,drift_BSC, BSC2, drift_BSC, drift_qd]
FODO_elements=[quad_f,drift_s]+cell+[drift_s, quad_d,drift_s]+cell+[drift_s]

cell_name=['drift_qd','drift_bsc', 'dip_bsc','drift_bsc', 'drift_dd', 
                'drift_bnc', 'dip_bnc', 'drift_bnc','drift_dd','drift_bsc','dip_bsc','drift_bsc',
                'drift_qd']
FODO_names=['quad_f','drift_s']+cell_name+['drift_s', 'quad_d','drift_s']+cell_name+['drift_s']

line_FODO=xt.Line(
    elements=FODO_elements,
    element_names=FODO_names)
line_FODO.particle_ref = xp.Particles(p0c=energy, #eV
                                      q0=1, mass0=xp.MUON_MASS_EV)

tw_FODO = line_FODO.twiss(method='4d') 
index_qpd=np.where(tw_FODO['name']=='quad_d')[0][0]
betx_fodo=tw_FODO['betx'][index_qpd]
bety_fodo=tw_FODO['bety'][index_qpd]     
dx_fodo=tw_FODO['dx'][index_qpd]     

#Transfer
tr_elements= [beg]+[quad_f1, drift_qd, sxt_f,drift_s_sxt]+cell+[drift_s, quad_d1, drift_qd, 
                sxt_d,drift_s_sxt]+cell+[drift_s,quad_f2,drift_s]+cell+[drift_s, quad_d2,
                    drift_s]+cell+[drift_s, quad_f3,drift_r, cavity, drift_r, quad_d3]+[end]
tr_names=['beg']+['quad_f', 'drift_qd', 'sxt_f','drift_s']+cell_name+['drift_s', 'quad_d', 'drift_qd', 
                'sxt_d','drift_s']+cell_name+['drift_s','quad_f','drift_s']+cell_name+['drift_s',
                    'quad_d', 'drift_s']+cell_name+['drift_s', 'quad_f','drift_r','cavity','drift_r', 
                                                     'quad_d']+['end']

line_tr=xt.Line(
    elements=tr_elements,
    element_names=tr_names)
line_tr.particle_ref = xp.Particles(p0c=energy, #eV
                                      q0=1, mass0=xp.MUON_MASS_EV)

#Matches FODO lattices to the transfer line: on first quad of the transfer == output optics of FODO
tw_init_FODO = tw_FODO.get_twiss_init(0)
tw_init_FODO.element_name = 'beg'
tw_tr=line_tr.twiss(start=line_tr.element_names[0],end='_end_point', method='4d',
                    # ele_init= "drift_ini",
                    init=tw_init_FODO
                    )

#Print ini quad strength
print('QUAD0', round(line_tr['quad_f'].knl[1], 3))
print('QUAD1', round(line_tr['quad_d'].knl[1], 3))
print('QUAD2', round(line_tr['quad_f_1'].knl[1], 3))
print('QUAD3', round(line_tr['quad_d_1'].knl[1], 3))
print('QUAD4', round(line_tr['quad_f_2'].knl[1], 3))
print('QUAD5', round(line_tr['quad_d_2'].knl[1], 3))

#Init of variables to be varied
line_tr._init_var_management()
line_tr.vars['k_f0'] = 1/f
line_tr.vars['k_d1'] = -1/f
line_tr.vars['k_f1'] = 1/f
line_tr.vars['k_d2'] = -1/f
line_tr.vars['k_f2'] = 1/f
line_tr.vars['k_d3'] = -1/f/2   #Half quad at the beginning and end of line_ds

line_tr.element_refs['quad_f'].knl[1] = line_tr.vars['k_f0']
line_tr.element_refs['quad_d'].knl[1] = line_tr.vars['k_d1']
line_tr.element_refs['quad_f_1'].knl[1] = line_tr.vars['k_f1']
line_tr.element_refs['quad_d_1'].knl[1] = line_tr.vars['k_d2']
line_tr.element_refs['quad_f_2'].knl[1] = line_tr.vars['k_f2']
line_tr.element_refs['quad_d_2'].knl[1] = line_tr.vars['k_d3']

#Matching
line_tr.match(
              start='beg',
              end='_end_point', 
              init=tw_init_FODO,
              vary=xt.VaryList([
                                'k_f0',
                                'k_d1',
                                'k_f1',
                                'k_d2',
                                'k_f2',
                                'k_d3'
                                ],
                      step=1e-8),
              targets=[
                    xt.Target('dx', at='end', value=0. ),
                    xt.Target('dpx', at='end', value=0. ),
                    xt.Target('alfx', at='end', value=0. ),
                    xt.Target('alfy', at='end', value=0. ),
                    ],
            #verbose=True
            )
#Print quad strength after matching 
print('QUAD0', round(line_tr['quad_f'].knl[1], 3))
print('QUAD1', round(line_tr['quad_d'].knl[1], 3))
print('QUAD2', round(line_tr['quad_f_1'].knl[1], 3))
print('QUAD3', round(line_tr['quad_d_1'].knl[1], 3))
print('QUAD4', round(line_tr['quad_f_2'].knl[1], 3))
print('QUAD5', round(line_tr['quad_d_2'].knl[1], 3))

#Calculate twiss after matching
tw_match=line_tr.twiss(start=line_tr.element_names[0],end='_end_point', method='4d',
                    # ele_init= "drift_ini",
                    # **dict_init_FODO
                    init=tw_init_FODO
                    )
#Print optical functions at matching point
print('AFTER MATCHING, end of line ')
print('betx', tw_match['betx'][-1])
print('bety', tw_match['bety'][-1])
print('alfx', tw_match['alfx'][-1])
print('alfy', tw_match['alfy'][-1])
print('dx', tw_match['dx'][-1])
print('dpx', tw_match['dpx'][-1])
print('dy', tw_match['dy'][-1])
print('dpy', tw_match['dpy'][-1])

#%%Dispersion suppressor
# Dispersion suppressor lattice 
ad_2_elements=tr_elements[1:-1]
ad_2_names=tr_names[1:-1]

ad_1_elements= tr_elements[1:-1]
ad_1_elements=ad_1_elements[::-1]
ad_1_names= tr_names[1:-1]
ad_1_names=ad_1_names[::-1]

ds_elements=ad_1_elements+FODO_elements[1:]+FODO_elements*3+ad_2_elements+[end] #+FODO_elements[1:]+FODO_elements*1
ds_names=ad_1_names+FODO_names[1:]+FODO_names*3+ad_2_names+['end'] #+FODO_names[1:]+FODO_names*1

line_ds=xt.Line(
    elements=ds_elements,
    element_names=ds_names)
line_ds.particle_ref = xp.Particles(p0c=energy, #eV
                                     q0=1, mass0=xp.MUON_MASS_EV)
tw_ds_bf=line_ds.twiss( method='4d')

line_ds._init_var_management()
# #Set knobs to be varied
line_ds.vars['k_f0'] = line_ds['quad_f_2'].knl[1]
line_ds.vars['k_d1'] = line_ds['quad_d_2'].knl[1]
line_ds.vars['k_f1'] = line_ds['quad_f_1'].knl[1]
line_ds.vars['k_d2'] = line_ds['quad_d_1'].knl[1]
line_ds.vars['k_f2'] = line_ds['quad_f'].knl[1]
line_ds.vars['k_d3'] = line_ds['quad_d'].knl[1]
line_ds.vars['s_d'] = -1/(f*0.26)
line_ds.vars['s_f'] = 1/(f*0.5)

#Attach knobs to elements
line_ds.element_refs['quad_f_2'].knl[1] = line_ds.vars['k_f0']
line_ds.element_refs['quad_d_2'].knl[1] = line_ds.vars['k_d1']
line_ds.element_refs['quad_f_1'].knl[1] = line_ds.vars['k_f1']
line_ds.element_refs['quad_d_1'].knl[1] = line_ds.vars['k_d2']
line_ds.element_refs['quad_f'].knl[1] = line_ds.vars['k_f2']
line_ds.element_refs['quad_d'].knl[1] = line_ds.vars['k_d3']

line_ds.element_refs['quad_f_6'].knl[1] = line_ds.vars['k_f0']
line_ds.element_refs['quad_d_7'].knl[1] = line_ds.vars['k_d1']
line_ds.element_refs['quad_f_7'].knl[1] = line_ds.vars['k_f1']
line_ds.element_refs['quad_d_8'].knl[1] = line_ds.vars['k_d2']
line_ds.element_refs['quad_f_8'].knl[1] = line_ds.vars['k_f2']
line_ds.element_refs['quad_d_9'].knl[1] = line_ds.vars['k_d3']

line_ds.element_refs['sxt_d'].knl[2] = line_ds.vars['s_d']
line_ds.element_refs['sxt_f'].knl[2] = line_ds.vars['s_f']
line_ds.element_refs['sxt_d_1'].knl[2] = line_ds.vars['s_d']
line_ds.element_refs['sxt_f_1'].knl[2] = line_ds.vars['s_f']

print('SXT BEFORE MATCHING')
print('SXT F', line_ds['sxt_f_1'].knl[2])
print('SXT D', line_ds['sxt_d_1'].knl[2])

line_ds.match(vary=xt.VaryList([
                                'k_f0',
                                'k_d1',
                                'k_f1',
                                'k_d2',
                                'k_f2',
                                'k_d3',
                                's_d',
                                's_f'
                                ],
                      step=1e-8),
              targets=[
                    xt.Target(tar='dx', at='end', value=0. , tol=1e-12, weight=1),
                    xt.Target(tar='betx', at='quad_d_3', value=betx_fodo , tol=1e-12, weight=1),
                    xt.Target(tar='bety', at='quad_d_3', value=bety_fodo , tol=1e-12, weight=1),
                    xt.Target(tar='dx', at='quad_d_3', value=dx_fodo , tol=1e-12, weight=1),
                    xt.Target('qx', value=2.60 ),
                    xt.Target('qy', value=2.23 ),
                    xt.Target('dqx', value=0., tol=1e-3),
                    xt.Target('dqy', value=0.,tol=1e-3)
                    ],
            #verbose=True,
            #solve=False,
            method='4d'
            )
tw_ds=line_ds.twiss( method='4d')

print('SXT AFTER MATCHING')
print('SXT F', line_ds['sxt_f_1'].knl[2])
print('SXT D', line_ds['sxt_d_1'].knl[2])

print('GLOBAL PARAMETER')
print('MCF=', round(tw_ds['momentum_compaction_factor'], 6))
print('Qx=', round(tw_ds['qx'], 5))
print('Qy', round(tw_ds['qy'], 5))
print('dqx', round(tw_ds['dqx'], 2))
print('dqy', round(tw_ds['dqy'], 2))

beta_x_end=tw_ds['betx'][-1]
beta_y_end=tw_ds['bety'][-1]
alf_x_end=tw_ds['alfx'][-1]
alf_y_end=tw_ds['alfy'][-1]

#Import line to json
line_ds.to_json('lattice_disp_suppr.json')

#%% Plotting 
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
def plot_twiss(tw,line):
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)
    plt.plot(tw['s'], tw['betx'], label='betx')
    plt.plot(tw['s'], tw['bety'], label='bety')
    for i, el in enumerate(line.element_names):
        if 'quad' in el:
            plt.axvline(x=tw['s'][i], color='grey', linestyle='--')
        elif 'sxt' in el:
            plt.axvline(x=tw['s'][i], color='red', linestyle='--')
        elif 'cavity' in el:
            plt.axvline(x=tw['s'][i], color='green', linestyle='--')
    plt.xlabel('s [m]')
    plt.ylabel('betx, bety [m]')
    plt.legend(loc='upper right')
    plt.subplot(2, 1, 2)
    plt.plot(tw['s'], tw['dx'], label='Dx')
    plt.plot(tw['s'], tw['dy'], label='Dy')
    for i, el in enumerate(line.element_names):
        if 'quad' in el:
            plt.axvline(x=tw['s'][i], color='grey', linestyle='--')
        elif 'sxt' in el:
            plt.axvline(x=tw['s'][i], color='red', linestyle='--')
        elif 'cavity' in el:
            plt.axvline(x=tw['s'][i], color='green', linestyle='--')
    plt.xlabel('s [m]')
    plt.ylabel('Dx, Dy [m]')
    plt.legend(loc='upper right')

# plot_twiss(tw_FODO,line_FODO)
# plot_twiss(tw_match,line_tr)
plot_twiss(tw_ds,line_ds)

#%% Single plotting
def plot_twiss_single(tw):
    plt.figure()
    plt.plot(tw['s'],tw['betx'], label='betx')
    plt.plot(tw['s'],tw['bety'], label='bety')
    plt.xlabel('s [m]')
    plt.ylabel('betx, bety [m]')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(tw['s'],tw['dx'], label='Dx')
    plt.plot(tw['s'],tw['dy'], label='Dy')
    plt.xlabel('s [m]')
    plt.ylabel('Dx, Dy [m]')
    plt.legend()
    plt.show()

# plot_twiss_single(tw_FODO)
# plot_twiss_single(tw_tr)
# plot_twiss_single(tw_ds)

#%% Survey: checking the orbit closes 
# line=xt.Line(
#     elements=ds_elements*nb_arc,
#     element_names=ds_names*nb_arc)
# line.particle_ref = xp.Particles(p0c=energy, #eV
#                                       q0=1, mass0=xp.MUON_MASS_EV)
# survey=line.survey()
# print(survey)
# plt.figure()
# plt.plot(survey['Z'], survey['X'])
# plt.scatter(survey['Z'][-1], survey['X'][-1])
# plt.axis('equal')
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.show()
