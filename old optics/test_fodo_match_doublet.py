# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:10:16 2024
@author: LS276867

Old code
Test optics Dispersion Suppressor using a doublet of quad 
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

#Call class_geo
file_input='C:\muco\class_geometry\para_RCS_ME.txt'
RCS = Geometry(file_input) 

#Time and energy of study
time=RCS.phi_inj
energy=RCS.E_inj

#Get data from class_geo
Ldd=RCS.dipole_spacing 
Ls=RCS.LSSS #Length of short straight section
Lqd=RCS.QP_dipole_spacing 
list_hn=RCS.hn(time) #List of bendings
theta=RCS.theta(time) #List of bending angle 
L_dd_path=RCS.L_dd_path(time) 
L_dip=RCS.L_dip(time)
pattern=RCS.pattern 
nb_cell_tot=RCS.nb_cell_arcs

N_cell=1
#Define elements
mu=np.pi/2   #Phase advance 
#mu=2*np.pi*(0.25+0.22/nb_cell_tot)  
f=RCS.cell_length/(4*np.sin(mu/2)) #Focusing strength of quadrupole
drift_qd=xt.Drift(length=Lqd)
drift_dd=xt.Drift(length=Ldd)
drift_s=xt.Drift(length=Ls/2)
drift_r=xt.Drift(length=RCS.cell_length/2)
quad_f=xt.Multipole(knl=[0., 1/f], ksl=[0., 0.])
quad_d=xt.Multipole(knl=[0., -1/f], ksl=[0., 0.])
quad_d1=xt.Multipole(knl=[0., -1/f], ksl=[0., 0.])
quad_f=xt.Multipole(knl=[0., 1/f], ksl=[0., 0.])
quad_f0=xt.Multipole(knl=[0., 1/f], ksl=[0., 0.])
quad_f1=xt.Multipole(knl=[0., 1/f], ksl=[0., 0.])
quad_d2=xt.Multipole(knl=[0., -1/f], ksl=[0., 0.])
quad_f2=xt.Multipole(knl=[0., 1/f], ksl=[0., 0.])
quad_d3=xt.Multipole(knl=[0., -1/f], ksl=[0., 0.])

beg = xt.Marker()
end = xt.Marker()

#Define line from pattern
# cell=[drift_qd]
# cell1_names=['c1_drift_beg']
# for i,key in enumerate (pattern):
#     cell.append(xt.Multipole(length=RCS.dipole_families[key]['length'], knl=[theta[i]], ksl=[0.0], hxl=theta[i], hyl=0.))
#     cell.append(xt.Drift(length=L_dip[i]))
#     cell1_names.append(f"c1_dip_{i+1}")
#     cell1_names.append(f"c1_drdp_{i+1}")
#     if i != len(pattern)-1:
#         cell.append(xt.Drift(length=L_dd_path[i]))
#         cell1_names.append(f"c1_drift_{i+1}")
#     else: 
#         cell.append(drift_qd)
#         cell1_names.append("c1_drift_end")
# cell2_names = [name.replace('c1', 'c2') for name in cell1_names]
# elements=[quad_f,drift_s]+cell+[quad_d,drift_s]+cell
# element_names=['quad_f','drift_qf']+cell1_names+['quad_d','drift_qd']+cell2_names

# line=xt.Line(elements=elements, element_names=element_names)

# line.build_tracker()

# line=xt.Line(
#     elements=[drift_qd,BNC,drift_qd],
#     element_names=['drift_beg','dip1','drift_end'])

#Define line manually
length_nc=RCS.dipole_families['BNC']['length']
i_BNC=pattern.index('BNC')
BNC=xt.Multipole(length=length_nc, 
                 knl=[theta[i_BNC]], 
                 ksl=[0.0], hxl=theta[i_BNC], hyl=0. )  
drift_BNC=xt.Drift(length=length_nc/2)
if len(set(pattern))>1:
    length_sc=RCS.dipole_families['BSC']['length']
    i_BSC=pattern.index('BSC')
    BSC=xt.Multipole(length=length_sc,
                     knl=[theta[i_BSC]], 
                     ksl=[0.0], hxl=theta[i_BSC], hyl=0. ) #,  
    drift_BSC=xt.Drift(length=length_sc/2)

#%% Optics
#FODO cell
cell=[drift_qd, drift_BSC, BSC, drift_BSC, drift_dd, drift_BNC, BNC, drift_BNC, drift_dd,drift_BSC, BSC, drift_BSC, drift_qd]
FODO_elements=[quad_f,drift_s]+cell+[drift_s, quad_d,drift_s]+cell+[drift_s]
FODO_names=['quad_f','drift_qf2','c1_drift_beg','c1_drdp1', 'c1_dip_1', 'c1_drdp2', 'c1_drift_1', 
                'c1_drdp3', 'c1_dip_2', 'c1_drdp4','c1_drift_2','c1_drdp5','c1_dip_3', 'c1_drdp6','c1_drift_end',
                'drift_qd1','quad_d','drift_qd2','c2_drift_beg', 'c2_drdp1', 'c2_dip_1','c2_drdp2', 'c2_drift_1', 
                'c2_drdp3','c2_dip_2', 'c2_drdp4', 'c2_drift_2', 'c2_drdp5','c2_dip_3','c2_drdp6',  'c2_drift_end','drift_qf1']
line_FODO=xt.Line(
    elements=FODO_elements*N_cell,
    element_names=FODO_names*N_cell)
line_FODO.particle_ref = xp.Particles(p0c=energy, #eV
                                      q0=1, mass0=xp.MUON_MASS_EV)

#at_s=np.linspace(0.1,RCS.cell_length-0.1,100)
tw_FODO = line_FODO.twiss(method='4d') 

#Transfer
tr_names=['beg']+FODO_names+FODO_names+['quad_f','drift_qd','quad_d','drift_end','end'] #['drift_ini']
tr_elements= [beg]+[quad_f,drift_s]+cell+[drift_s, quad_d1,drift_s]+cell+[drift_s]+[
    quad_f1,drift_s]+cell+[drift_s, quad_d2,drift_s]+cell+[drift_s]+[quad_f2,drift_qd,
                                                                           quad_d3,drift_r]+[end] #[drift_r]+

line_tr=xt.Line(
    elements=tr_elements,
    element_names=tr_names)
line_tr.particle_ref = xp.Particles(p0c=energy, #eV
                                      q0=1, mass0=xp.MUON_MASS_EV)

#Matches FODO lattices to the transfer line: on first quad of the transfer == output optics of FODO
tw_init_FODO = tw_FODO.get_twiss_init(0)
tw_init_FODO.element_name = 'beg'
tw_tr=line_tr.twiss(ele_start=0,ele_stop='_end_point', method='4d',
                    # ele_init= "drift_ini",
                    twiss_init=tw_init_FODO
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
line_tr.vars['k_d3'] = -1/f

line_tr.element_refs['quad_f'].knl[1] = line_tr.vars['k_f0']
line_tr.element_refs['quad_d'].knl[1] = line_tr.vars['k_d1']
line_tr.element_refs['quad_f_1'].knl[1] = line_tr.vars['k_f1']
line_tr.element_refs['quad_d_1'].knl[1] = line_tr.vars['k_d2']
line_tr.element_refs['quad_f_2'].knl[1] = line_tr.vars['k_f2']
line_tr.element_refs['quad_d_2'].knl[1] = line_tr.vars['k_d3']

#Matching
line_tr.match(
              ele_start='beg',
              ele_stop='_end_point', 
              twiss_init=tw_init_FODO,
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
tw_match=line_tr.twiss(ele_start=0,ele_stop='_end_point', method='4d',
                    # ele_init= "drift_ini",
                    # **dict_init_FODO
                    twiss_init=tw_init_FODO
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
ad_1_elements=tr_elements[2:-1]
ad_1_elements=ad_1_elements[::-1]
ad_2_elements= tr_elements[1:]
ad_2_names=FODO_names+FODO_names+['quad_f','drift_qd','quad_d','drift_end','end']
ad_1_names=['drift_beg','quad_d','drift_qd']+FODO_names+FODO_names

ds_elements=ad_1_elements+FODO_elements+ad_2_elements
ds_names=ad_1_names+FODO_names+ad_2_names

line_ds=xt.Line(
    elements=ds_elements,
    element_names=ds_names)
line_ds.particle_ref = xp.Particles(p0c=energy, #eV
                                     q0=1, mass0=xp.MUON_MASS_EV)

#Set knobs to be varied
line_ds._init_var_management()
line_ds.vars['k_f0_d'] = line_ds['quad_f_3'].knl[1]
line_ds.vars['k_d1_d'] = line_ds['quad_d_4'].knl[1]
line_ds.vars['k_f1_d'] = line_ds['quad_f_4'].knl[1]
line_ds.vars['k_d2_d'] = line_ds['quad_d_5'].knl[1]
line_ds.vars['k_f2_d'] = line_ds['quad_f_5'].knl[1]
line_ds.vars['k_d3_d'] = line_ds['quad_d_6'].knl[1]

line_ds.element_refs['quad_f_3'].knl[1] = line_ds.vars['k_f0_d']
line_ds.element_refs['quad_d_4'].knl[1] = line_ds.vars['k_d1_d']
line_ds.element_refs['quad_f_4'].knl[1] = line_ds.vars['k_f1_d']
line_ds.element_refs['quad_d_5'].knl[1] = line_ds.vars['k_d2_d']
line_ds.element_refs['quad_f_5'].knl[1] = line_ds.vars['k_f2_d']
line_ds.element_refs['quad_d_6'].knl[1] = line_ds.vars['k_d3_d']

line_ds.match(vary=xt.VaryList([
                                'k_f0_d',
                                'k_d1_d',
                                'k_f1_d',
                                'k_d2_d',
                                'k_f2_d',
                                'k_d3_d'
                                ],
                      step=1e-8),
              targets=[
                    xt.Target('dx', at='end', value=0. ),
                    xt.Target('dpx', at='end', value=0. ),
                    xt.Target('alfx', at='end', value=0. ),
                    xt.Target('alfy', at='end', value=0. ),
                    #xt.Target('qx', value=1.45 ),
                    #xt.Target('qy', value=1.1 ),
                    ],
            verbose=True,
            method='4d'
            )

tw_ds=line_ds.twiss(method='4d')

print('GLOBAL PARAMETER')
print('MCF=', round(tw_ds['momentum_compaction_factor'], 5))
print('Qx=', round(tw_ds['qx'], 2))
print('Qy', round(tw_ds['qy'], 2))
print('chromaticity', round(tw_ds['dqx'], 2))


#%% FODO parameter calculation (see code calcul_fodo_para.py)
# Lp=RCS.cell_length
# theta=RCS.cell_angle
# beta_max=Lp*(1+np.sin(mu/2))/np.sin(mu)
# beta_min=Lp*(1-np.sin(mu/2))/np.sin(mu)
# alpha_max= (-1-np.sin(mu/2))/np.cos(mu/2)
# alpha_min= ( 1-np.sin(mu/2))/np.cos(mu/2)
# D_max=Lp*theta*(1+np.sin(mu/2)/2)/(4*(np.sin(mu/2))**2)
# D_min=Lp*theta*(1-np.sin(mu/2)/2)/(4*(np.sin(mu/2))**2)
# D_av= Lp*theta/4*( 1/(np.sin(mu/2))**2 - 1/12)
# mcf=theta*D_av/Lp
# chromaticity=-1/np.pi*np.tan(mu/2)
# print('Theoretical and calculated FODO parameters')
# print(f'beta_max = {beta_max:.3f}  beta_max_xs = {np.max(tw["betx"]):.3f}')
# print(f'beta_min = {beta_min:.3f}  beta_min_xs = {np.min(tw["betx"]):.3f}')
# print(f'alpha_max = {alpha_max:.3f}  alpha_max_xs = {np.max(tw["alfx"]):.3f}')
# print(f'alpha_min = {alpha_min:.3f}  alpha_min_xs = {np.min(tw["alfx"]):.3f}')
# print(f'D_max = {D_max:.3f}  D_max_xs = {np.max(tw["dx"]):.3f}')
# print(f'D_min = {D_min:.3f}  D_min_xs = {np.min(tw["dx"]):.3f}')
# print(f'D_av = {D_av:.3f}  D_av_xs = {np.mean(tw["dx"]):.3f}')
# print(f'MCF_th = {mcf:.5f}  MCF_xs = {tw["momentum_compaction_factor"]:.5f}')
# print(f'chromaticity_th = {chromaticity:.3f}  chromaticity_xs = {tw["dqx"]:.3f}')

#%% Plotting 
#FODO
plt.figure(figsize=(12, 12))
#plt.title('FODO')
plt.subplot(3, 1, 1)
plt.plot(tw_FODO['s'], tw_FODO['betx'], label='betx')
plt.plot(tw_FODO['s'], tw_FODO['bety'], label='bety')
for i, el in enumerate(line_FODO.element_names):
    if 'quad' in el:
        plt.axvline(x=tw_FODO['s'][i], color='grey', linestyle='--')
plt.xlabel('s [m]')
plt.ylabel('betx, bety [m]')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(tw_FODO['s'], tw_FODO['dx'], label='Dx')
plt.plot(tw_FODO['s'], tw_FODO['dy'], label='Dy')
for i, el in enumerate(line_FODO.element_names):
    if 'quad' in el:
        plt.axvline(x=tw_FODO['s'][i], color='grey', linestyle='--')
plt.xlabel('s [m]')
plt.ylabel('Dx, Dy [m]')
plt.legend()

plt.subplot(3, 1, 3)
plt.scatter(tw_FODO['s'], tw_FODO['alfx'], label='alfx')
plt.scatter(tw_FODO['s'], tw_FODO['alfy'], label='alfy')
for i, el in enumerate(line_FODO.element_names):
    if 'quad' in el:
        plt.axvline(x=tw_FODO['s'][i], color='grey', linestyle='--')
plt.xlabel('s [m]')
plt.ylabel('alfx, alfy')
plt.legend()
plt.tight_layout()  
plt.show()

##Transfer Line, before matching
# plt.figure(figsize=(12, 12))
# #plt.title('Transfer, before matching')
# plt.subplot(3, 1, 1)
# plt.plot(tw_tr['s'], tw_tr['betx'], label='betx')
# plt.plot(tw_tr['s'], tw_tr['bety'], label='bety')
# for i, el in enumerate(line_tr.element_names):
#     if 'quad' in el:
#         plt.axvline(x=tw_tr['s'][i], color='grey', linestyle='--')
# plt.xlabel('s [m]')
# plt.ylabel('betx, bety [m]')
# plt.legend()

# plt.subplot(3, 1, 2)
# plt.plot(tw_tr['s'], tw_tr['dx'], label='Dx')
# plt.plot(tw_tr['s'], tw_tr['dy'], label='Dy')
# for i, el in enumerate(line_tr.element_names):
#     if 'quad' in el:
#         plt.axvline(x=tw_tr['s'][i], color='grey', linestyle='--')
# plt.xlabel('s [m]')
# plt.ylabel('Dx, Dy [m]')
# plt.legend()

# plt.subplot(3, 1, 3)
# plt.scatter(tw_tr['s'], tw_tr['alfx'], label='alfx')
# plt.scatter(tw_tr['s'], tw_tr['alfy'], label='alfy')
# for i, el in enumerate(line_tr.element_names):
#     if 'quad' in el:
#         plt.axvline(x=tw_tr['s'][i], color='grey', linestyle='--')
# plt.xlabel('s [m]')
# plt.ylabel('alfx, alfy')
# plt.legend()
# plt.tight_layout()  
# plt.show()

#Transfer line, after matching
plt.figure(figsize=(12, 12))
#plt.title('Transfer, after matching')
plt.subplot(3, 1, 1)
plt.plot(tw_match['s'], tw_match['betx'], label='betx')
plt.plot(tw_match['s'], tw_match['bety'], label='bety')
for i, el in enumerate(line_tr.element_names):
    if 'quad' in el:
        plt.axvline(x=tw_match['s'][i], color='grey', linestyle='--')
plt.xlabel('s [m]')
plt.ylabel('betx, bety [m]')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(tw_match['s'], tw_match['dx'], label='Dx')
plt.plot(tw_match['s'], tw_match['dy'], label='Dy')
for i, el in enumerate(line_tr.element_names):
    if 'quad' in el:
        plt.axvline(x=tw_match['s'][i], color='grey', linestyle='--')
plt.xlabel('s [m]')
plt.ylabel('Dx, Dy [m]')
plt.legend()

plt.subplot(3, 1, 3)
plt.scatter(tw_match['s'], tw_match['alfx'], label='alfx')
plt.scatter(tw_match['s'], tw_match['alfy'], label='alfy')
for i, el in enumerate(line_tr.element_names):
    if 'quad' in el:
        plt.axvline(x=tw_match['s'][i], color='grey', linestyle='--')
plt.xlabel('s [m]')
plt.ylabel('alfx, alfy')
plt.legend()
plt.tight_layout()  
plt.show()

#Dispersion suppressor
plt.figure(figsize=(12, 12))
#plt.title('Transfer, after matching')
plt.subplot(3, 1, 1)
plt.plot(tw_ds['s'], tw_ds['betx'], label='betx')
plt.plot(tw_ds['s'], tw_ds['bety'], label='bety')
for i, el in enumerate(line_ds.element_names):
    if 'quad' in el:
        plt.axvline(x=tw_ds['s'][i], color='grey', linestyle='--')
plt.xlabel('s [m]')
plt.ylabel('betx, bety [m]')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(tw_ds['s'], tw_ds['dx'], label='Dx')
plt.plot(tw_ds['s'], tw_ds['dy'], label='Dy')
for i, el in enumerate(line_ds.element_names):
    if 'quad' in el:
        plt.axvline(x=tw_ds['s'][i], color='grey', linestyle='--')
plt.xlabel('s [m]')
plt.ylabel('Dx, Dy [m]')
plt.legend()

plt.subplot(3, 1, 3)
plt.scatter(tw_ds['s'], tw_ds['alfx'], label='alfx')
plt.scatter(tw_ds['s'], tw_ds['alfy'], label='alfy')
for i, el in enumerate(line_ds.element_names):
    if 'quad' in el:
        plt.axvline(x=tw_ds['s'][i], color='grey', linestyle='--')
plt.xlabel('s [m]')
plt.ylabel('alfx, alfy')
plt.legend()
plt.tight_layout()  
plt.show()

#%% Single plotting

#FODO
# plt.figure()
# plt.plot(tw_FODO['s'],tw_FODO['betx'], label='betx')
# plt.plot(tw_FODO['s'],tw_FODO['bety'], label='bety')
# for i, el in enumerate(line_FODO.element_names):
#     if 'quad' in el:
#         plt.axvline(x=tw_FODO['s'][i], color='grey', linestyle='--')
# plt.xlabel('s [m]')
# plt.ylabel('betx, bety [m]')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(tw_FODO['s'],tw_FODO['dx'], label='Dx')
# plt.plot(tw_FODO['s'],tw_FODO['dy'], label='Dy')
# for i, el in enumerate(line_FODO.element_names):
#     if 'quad' in el:
#         plt.axvline(x=tw_FODO['s'][i], color='grey', linestyle='--')
# plt.xlabel('s [m]')
# plt.ylabel('Dx, Dy [m]')
# plt.legend()
# plt.show()

# plt.figure()
# plt.scatter(tw_FODO['s'],tw_FODO['alfx'], label='alfx')
# plt.scatter(tw_FODO['s'],tw_FODO['alfy'], label='alfy')
# for i, el in enumerate(line_FODO.element_names):
#     if 'quad' in el:
#         plt.axvline(x=tw_FODO['s'][i], color='grey', linestyle='--')
# plt.xlabel('s [m]')
# plt.ylabel('alfx, alfy ')
# plt.legend()
# plt.show()

# #BEFORE MATCHING
# plt.figure()
# plt.plot(tw_tr['s'],tw_tr['betx'], label='betx')
# plt.plot(tw_tr['s'],tw_tr['bety'], label='bety')
# for i, el in enumerate(line_tr.element_names):
#     if 'quad' in el:
#         plt.axvline(x=tw_tr['s'][i], color='grey', linestyle='--')
# plt.xlabel('s [m]')
# plt.ylabel('betx, bety [m]')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(tw_tr['s'],tw_tr['dx'], label='Dx')
# plt.plot(tw_tr['s'],tw_tr['dy'], label='Dy')
# for i, el in enumerate(line_tr.element_names):
#     if 'quad' in el:
#         plt.axvline(x=tw_tr['s'][i], color='grey', linestyle='--')
# plt.xlabel('s [m]')
# plt.ylabel('Dx, Dy [m]')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(tw_tr['s'],tw_tr['alfx'], label='alfx')
# plt.plot(tw_tr['s'],tw_tr['alfy'], label='alfy')
# # plt.axhline(y=alpha_max, color='grey', linestyle='--')
# # plt.axhline(y=alpha_min, color='grey', linestyle='--')
# plt.xlabel('s [m]')
# plt.ylabel('alfx, alfy ')
# plt.legend()
# plt.show()

# #MATCHING
# plt.figure()
# plt.plot(tw_match['s'],tw_match['betx'], label='betx')
# plt.plot(tw_tr['s'],tw_tr['bety'], label='bety')
# for i, el in enumerate(line_tr.element_names):
#     if 'quad' in el:
#         plt.axvline(x=tw_tr['s'][i], color='grey', linestyle='--')
# plt.xlabel('s [m]')
# plt.ylabel('betx, bety [m]')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(tw_match['s'],tw_match['dx'], label='Dx')
# plt.plot(tw_tr['s'],tw_tr['dy'], label='Dy')
# for i, el in enumerate(line_tr.element_names):
#     if 'quad' in el:
#         plt.axvline(x=tw_tr['s'][i], color='grey', linestyle='--')
# plt.xlabel('s [m]')
# plt.ylabel('Dx, Dy [m]')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(tw_match['s'],tw_match['alfx'], label='alfx')
# plt.plot(tw_match['s'],tw_match['alfy'], label='alfy')
# plt.xlabel('s [m]')
# plt.ylabel('alfx, alfy ')
# plt.legend()
# plt.show()
#%% Survey: checking the orbit closes 
# line=xt.Line(
#     elements=elements*nb_cell_tot,
#     element_names=element_names*nb_cell_tot)
# line.particle_ref = xp.Particles(p0c=energy, #eV
#                                       q0=1, mass0=xp.MUON_MASS_EV)
# survey=line.survey()
# print(survey)
# plt.figure()
# plt.plot(survey['Z'], survey['X'])
# plt.axis('equal')
# # for i, el in enumerate(line.element_names):
# #     if 'dip' in el:
# #         plt.axvline(x=survey['Z'][i], color='grey', linestyle='--')
# #     elif 'quad' in el:
# #         plt.axvline(x=survey['Z'][i], color='red', linestyle='--')
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.show()

