# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:10:16 2024

@author: LS276867
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt
import xpart as xp
import sys
sys.path.append('/mnt/c/muco/mhega/rcsparameters/rcsparameters')
sys.path.append('/mnt/c/muco/code')
from geometry.geometry import Geometry 
from track_function import track
from optics_function import plot_twiss, plot_twiss_single
import json

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

file_input='/mnt/c/muco/code/class_geometry/parameter_files/para_RCS_ME.txt'
time_frac=0
n_slice=0
option='var_ref'

RCS = Geometry(file_input)
print('Input file', file_input)
time=time_frac
t_ref=0.5
energy=RCS.E_inj+(RCS.E_ext-RCS.E_inj)*time_frac
frequency_rf=1300e6
print('Time', time)
print('Energy', energy)

#Get data from class_geo
Ldd=RCS.dipole_spacing 
Ls=RCS.LSSS #Length of short straight section
# Lqd=RCS.QP_dipole_spacing
Lqd=RCS.L_qp_dip_path(time)
Lins=RCS.insertion_length
Lcell=RCS.cell_length
list_hn=RCS.hn(time) #List of bendings
list_hn_ref=RCS.hn(t_ref) #List of reference bendings
theta=RCS.theta(time) #List of bending angle 
epsilon=RCS.epsilon(time)
L_dd_path=RCS.L_dd_path(time) 
L_dip_path=RCS.L_dip_path(time)
L_dip_path_ref=RCS.L_dip_path(t_ref)
theta_ref=RCS.theta(t_ref)
pattern=RCS.pattern 
nb_cell_tot=RCS.nb_cell_rcs
nb_arc=RCS.nb_arc
n_cav_arc=2

#RF input data
N_turn_rcs=55
N_frac=1
n_turns = N_turn_rcs*nb_arc*N_frac
energy_increment_per_arc = (RCS.E_ext-RCS.E_inj)/n_turns
sync_phase=45
phase=180-sync_phase
volt_arc=energy_increment_per_arc/np.sin(phase*np.pi/180)
volt_cav=volt_arc/n_cav_arc

#Define elements: drifts, quads, sextupoles
mu=np.pi/2 #Phase advance 
f=RCS.cell_length/(4*np.sin(mu/2)) #Focusing strength of quadrupole
drift_qd=xt.Drift(length=Lqd)
drift_dd=xt.Drift(length=L_dd_path[1])
drift_s=xt.Drift(length=Ls/2)
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
s_d=-s_f
sxt_d=xt.Multipole(knl=[0., 0.,s_d], ksl=[0., 0.,0.])
sxt_f=xt.Multipole(knl=[0., 0.,s_f], ksl=[0., 0.,0.])

beg = xt.Marker()
end = xt.Marker()

#Step to match
step_quad=1e-8
step_sxt=1e-5

#Cavity settings
V=0
freq=0
phi=0
cavity=xt.Cavity(voltage=V, frequency=freq, lag=phi)
RF_acc=xt.ReferenceEnergyIncrease(Delta_p0c=0)
dz_acc=xt.ZetaShift(dzeta=0)

#Goal tune and chroma
qx_goal=2.6067
qy_goal=2.252
dqx_goal_arc=5/nb_arc
dqy_goal_arc=5/nb_arc


#Define dipoles 
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

# Define edges
eps=RCS.epsilon(t_ref)
hn=list_hn
model='full'
en1=xt.DipoleEdge(k=hn[0], e1=-eps[0], side='entry',model = model)
ex1=xt.DipoleEdge(k=hn[0], e1=eps[1], side='exit',model =model)
en2=xt.DipoleEdge(k=hn[1], e1=-eps[1], side='entry',model =model)
ex2=xt.DipoleEdge(k=hn[1], e1=eps[2], side='exit',model =model)
en3=xt.DipoleEdge(k=hn[2], e1=-eps[2], side='entry',model =model)
ex3=xt.DipoleEdge(k=hn[2], e1=eps[3], side='exit',model =model)

cell=[drift_qd]+[en1,BSC,ex1]+ [drift_dd]+ [en2, BNC,ex2]+ [drift_dd]+[en3, BSC,ex3]+[drift_qd]
# cell=[drift_qd]+[BSC]+ [drift_dd]+ [ BNC]+ [drift_dd]+[ BSC]+[drift_qd] #without edeges
cell_name=['drift_qd']+['en1','BSC','ex1']+ ['drift_dd']+ ['en2', 'BNC','ex2']+ ['drift_dd']+['en3','BSC','ex3']+['drift_qd']
# cell_name=['drift_qd']+['BSC']+ ['drift_dd']+ [ 'BNC']+ ['drift_dd']+['BSC']+['drift_qd'] #without edeges

FODO_elements=[quad_f,drift_s]+cell+[drift_s, quad_d,drift_s]+cell+[drift_s]
FODO_names=['quad_f','drift_s']+cell_name+['drift_s', 'quad_d','drift_s']+cell_name+['drift_s']

line_cell=xt.Line(
    elements=cell,
    element_names=cell_name)
line_cell.particle_ref = xp.Particles(p0c=energy, #eV
                                    q0=1, mass0=xp.MUON_MASS_EV)
if n_slice > 0:
    line_cell.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Teapot(n_slice))])

line_FODO=xt.Line(
    elements=FODO_elements,
    element_names=FODO_names)
line_FODO.particle_ref = xp.Particles(p0c=energy, #eV
                                    q0=1, mass0=xp.MUON_MASS_EV)

tw_FODO = line_FODO.twiss(method='4d') 
betx_qd_fodo=tw_FODO['betx', 'quad_d']
bety_qd_fodo=tw_FODO['bety', 'quad_d'] 
dx_qd_fodo=tw_FODO['dx', 'quad_d']  
betx_qf_fodo=tw_FODO['betx', 'quad_f']
bety_qf_fodo=tw_FODO['bety', 'quad_f'] 
dx_qf_fodo=tw_FODO['dx', 'quad_f'] 

#Transfer
tr_elements= [beg]+[quad_f1 ,drift_s]+cell+[drift_s, quad_d1,drift_s]+cell+[drift_s,quad_f2,drift_s
            ]+cell+[drift_s, quad_d2, drift_s]+cell+[drift_s, quad_f3,drift_r, cavity, drift_r, quad_d3]+[end]
tr_names=['beg']+['quad_f', 'drift_s']+cell_name+['drift_s', 'quad_d','drift_s']+cell_name+['drift_s',
        'quad_f','drift_s']+cell_name+['drift_s', 'quad_d', 'drift_s']+cell_name+['drift_s', 'quad_f',
                    'drift_r','cavity','drift_r', 'quad_d']+['end']

line_tr=xt.Line(
    elements=tr_elements,
    element_names=tr_names)
line_tr.particle_ref = xp.Particles(p0c=energy, #eV
                                    q0=1, mass0=xp.MUON_MASS_EV)

#Matches FODO lattices to the transfer line: on first quad of the transfer == output optics of FODO
tw_init_FODO = tw_FODO.get_twiss_init(0)
tw_init_FODO.element_name = 'beg'
tw_tr=line_tr.twiss(start=line_tr.element_names[0],end='_end_point', method='4d',
                    init=tw_init_FODO
                    )

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

# Matching
match_tr=line_tr.match(
            start='beg',
            end='_end_point', 
            init=tw_init_FODO,
            vary=xt.VaryList(['k_f0','k_d1','k_f1','k_d2','k_f2','k_d3'],
                            step=step_quad,
                            tag='quad'),
            targets=[xt.TargetSet(dx=0,dpx=0,alfx=0,alfy=0, at='end',tol=1e-9)],
            # verbose=True
            )
print('RESULTS FROM TR MATCHING')
match_tr.target_status()
match_tr.vary_status()

#Calculate twiss after matching
tw_match=line_tr.twiss(start=line_tr.element_names[0],end='_end_point', method='4d',
                    init=tw_init_FODO
                    )
# plot_twiss_single(tw_match)

# Dispersion suppressor lattice 
tr_elements_inv=[end]+[quad_d3,drift_r,cavity,drift_r,quad_f3,drift_s]+cell+[drift_s,quad_d2,drift_s
                ]+cell+[drift_s,quad_f2,drift_s]+cell+[drift_s,quad_d1,drift_s]+cell+[drift_s,quad_f1]+[beg]
tr_names_inv=['end']+['quad_d','drift_r','cavity','drift_r','quad_f','drift_s']+cell_name+[
    'drift_s','quad_d','drift_s']+cell_name+['drift_s','quad_f','drift_s'
    ]+cell_name+['drift_s','quad_d','drift_s']+cell_name+['drift_s','quad_f']+['beg']
ad_2_elements=tr_elements[1:-1]
ad_2_names=tr_names[1:-1]
ad_1_elements= tr_elements_inv[1:-1]
ad_1_names= tr_names_inv[1:-1]

ds_elements=ad_1_elements+FODO_elements[1:]+FODO_elements*3+ad_2_elements+[end] 
ds_names=ad_1_names+FODO_names[1:]+FODO_names*3+ad_2_names+['end'] 

line_ds=xt.Line(
    elements=ds_elements,
    element_names=ds_names)

tab_bf=line_ds.get_table()
for el in tab_bf.rows['quad.*'].name[3:-3]:
    if 'quad_d_' in el:
        line_ds.insert_element('sxt_d_'+el[-1],sxt_d,at_s=tab_bf['s', el])
    if 'quad_f' in el:
        line_ds.insert_element('sxt_f_'+el[-1],sxt_f,at_s=tab_bf['s', el])
# line_ds.insert_element('sxt_d_1',sxt_d,at_s=tab_bf['s','quad_d_1'])
# line_ds.insert_element('sxt_d_2',sxt_d,at_s=tab_bf['s','quad_d_3'])
# line_ds.insert_element('sxt_d_3',sxt_d,at_s=tab_bf['s','quad_d_6'])
# line_ds.insert_element('sxt_d_4',sxt_d,at_s=tab_bf['s','quad_d_8'])
# line_ds.insert_element('sxt_f_1',sxt_f,at_s=tab_bf['s','quad_f_1'])
# line_ds.insert_element('sxt_f_2',sxt_f,at_s=tab_bf['s','quad_f_2'])
# line_ds.insert_element('sxt_f_3',sxt_f,at_s=tab_bf['s','quad_f_3'])
# line_ds.insert_element('sxt_f_4',sxt_f,at_s=tab_bf['s','quad_f_4'])
line_ds.insert_element('RF_acc', RF_acc, at=2)
line_ds.insert_element('RF_acc_1', RF_acc, at=-4)
line_ds.insert_element('dz_acc', dz_acc, at=3)
line_ds.insert_element('dz_acc_1', dz_acc, at=-4)

line_ds.particle_ref = xp.Particles(p0c=energy, #eV
                                    q0=1, mass0=xp.MUON_MASS_EV)
if n_slice > 0:
    line_ds.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Teapot(n_slice))])

# tw_ds_bf=line_ds.twiss( method='4d')

line_ds._init_var_management()
#Set knobs to be varied
line_ds.vars['k_f0'] = line_ds['quad_f_2'].knl[1]
line_ds.vars['k_d1'] = line_ds['quad_d_2'].knl[1]
line_ds.vars['k_f1'] = line_ds['quad_f_1'].knl[1]
line_ds.vars['k_d2'] = line_ds['quad_d_1'].knl[1]
line_ds.vars['k_f2'] = line_ds['quad_f'].knl[1]
line_ds.vars['k_d3'] = line_ds['quad_d'].knl[1]
# line_ds.vars['s_d'] = -1/(f*0.5)*8
# line_ds.vars['s_f'] = 1/(f*0.25)*8

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

tab_sxt=line_ds.get_table()
list_sd=tab_sxt.rows['sxt_d.*'].name
list_sf=tab_sxt.rows['sxt_f.*'].name
list_s=['s_d','s_d_1','s_f','s_f_1']
# list_s=[]
line_ds.vars['s_d']=-1/(f*0.25)
line_ds.vars['s_d_1']=-1/(f*0.25)
line_ds.vars['s_f']=1/(f*0.5)
line_ds.vars['s_f_1']=1/(f*0.5)

# for i,sxt in enumerate(list_sd[0:math.ceil(len(list_sd)/2)]):
#         line_ds.vars['s_'+sxt[-3:]]=-1/(f*0.5)
#         line_ds.element_refs[sxt].knl[2] = line_ds.vars['s_'+sxt[-3:]]
#         line_ds.element_refs[list_sd[-i-1]].knl[2] = line_ds.vars['s_'+sxt[-3:]]
#         list_s.append('s_'+sxt[-3:])
# for i,sxt in enumerate(list_sf[0:math.ceil(len(list_sf)/2)]):
#         line_ds.vars['s_'+sxt[-3:]]=1/(f*0.25)
#         line_ds.element_refs[sxt].knl[2] = line_ds.vars['s_'+sxt[-3:]]
#         line_ds.element_refs[list_sf[-i-1]].knl[2] = line_ds.vars['s_'+sxt[-3:]]
#         list_s.append('s_'+sxt[-3:])

line_ds.element_refs['sxt_d_2'].knl[2] = line_ds.vars['s_d']
line_ds.element_refs['sxt_d_3'].knl[2] = line_ds.vars['s_d']
line_ds.element_refs['sxt_d_4'].knl[2] = line_ds.vars['s_d_1']
line_ds.element_refs['sxt_d_5'].knl[2] = line_ds.vars['s_d_1']
line_ds.element_refs['sxt_d_6'].knl[2] = line_ds.vars['s_d']
line_ds.element_refs['sxt_d_7'].knl[2] = line_ds.vars['s_d']

line_ds.element_refs['sxt_f_1'].knl[2] = line_ds.vars['s_f']
line_ds.element_refs['sxt_f_2'].knl[2] = line_ds.vars['s_f']
line_ds.element_refs['sxt_f_3'].knl[2] = line_ds.vars['s_f']
line_ds.element_refs['sxt_f_4'].knl[2] = line_ds.vars['s_f_1']
line_ds.element_refs['sxt_f_5'].knl[2] = line_ds.vars['s_f']
line_ds.element_refs['sxt_f_6'].knl[2] = line_ds.vars['s_f']
line_ds.element_refs['sxt_f_7'].knl[2] = line_ds.vars['s_f']

# match_ds_4d=line_ds.match(vary=xt.VaryList(['k_f0','k_d1','k_f1','k_d2','k_f2','k_d3'],
#                             step=step_quad,
#                             tag='quad'),
#             targets=[
#                     xt.Target(tar='dx', at='end', value=0., tol=1e-9,tag='DS'),
#                     xt.TargetSet(betx=betx_qd_fodo,bety=bety_qd_fodo, at='quad_d_3', tol=1e-9, tag='FODO'),
#                     xt.TargetSet(qx=2.6067, qy=2.252, tol=1e-3, tag='tune')
#                     ],
#             # solve=False,
#             method='4d',
#             # verbose=True
#             )

# print('MATCHING QUAD DS 4D')
# match_ds_4d.target_status()
# match_ds_4d.vary_status()
tw_ds_4d=line_ds.twiss( method='4d')

#Import line to json
# line_ds.to_json('lattice_disp_suppr.json') 

# Cavity settings
frequency = np.round(frequency_rf*tw_ds_4d.T_rev0)/tw_ds_4d.T_rev0
line_ds['cavity'].frequency=frequency
line_ds['cavity'].lag=phase
line_ds['cavity'].voltage=volt_cav
line_ds['cavity_1'].frequency=frequency
line_ds['cavity_1'].lag=phase
line_ds['cavity_1'].voltage=volt_cav
tab = line_ds.get_table()

h_rf = np.round(frequency_rf*tw_ds_4d.T_rev0*nb_arc) #data tw_ds on 1 arc
Delta_p0c=energy_increment_per_arc/n_cav_arc
line_ds.discard_tracker()
line_ds['RF_acc'].Delta_p0c=Delta_p0c 
line_ds['RF_acc_1'].Delta_p0c=Delta_p0c 
line_ds.build_tracker()

match_ds_6d=line_ds.match(vary=xt.VaryList(['k_f0','k_d1','k_f1','k_d2','k_f2','k_d3'],
                            step=step_quad,
                            tag='quad'),
            targets=[xt.TargetSet(dx=0, at='end', tol=1e-9,tag='DS'),
                    # xt.TargetSet(betx=betx_fodo,bety=bety_fodo,dx=dx_fodo, at='quad_d_3', tol=1e-9, tag='FODO'),
                    xt.TargetSet(qx=qx_goal, qy=qy_goal, tol=1e-6, tag='tune')
                    # xt.TargetSet(qx=1.928704490718697, qy=1.3653071034699813, tol=1e-6, tag='tune')
                    ],
            method='6d',
            matrix_stability_tol=5e-3,
            # verbose=True
            )
print('MATCHING QUAD DS 6D')
match_ds_6d.target_status()
match_ds_6d.vary_status()
tw = line_ds.twiss(method='4d')
print(tw["qx"], tw["qy"])
plt.plot(tw['s'], tw['betx'], "r-", label=r'$\beta_x$')
plt.plot(tw['s'], tw['bety'], "b-", label=r'$\beta_y$')
plt.legend()
plt.show()

match_ds_6d_sxt=line_ds.match(vary=xt.VaryList(list_s,
                        step=step_sxt,
                        tag='sxt'),
            targets=[
                    xt.TargetSet(dqx=dqx_goal_arc, dqy=dqy_goal_arc, tol=1e-6, tag='chroma')],
            solve=False,
            method='6d',
             matrix_stability_tol=5e-3,
            # verbose=True
            )
match_ds_6d_sxt.step(20)
print('RESULTS SXT DS MATCH 6D')
match_ds_6d_sxt.target_status()
match_ds_6d_sxt.vary_status()

# match_ds_6d_chroma=line_ds.match(vary=[
#                     xt.VaryList(list_s, step=step_sxt, tag='sxt'),
#                     xt.VaryList(['k_f0','k_d1','k_f1','k_d2','k_f2','k_d3'],step=step_quad, tag='quad')
#                     ],           
#             targets=[
#                     xt.TargetSet(dx=0, at='end', tol=1e-6,tag='DS'),
#                     xt.TargetSet(dqx=dqx_goal_arc, dqy=dqy_goal_arc, tol=1e-6, tag='chroma'),
#                     xt.TargetSet(qx=qx_goal, qy=qy_goal, tol=1e-6, tag='tune'),
#                     # xt.TargetSet(qx=1.928704490718697, qy=1.3653071034699813, tol=1e-6, tag='tune'),
#                     xt.TargetSet(bx_chrom=0., by_chrom=0., tol=1e-3, at='end', tag='B')
#                     ],
#             solve=False,
#             method='6d',
#             matrix_stability_tol=5e-3,
#             compute_chromatic_properties=True
#             # verbose=True
#             )
# match_ds_6d_chroma.step(60)
# print('RESULTS SXT DS MATCH 6D')
# match_ds_6d_chroma.target_status()
# match_ds_6d_chroma.vary_status()

def plot_twiss_AB(tw,line):
    plt.figure()
    plt.plot(tw['s'], tw['ax_chrom'], label='$A_x$')
    plt.plot(tw['s'], tw['ay_chrom'], label='$A_y$')
    for i, el in enumerate(line.element_names):
        # if 'quad' in el:
        #     plt.axvline(x=tab['s',el], color='grey', linestyle='--',alpha=0.7)
        if 'sxt' in el:
            plt.axvline(x=tab['s',el], color='grey', linestyle='--')
        # elif 'cavity' in el:
        #     plt.axvline(x=tab['s',el], color='green', linestyle='--')
    plt.xlabel('s [m]')
    plt.ylabel('$A_x, A_y$ [m]')
    plt.legend(loc='upper right')
    plt.show()
    plt.figure()
    plt.plot(tw['s'], tw['bx_chrom'], label='$B_x$')
    plt.plot(tw['s'], tw['by_chrom'], label='$B_y$')
    for i, el in enumerate(line.element_names):
        # if 'quad' in el:
        #     plt.axvline(x=tab['s',el], color='grey', linestyle='--',alpha=0.7)
        if 'sxt' in el:
            plt.axvline(x=tab['s',el], color='grey', linestyle='--')
        # elif 'cavity' in el:
        #     plt.axvline(x=tab['s',el], color='green', linestyle='--')
    plt.xlabel('s [m]')
    plt.ylabel('$B_x, B_y$ [m]')
    plt.legend(loc='upper right')
    plt.show()


def plot_twiss_WD(tw,line):
    fig, ax1 = plt.figure(), plt.gca() 
    ax2 = ax1.twinx() 
    ax1.plot(tw['s'], tw['wx_chrom'], label='$W_x$')
    ax1.plot(tw['s'], tw['wy_chrom'], label='$W_y$')
    ax2.plot(tw['s'], tw['ddx'], label='Ddx', color='tab:red')
    for i, el in enumerate(line.element_names):
        if 'sxt' in el:
            ax1.axvline(x=tab['s', el], color='grey', linestyle='--')
    ax1.set_xlabel('s [m]')
    ax1.set_ylabel('$W_x, W_y$ [m]')
    ax2.set_ylabel('Ddx [m]')
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right')
    plt.show()

tw_6d=line_ds.twiss(method='6d', matrix_stability_tol=5e-3, 
                 compute_chromatic_properties=True)
plot_twiss(tw_6d, line_ds)
plot_twiss_WD(tw_6d,line_ds)
plot_twiss_AB(tw_6d,line_ds)

tw=tw_6d
plt.rc('legend', fontsize=10)    # legend fontsize

fig1 = plt.figure(1, figsize=(6.4, 4.8*1.5))
spbet = plt.subplot(2,1,1)
spdisp = plt.subplot(2,1,2, sharex=spbet)

spbet.plot(tw.s, tw.betx, label=r'$\beta_x$')
spbet.plot(tw.s, tw.bety,label=r'$\beta_y$')
spbet.set_ylabel(r'$\beta_{x,y}$ [m]')
spbet.legend()

# spbet.legend()
spdisp.plot(tw.s, tw.dx,label=r'$D_x')
spdisp.plot(tw.s, tw.dy,label=r'$D_y$')
spdisp.set_ylabel(r'$D_{x,y}$ [m]')
spdisp.legend()
fig1.subplots_adjust(left=.15, right=.92, hspace=.27)
plt.show()

# line_ds.to_json('lattice_disp_suppr_6d.json')

# print('GLOBAL PARAMETER')
# print(f'RCS Qx: {tw_6d["qx"]*nb_arc:.3f}')
# print(f'Arc Qx: {tw_6d["qx"]:.3f}')
# print(f'RCS Qy: {tw_6d["qy"]*nb_arc:.3f}')
# print(f'Arc Qy: {tw_6d["qy"]:.3f}')
# print(f'Arc Qs: {tw_6d["qs"]:.4f}')
# print(f'RCS Qs: {tw_6d["qs"]*nb_arc:.3f}')
# print(f'MCF: {tw_6d["momentum_compaction_factor"]:.5f}')

# Plot trajectory
# plt.figure()
# tab=line_cell.get_table()
# plt.plot(survey_cell['s'], survey_cell['X']+rec_cell.x[0], label='Inj')
# for i, el in enumerate(line_cell.element_names):
#     if 'en' in el:
#         plt.axvline(x=tab['s',el], color='grey', linestyle='--')
#     if 'ex' in el:
#         plt.axvline(x=tab['s',el], color='grey', linestyle='--')
# plt.xlabel('s [m]')
# plt.ylabel('x [m]')
# plt.legend()
# plt.show()

#     #Plot layout of arc
#     import xplt
#     sv = line_ds.survey()
#     plot = xplt.FloorPlot(
#     sv,
#     line_ds,
#     projection="ZX",
#     boxes={  # adjust box style for element names matching regex
#         "BSC": dict(color="maroon",width=5),
#         "BNC": dict(color="blue",width=5),
#         "quad..." : dict(color="green",width=15),
#         "sxt":dict(color="purple",width=20)
#     }
# )
#     plot.ax.set_ylim(-50, 50)
#     # plot.add_scale()
#     plot.legend(loc="upper left")

# line_cell_ext,line_ds_ext= make_optics(file_input,1,n_k,method) 
# survey_cell_ext=line_cell_ext.survey(theta0=RCS.cell_angle/4)
# rec_cell_ext=track(line_cell_ext,var=False)

# tab=line_cell_ext.get_table()
# plt.figure()
# plt.plot(survey_cell['s'], survey_cell['X'], label='Inj')
# plt.plot(survey_cell_ext['s'], survey_cell_ext['X'], label='Ext')
# plt.axvline(x=tab['s','en1'], color='grey', linestyle='--')
# plt.axvline(x=tab['s','ex1'], color='grey', linestyle='--')
# plt.axvline(x=tab['s','en2'], color='grey', linestyle='--')
# plt.axvline(x=tab['s','ex2'], color='grey', linestyle='--')
# plt.axvline(x=tab['s','en3'], color='grey', linestyle='--')
# plt.axvline(x=tab['s','ex3'], color='grey', linestyle='--')
# plt.text(tab['s','BSC..10'], 0.10, 'SC', horizontalalignment='center')
# plt.text(tab['s','BNC..10'], 0.10, 'NC', horizontalalignment='center')
# plt.text(tab['s','BSC_1..10'], 0.10, 'SC', horizontalalignment='center')
