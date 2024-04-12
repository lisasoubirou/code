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
from interpolation import calc_dip_coef,eval_horner
import json

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

#Importing line
method='var_k'
n_slice=21
t_ref=0.5
t_acc=1.09704595*1e-3

#Tracking parameters
n_turns = 26*55
n_part=1

file_input='/mnt/c/muco/code/class_geometry/para_RCS_ME.txt'
RCS = Geometry(file_input)
file_seq = 'lattice_disp_suppr_6d.json'
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

line._init_var_management()
line.vars['t_turn_s']=0.

coef=calc_dip_coef(file_input,t_ref)
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

line.element_refs['dz_acc'].dzeta=line.vars['dz_rf']
line.element_refs['dz_acc_1'].dzeta=line.vars['dz_rf']

h_ref_NC=RCS.hn(t_ref)[i_BNC]
h_ref_SC=RCS.hn(t_ref)[i_BSC]
l_dip_ref_NC=RCS.L_dip_path(t_ref)[i_BNC]
l_dip_ref_SC=RCS.L_dip_path(t_ref)[i_BSC]

# Time dependent knobs on dipole parameters
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
        
line.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Teapot(n_slice))])
tab_sliced=line.get_table()

mon_sc0_mid=xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns,
                        num_particles=n_part)
mon_sc0_en=xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns,
                        num_particles=n_part)
mon_sc0_ex=xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns,
                        num_particles=n_part)
mon_nc0_en=xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns,
                        num_particles=n_part)
mon_nc0_mid=xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns,
                        num_particles=n_part)
mon_nc0_ex=xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns,
                        num_particles=n_part)
mon_sc1_mid=xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns,
                        num_particles=n_part)
mon_sc1_en=xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns,
                        num_particles=n_part)
mon_sc1_ex=xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns,
                        num_particles=n_part)

line.insert_element('m_sc0_en',mon_sc0_en,at_s=tab_sliced['s','en1'])
line.insert_element('m_sc0_mid',mon_sc0_mid,at_s=tab_sliced['s','BSC..10'])
line.insert_element('m_sc0_ex',mon_sc0_ex,at_s=tab_sliced['s','ex1'])
line.insert_element('m_nc0_en',mon_nc0_en,at_s=tab_sliced['s','en2'])
line.insert_element('m_nc0_mid',mon_nc0_mid,at_s=tab_sliced['s','BNC..10'])
line.insert_element('m_nc0_ex',mon_nc0_ex,at_s=tab_sliced['s','ex2'])
line.insert_element('m_sc1_en',mon_sc1_en,at_s=tab_sliced['s','en3'])
line.insert_element('m_sc1_mid',mon_sc1_mid,at_s=tab_sliced['s','BSC_1..10'])
line.insert_element('m_sc1_ex',mon_sc1_ex,at_s=tab_sliced['s','ex3'])


# line.insert_element('m_sc0_en',mon_sc0_en,at_s=tab_sliced['s','en1_8'])
# line.insert_element('m_sc0_mid',mon_sc0_mid,at_s=tab_sliced['s','BSC_16..10'])
# line.insert_element('m_sc0_ex',mon_sc0_ex,at_s=tab_sliced['s','ex1_8'])
# line.insert_element('m_nc0_en',mon_nc0_en,at_s=tab_sliced['s','en2_8'])
# line.insert_element('m_nc0_mid',mon_nc0_mid,at_s=tab_sliced['s','BNC_8..10'])
# line.insert_element('m_nc0_ex',mon_nc0_ex,at_s=tab_sliced['s','ex2_8'])
# line.insert_element('m_sc1_en',mon_sc1_en,at_s=tab_sliced['s','en3_8'])
# line.insert_element('m_sc1_mid',mon_sc1_mid,at_s=tab_sliced['s','BSC_17..10'])
# line.insert_element('m_sc1_ex',mon_sc1_ex,at_s=tab_sliced['s','ex3_8'])

def track_cell(line, num_turns=1):
    line.build_tracker()
    part=line.build_particles()
    if num_turns==1:
        line.track(part,num_turns=1, turn_by_turn_monitor='ONE_TURN_EBE',ele_start='en1',ele_stop='ex3')
    else:
        line.track(part,num_turns=num_turns, turn_by_turn_monitor=True)
    rec=line.record_last_track
    return (rec)

#Check traj in cells for diff t_turn_s
# plt.figure()
# t_traj=np.linspace(0,1,8)
# for t in t_traj:
#     line.vars['t_turn_s']=t
#     rec=track_cell(line)
#     plt.plot(rec.s[0],rec.x[0]) #,label=f't={t}'
# plt.legend()
# plt.xlabel('s [m]')
# plt.ylabel('x [m]')
# plt.show()    

## Plotting check
# t_test=np.linspace(0,1,20)*t_acc
# h_nc_test=[]
# h_sc_test=[]
# dz=[]
# for tt in t_test:
#     line.vars['t_turn_s'] = tt
#     h_nc_test.append(line.vars['h_NC']._get_value())
#     h_sc_test.append(line.vars['h_SC']._get_value())
#     dz.append(line.vars['dz_rf']._get_value())
# plt.figure()
# plt.plot(t_test,dz)
# plt.show()
# plt.figure()
# plt.plot(t_test,h_nc_test)
# plt.show()
# plt.figure()
# plt.plot(t_test,h_sc_test)
# plt.show()

line.build_tracker()
line.enable_time_dependent_vars = True
particles = line.build_particles(x=0, px=0,y=0, py=0, delta=0,
    method='6d')
line.track(particles, num_turns=n_turns, turn_by_turn_monitor=True)
particles.sort(interleave_lost_particles=True)
rec=line.record_last_track

#Results of monitors
plt.figure(1)
plt.scatter(mon_sc0_en.s[0][::143],mon_sc0_en.x[0][::143],label='sc en')
plt.scatter(mon_sc0_mid.s[0][::143], mon_sc0_mid.x[0][::143],label='sc mid')
plt.scatter(mon_sc0_ex.s[0][::143],mon_sc0_ex.x[0][::143],label='sc ex')
plt.scatter(mon_nc0_en.s[0][::143],mon_nc0_en.x[0][::143],label='nc en')
plt.scatter(mon_nc0_mid.s[0][::143],mon_nc0_mid.x[0][::143],label='nc mid')
plt.scatter(mon_nc0_ex.s[0][::143],mon_nc0_ex.x[0][::143],label='nc ex')
plt.scatter(mon_sc1_en.s[0][::143],mon_sc1_en.x[0][::143],label='sc2 en')
plt.scatter(mon_sc1_mid.s[0][::143], mon_sc1_mid.x[0][::143],label='sc2 mid')
plt.scatter(mon_sc1_ex.s[0][::143],mon_sc1_ex.x[0][::143],label='sc2 ex')

plt.axvline(x=mon_sc0_en.s[0][0], color='grey', linestyle='--')
plt.axvline(x=mon_sc0_ex.s[0][0], color='grey', linestyle='--')
plt.axvline(x=mon_nc0_en.s[0][0], color='grey', linestyle='--')
plt.axvline(x=mon_nc0_ex.s[0][0], color='grey', linestyle='--')
plt.axvline(x=mon_sc1_en.s[0][0], color='grey', linestyle='--')
plt.axvline(x=mon_sc1_ex.s[0][0], color='grey', linestyle='--')

plt.text(mon_sc0_mid.s[0][0], 0.006, 'SC', horizontalalignment='center')
plt.text(mon_nc0_mid.s[0][0], 0.006, 'NC', horizontalalignment='center')
plt.text(mon_sc1_mid.s[0][0], 0.006, 'SC', horizontalalignment='center')
plt.xlabel('s [m] ')
plt.ylabel('x [m]')
# plt.legend(loc="lower left")
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))  
axes[0, 0].scatter(rec.x, rec.px,s=2)
axes[0, 0].set_xlabel('x [m]')
axes[0, 0].set_ylabel("x'")
# axes[0, 0].set_ylim(-1e-4, 1e-4)
# axes[0, 0].set_xlim(-1e-4, 1e-4)
#axes[0, 0].axis('equal')
axes[0, 1].scatter(rec.y, rec.py,s=2)
axes[0, 1].set_xlabel('y [m]')
axes[0, 1].set_ylabel("y'")
# axes[0, 1].set_ylim(-1e-4, 1e-4)
# axes[0, 1].set_xlim(-1e-4, 1e-4)
#axes[0, 1].axis('equal')
axes[1, 0].scatter(rec.x, rec.y,s=2)
axes[1, 0].set_xlabel('x [m]')
axes[1, 0].set_ylabel('y [m]')
# axes[1, 0].set_ylim(-1e-4, 1e-4)
# axes[1, 0].set_xlim(-1e-4, 1e-4)
axes[1, 1].scatter(rec.zeta, rec.delta, s=2)
axes[1, 1].set_xlabel('z [m]')
axes[1, 1].set_ylabel('$\delta$')
for ax in axes.flat:
    ax.ticklabel_format(style='sci', scilimits=(-3, 3), axis='both')
plt.tight_layout()
plt.show()

#Printing test
# print('Time',line.vars['t_turn_s']._get_value())
# print('BSC',line['BSC'].k0)
# print('BSC, vars',line.vars['h_SC']._get_value())
# print('BSC geo', RCS.hn(0)[0])
# print('BNC',line['BNC'].k0)
# print('BNC, vars',line.vars['h_NC']._get_value())
# print('BNC geo', RCS.hn(0)[1])
# line.vars['t_turn_s']=0.5
# print('Time',line.vars['t_turn_s']._get_value())
# print('BSC',line['BSC'].k0)
# print('BSC vars',line.vars['h_SC']._get_value())
# print('BSC geo', RCS.hn(0.5)[0])
# print('BNC',line['BNC'].k0)
# print('BNC, vars',line.vars['h_NC']._get_value())
# print('BNC geo', RCS.hn(0.5)[1])

# Plot x function of turn in dipole
plt.figure()
plt.plot(mon_sc0_en.x[0],label='sc en')
plt.plot(mon_sc0_mid.x[0],label='sc mid')
plt.plot(mon_sc0_ex.x[0],label='sc ex')
plt.xlabel('n_turns in arc')
plt.ylabel('x [mm]')
plt.legend()
plt.show()
plt.figure()
plt.plot(mon_nc0_en.x[0],label='nc en')
plt.plot(mon_nc0_mid.x[0],label='nc mid')
plt.plot(mon_nc0_ex.x[0],label='nc ex')
plt.xlabel('n_turns in arc')
plt.ylabel('x [mm]')
plt.legend()
plt.show()
plt.figure()
plt.plot(mon_sc1_en.x[0],label='sc2 en')
plt.plot(mon_sc1_mid.x[0],label='sc2 mid')
plt.plot(mon_sc1_ex.x[0],label='sc2 ex')
plt.xlabel('n_turns in arc')
plt.ylabel('x [mm]')
plt.legend()
plt.show()