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
sys.path.append('/mnt/c/muco')
from class_geometry.class_geo import Geometry 
from optics_function import plot_twiss
from interpolation import calc_dip_coef,eval_horner
from track_function import track_cell,track,distribution_lost_turn,distribution_lost_turn_long
from track_function import compute_emit_x_n, compute_emit_y_n, compute_emit_s, compute_sigma_z, compute_x_mean, compute_y_mean,calculate_transmission
import json

from scipy.interpolate import CubicSpline

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

#Constants
muon_mass_kg=sc.physical_constants['muon mass'][0]
e_charge=sc.e
c=sc.c

#Importing line
method='var_k'
n_slice=21
t_ref=0.5
t_acc=1.09704595*1e-3
n_part=5000

file_input='/mnt/c/muco/code/class_geometry/para_RCS_ME.txt'
RCS = Geometry(file_input)
C=RCS.C
nb_arc=RCS.nb_arc
n_cav_arc = 2 
E_inj=RCS.E_inj
E_ext=RCS.E_ext
frequency_rf=RCS.RF_freq
N_turn_rcs=55
n_turns = N_turn_rcs*nb_arc
energy_increment_per_arc = (E_ext-E_inj)/n_turns
sync_phase=45
phase=180-sync_phase
volt_arc=energy_increment_per_arc/np.sin(phase*np.pi/180)
volt_cav=volt_arc/n_cav_arc
gamma_0=RCS.inj_gamma
beta_0=np.sqrt(1-1/gamma_0**2)

file_seq = '/mnt/c/muco/code/lattice_disp_suppr_6d_dq_5.json'
line=xt.Line.from_json(file_seq)

line.particle_ref = xt.Particles(mass0=xp.MUON_MASS_EV, q0=1.,
                                    energy0=RCS.E_inj,
                                 )

arr_delta = np.linspace(-0.02, 0.02, 41)
arr_twiss = np.zeros((41, 4))
for i, delta in enumerate(arr_delta):
    tw_6d = line.twiss(method='6d', matrix_stability_tol=5e-3, delta0=delta)
    arr_twiss[i, 0] = tw_6d.betx[0]
    arr_twiss[i, 1] = tw_6d.alfx[0]
    arr_twiss[i, 2] = tw_6d.bety[0]
    arr_twiss[i, 3] = tw_6d.alfy[0]

twiss_delta = CubicSpline(arr_delta, arr_twiss, bc_type='not-a-knot')

tw_6d=line.twiss(method='6d', matrix_stability_tol=5e-3)

tw_6d=line.twiss(method='6d', matrix_stability_tol=5e-3, 
                 compute_chromatic_properties=True)
h_rf = np.round(frequency_rf*tw_6d.T_rev0*nb_arc)
MCF=tw_6d['momentum_compaction_factor']

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

#Errors dipole
b3=0.3
b5=-0.8
r_rad=0.01

# Time dependent knobs on dipole parameters
for el in tab.rows[tab.element_type == 'Bend'].name:
    if 'BSC' in el:
         line.element_refs[el].knl[2]=2*10e-4*b3*r_rad**(-2)*line.vars['h_SC']
         line.element_refs[el].knl[4]=24*10e-4*b5*r_rad**(-4)*line.vars['h_SC']
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
print('slicing done')
tab_sliced=line.get_table()

print('insert monitors')
# mon_sc0_mid=xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns,
#                         num_particles=n_part)
# mon_sc0_en=xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns,
#                         num_particles=n_part)
# mon_sc0_ex=xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns,
#                         num_particles=n_part)
# mon_nc0_en=xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns,
#                         num_particles=n_part)
# mon_nc0_mid=xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns,
#                         num_particles=n_part)
# mon_nc0_ex=xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns,
#                         num_particles=n_part)
# mon_sc1_mid=xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns,
#                         num_particles=n_part)
# mon_sc1_en=xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns,
#                         num_particles=n_part)
# mon_sc1_ex=xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns,
#                         num_particles=n_part)

#FODO cell
# line.insert_element('m_sc0_en',mon_sc0_en,at_s=tab_sliced['s','en1_8'])
# line.insert_element('m_sc0_mid',mon_sc0_mid,at_s=tab_sliced['s','BSC_16..10'])
# line.insert_element('m_sc0_ex',mon_sc0_ex,at_s=tab_sliced['s','ex1_8'])
# line.insert_element('m_nc0_en',mon_nc0_en,at_s=tab_sliced['s','en2_8'])
# line.insert_element('m_nc0_mid',mon_nc0_mid,at_s=tab_sliced['s','BNC_8..10'])
# line.insert_element('m_nc0_ex',mon_nc0_ex,at_s=tab_sliced['s','ex2_8'])
# line.insert_element('m_sc1_en',mon_sc1_en,at_s=tab_sliced['s','en3_8'])
# line.insert_element('m_sc1_mid',mon_sc1_mid,at_s=tab_sliced['s','BSC_17..10'])
# line.insert_element('m_sc1_ex',mon_sc1_ex,at_s=tab_sliced['s','ex3_8'])

#DS cell
# line.insert_element('m_sc0_en',mon_sc0_en,at_s=tab_sliced['s','en1'])
# line.insert_element('m_sc0_mid',mon_sc0_mid,at_s=tab_sliced['s','BSC..10'])
# line.insert_element('m_sc0_ex',mon_sc0_ex,at_s=tab_sliced['s','ex1'])
# line.insert_element('m_nc0_en',mon_nc0_en,at_s=tab_sliced['s','en2'])
# line.insert_element('m_nc0_mid',mon_nc0_mid,at_s=tab_sliced['s','BNC..10'])
# line.insert_element('m_nc0_ex',mon_nc0_ex,at_s=tab_sliced['s','ex2'])
# line.insert_element('m_sc1_en',mon_sc1_en,at_s=tab_sliced['s','en3'])
# line.insert_element('m_sc1_mid',mon_sc1_mid,at_s=tab_sliced['s','BSC_1..10'])
# line.insert_element('m_sc1_ex',mon_sc1_ex,at_s=tab_sliced['s','ex3'])

# line.to_json('6d_dq_5_sliced_ramp.json')

#Tracking parameters
nemitt_x_0 = 25e-6
nemitt_y_0 = 25e-6
emit_s=0.025 #eVs
# deltaz_4_sigma_ns=0.077
# sigma_z = deltaz_4_sigma_ns*1e-9*sc.c
# sigma_z=0.01

print('go to tracker')
line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))
# line.build_tracker()
print('tracker built')
particles = xp.generate_matched_gaussian_bunch(line=line,
                                               matrix_stability_tol=5e-3,
                                               num_particles=n_part,
                                               nemitt_x=nemitt_x_0, 
                                               nemitt_y=nemitt_y_0, 
                                               sigma_z=0,
                                               )

rfbucket = xp.longitudinal.rf_bucket.RFBucket(
                            circumference=C/nb_arc,
                            gamma=gamma_0,
                            mass_kg=muon_mass_kg,
                            charge_coulomb=e_charge,
                            alpha_array=np.atleast_1d(MCF),
                            harmonic_list=np.atleast_1d(h_rf/nb_arc),
                            voltage_list=np.atleast_1d(volt_arc), #*nb_arc
                            phi_offset_list=np.atleast_1d((sync_phase)*np.pi/180),
                            p_increment=energy_increment_per_arc * e_charge / c) #*nb_arc

matcher = xp.longitudinal.rfbucket_matching.RFBucketMatcher(rfbucket=rfbucket,
                          distribution_type=xp.longitudinal.rfbucket_matching.ThermalDistribution,
                        #   sigma_z=sigma_z,
                          epsn_z=4*np.pi*emit_s
                        )

z_particles, delta_particles, _, _ = matcher.generate(macroparticlenumber=n_part)
particles.zeta=z_particles
particles.delta=delta_particles
tw_loc = twiss_delta(particles.delta)
Ax = np.sqrt(tw_loc[:, 0]/tw_6d.betx[0])
Bx = (tw_6d.alfx[0]-tw_loc[:,1])/np.sqrt(tw_6d.betx[0]*tw_loc[:,0])
Ay = np.sqrt(tw_loc[:, 2]/tw_6d.bety[0])
By = (tw_6d.alfy[0]-tw_loc[:,3])/np.sqrt(tw_6d.bety[0]*tw_loc[:,2])
particles.px = -Bx*particles.x + particles.px/Ax + tw_6d.ddpx[0]*particles.delta**2
particles.x = Ax*particles.x + tw_6d.ddx[0]*particles.delta**2
particles.py = -By*particles.y + particles.py/Ay + tw_6d.ddpy[0]*particles.delta**2
particles.y = Ay*particles.y + tw_6d.ddy[0]*particles.delta**2
particles0=particles.copy()
print('Particles ok')

track_log = xt.Log(x_mean=compute_x_mean,y_mean=compute_y_mean, 
                   eps_x=compute_emit_x_n, eps_y=compute_emit_y_n,
                   eps_s=compute_emit_s, sig_z=compute_sigma_z)
line.enable_time_dependent_vars = True
line.track(particles, log=track_log, num_turns=n_turns, 
        with_progress=True,turn_by_turn_monitor=True)
rec=line.record_last_track
particles.sort(interleave_lost_particles=True)
x_mean = line.log_last_track['x_mean']
y_mean = line.log_last_track['y_mean']
eps_x=line.log_last_track['eps_x']
eps_y=line.log_last_track['eps_y']
eps_s=line.log_last_track['eps_s']
sig_z=line.log_last_track['sig_z']

plt.figure()
ax1 = plt.subplot(2,1,1)
plt.ylabel(r'$x_{centroid}$')
plt.plot(x_mean)
ax2 = plt.subplot(2,1,2, sharex=ax1)
plt.plot(y_mean)
plt.ylabel(r'$y_{centroid}$')
plt.xlabel('Turn')
plt.show()

plt.figure(figsize=(7, 7))
ax1 = plt.subplot(4, 1, 1)
plt.ylabel(r'$\Delta \epsilon / \epsilon_{x,0}$')
plt.plot((eps_x - eps_x[0]) / eps_x[0])
ax2 = plt.subplot(4, 1, 2, sharex=ax1)
plt.ylabel(r'$\Delta \epsilon /\epsilon_{y,0}$')
plt.plot((eps_y - eps_y[0]) / eps_y[0])
ax3 = plt.subplot(4, 1, 3, sharex=ax1)
color = 'tab:red'
ax3.set_ylabel('$\sigma_z$ [m]', color=color)
ax3.plot(sig_z, color=color)
ax3.tick_params(axis='y', labelcolor=color)
ax3_2 = ax3.twinx()  
color = 'tab:blue'
ax3_2.set_ylabel('$ \Delta \epsilon_s / \epsilon_s$ [eVs]', color=color)
ax3_2.plot((eps_s-eps_s[0])/eps_s[0], color=color)
ax3_2.tick_params(axis='y', labelcolor=color)
lost_part=particles.at_turn[particles.state<0]
ax4 = plt.subplot(4, 1, 4, sharex=ax1)
ax4.set_ylabel('tr')
ax4.plot(calculate_transmission(sorted(lost_part),n_part,n_turns),color=color)
plt.xlabel('Turn')
plt.tight_layout()
plt.show()

distribution_lost_turn(particles0,particles)
distribution_lost_turn_long(particles0,particles)

mask= particles.state > 0
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))  
axes[0, 0].scatter(rec.x[mask], rec.px[mask],s=2)
axes[0, 0].set_xlabel('x [m]')
axes[0, 0].set_ylabel("x'")
# axes[0, 0].set_ylim(-1e-4, 1e-4)
# axes[0, 0].set_xlim(-1e-4, 1e-4)
#axes[0, 0].axis('equal')
axes[0, 1].scatter(rec.y[mask], rec.py[mask],s=2)
axes[0, 1].set_xlabel('y [m]')
axes[0, 1].set_ylabel("y'")
# axes[0, 1].set_ylim(-1e-4, 1e-4)
# axes[0, 1].set_xlim(-1e-4, 1e-4)
#axes[0, 1].axis('equal')
axes[1, 0].scatter(rec.x[mask], rec.y[mask],s=2)
axes[1, 0].set_xlabel('x [m]')
axes[1, 0].set_ylabel('y [m]')
# axes[1, 0].set_ylim(-1e-4, 1e-4)
# axes[1, 0].set_xlim(-1e-4, 1e-4)
axes[1, 1].scatter(rec.zeta[mask], rec.delta[mask], s=2)
axes[1, 1].set_xlabel('z [m]')
axes[1, 1].set_ylabel(r'$\delta$')
for ax in axes.flat:
    ax.ticklabel_format(style='sci', scilimits=(-3, 3), axis='both')
plt.tight_layout()
plt.show()

# #Check traj in cells for diff t_turn_s
# plt.figure()
# t_traj=np.linspace(0,t_acc,5)
# for t in t_traj:
#     line.vars['t_turn_s']=t
#     rec=track_cell(line)
#     plt.plot(rec.s[0]+tab_sliced['s','en1'],rec.x[0]) #,label=f't={t}'
# plt.axvline(x=tab_sliced['s','en1'], color='grey', linestyle='--')
# plt.axvline(x=tab_sliced['s','ex1'], color='grey', linestyle='--')
# plt.axvline(x=tab_sliced['s','en2'], color='grey', linestyle='--')
# plt.axvline(x=tab_sliced['s','ex2'], color='grey', linestyle='--')
# plt.axvline(x=tab_sliced['s','en3'], color='grey', linestyle='--')
# plt.axvline(x=tab_sliced['s','ex3'], color='grey', linestyle='--')
# plt.text(tab_sliced['s','BSC..10'], 0.007, 'SC', horizontalalignment='center')
# plt.text(tab_sliced['s','BNC..10'], 0.007, 'NC', horizontalalignment='center')
# plt.text(tab_sliced['s','BSC_1..10'], 0.007, 'SC', horizontalalignment='center')
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

#Track 1 particle

# rec,partciles=track(line,True,num_turns=n_turns)

# Results of monitors (1part)
# plt.figure()
# plt.scatter(mon_sc0_en.s[0][::143],mon_sc0_en.x[0][::143],label='sc en')
# plt.scatter(mon_sc0_mid.s[0][::143], mon_sc0_mid.x[0][::143],label='sc mid')
# plt.scatter(mon_sc0_ex.s[0][::143],mon_sc0_ex.x[0][::143],label='sc ex')
# plt.scatter(mon_nc0_en.s[0][::143],mon_nc0_en.x[0][::143],label='nc en')
# plt.scatter(mon_nc0_mid.s[0][::143],mon_nc0_mid.x[0][::143],label='nc mid')
# plt.scatter(mon_nc0_ex.s[0][::143],mon_nc0_ex.x[0][::143],label='nc ex')
# plt.scatter(mon_sc1_en.s[0][::143],mon_sc1_en.x[0][::143],label='sc2 en')
# plt.scatter(mon_sc1_mid.s[0][::143], mon_sc1_mid.x[0][::143],label='sc2 mid')
# plt.scatter(mon_sc1_ex.s[0][::143],mon_sc1_ex.x[0][::143],label='sc2 ex')
# plt.axvline(x=mon_sc0_en.s[0][0], color='grey', linestyle='--')
# plt.axvline(x=mon_sc0_ex.s[0][0], color='grey', linestyle='--')
# plt.axvline(x=mon_nc0_en.s[0][0], color='grey', linestyle='--')
# plt.axvline(x=mon_nc0_ex.s[0][0], color='grey', linestyle='--')
# plt.axvline(x=mon_sc1_en.s[0][0], color='grey', linestyle='--')
# plt.axvline(x=mon_sc1_ex.s[0][0], color='grey', linestyle='--')
# plt.text(mon_sc0_mid.s[0][0], 0.006, 'SC', horizontalalignment='center')
# plt.text(mon_nc0_mid.s[0][0], 0.006, 'NC', horizontalalignment='center')
# plt.text(mon_sc1_mid.s[0][0], 0.006, 'SC', horizontalalignment='center')
# plt.xlabel('s [m] ')
# plt.ylabel('x [m]')
# # plt.legend(loc="lower left")
# plt.show()

# Plot x function of turn in dipole (1part)
# plt.figure()
# plt.plot(mon_sc0_en.x[0],label='sc en')
# plt.plot(mon_sc0_mid.x[0],label='sc mid')
# plt.plot(mon_sc0_ex.x[0],label='sc ex')
# plt.xlabel('n_turns in arc')
# plt.ylabel('x [mm]')
# plt.legend()
# plt.show()
# plt.figure()
# plt.plot(mon_nc0_en.x[0],label='nc en')
# plt.plot(mon_nc0_mid.x[0],label='nc mid')
# plt.plot(mon_nc0_ex.x[0],label='nc ex')
# plt.xlabel('n_turns in arc')
# plt.ylabel('x [mm]')
# plt.legend()
# plt.show()
# plt.figure()
# plt.plot(mon_sc1_en.x[0],label='sc2 en')
# plt.plot(mon_sc1_mid.x[0],label='sc2 mid')
# plt.plot(mon_sc1_ex.x[0],label='sc2 ex')
# plt.xlabel('n_turns in arc')
# plt.ylabel('x [mm]')
# plt.legend()
# plt.show()
