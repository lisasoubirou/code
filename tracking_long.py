# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:10:49 2024
@author: LS276867

This code allows tracking in 6d. Here, the longitudinal plane is of interest.
Plot the RF bucket/separatrix.
Create matched particle distribution to the RF bucket.
Functions:
    - calc_separatrix: calculate the separatrix parameters and plot it
    - calc_emitt: calculate emittance "manually" using the covariant matrix
Routines to plot space phase

"""
import numpy as np
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt
import xpart as xp
import json
import sys
import scipy.constants as sc

#Constants
muon_mass_kg=sc.physical_constants['muon mass'][0]
e_charge=sc.e
c=sc.c

#Plotting default parameters
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

#RCS parameters
C=5990
nb_arc=26
n_rf = 24
n_cav_arc = 2 
t_acc=1.097
E_inj=313.830e9
gamma_inj= (xp.MUON_MASS_EV + E_inj)/xp.MUON_MASS_EV
E_ext=750e9
frequency_rf=1300*1e6
N_turn_rcs=55
N_frac=1
n_turns = N_turn_rcs*nb_arc*N_frac
energy_increment_per_arc = (E_ext-E_inj)/n_turns
sync_phase=45
phase=180-sync_phase
volt_arc=energy_increment_per_arc/np.sin(phase*np.pi/180)
volt_cav=volt_arc/n_cav_arc
# nemitt_x=3e-6
# nemitt_y=3e-6
emit_s=0.025 #eVs
sigma_z=9.5*1e-3
n_particles=5000
n_monitored=5000

#Importing line
# file_seq = 'lattice_disp_suppr.json'
file_seq = 'lattice_disp_suppr_6d.json'

line_ds=xt.Line.from_json(file_seq)
line_ds.particle_ref = xt.Particles(mass0=xp.MUON_MASS_EV, q0=1.,
                                    energy0=xp.MUON_MASS_EV + E_inj,
                                 )
tw_ds_bf=line_ds.twiss(method='4d')
method = "6d" 

#Cavity settings
# frequency = np.round(frequency_rf*tw_ds_bf.T_rev0)/tw_ds_bf.T_rev0
# line_ds['cavity'].frequency=frequency
# line_ds['cavity'].lag=phase
# line_ds['cavity'].voltage=volt_cav
# line_ds['cavity_1'].frequency=frequency
# line_ds['cavity_1'].lag=phase
# line_ds['cavity_1'].voltage=volt_cav

#%% Optics
#Add skew components
#line_ds['sxt_d'].ksl[2]=-0.1
#line_ds['quad_d_2'].ksl[1]=1e-6

#Parameters for new matching
# betx_fodo=tw_ds_bf.rows['quad_d_3']['betx'][0]
# bety_fodo=tw_ds_bf.rows['quad_d_3']['bety'][0]
# dx_fodo=tw_ds_bf.rows['quad_d_3']['dx'][0]
# Qx= 1.61
# Qy=1.33
# print("Start Matching")
# opt = line_ds.match(vary=xt.VaryList([
#                                 'k_f0',
#                                 'k_d1',
#                                 'k_f1',
#                                 'k_d2',
#                                 'k_f2',
#                                 'k_d3',
#                                 's_d',
#                                 's_f'
#                                 ],
#                       step=1e-8),
#               targets=[
#                     xt.Target(tar='dx', at='end', value=0. , tol=1e-12, weight=1),
#                     xt.Target(tar='betx', at='quad_d_3', value=betx_fodo , tol=1e-12, weight=1),
#                     xt.Target(tar='bety', at='quad_d_3', value=bety_fodo , tol=1e-12, weight=1),
#                     xt.Target(tar='dx', at='quad_d_3', value=dx_fodo , tol=1e-12, weight=1),
#                     xt.Target('qx', value=1.61 ),
#                     xt.Target('qy', value=1.33 ),
#                     xt.Target('dqx', value=0., tol=1e-3),
#                     xt.Target('dqy', value=0.,tol=1e-3)
#                     ],
#             #verbose=True,
#             #assert_within_tol=False,
#             #solve=False,
#             method='4d',
#             )
# print(opt.target_status())
# print(opt.vary_status())
# print("End matching")

#Twiss
tab = line_ds.get_table()
beta_x_end=tw_ds_bf['betx'][-1]
beta_y_end=tw_ds_bf['bety'][-1]
alf_x_end=tw_ds_bf['alfx'][-1]
alf_y_end=tw_ds_bf['alfy'][-1]

#Accelerating elements
h_rf = np.round(frequency_rf*tw_ds_bf.T_rev0*nb_arc) #data tw_ds on 1 arc
# RF_acc=xt.ReferenceEnergyIncrease(Delta_p0c=energy_increment_per_arc/n_cav_arc)
# line_ds.discard_tracker()
# line_ds.insert_element('RF_acc', RF_acc, at_s=tab['s','cavity'])
# line_ds.insert_element('RF_acc_1', RF_acc, at_s=tab['s','cavity_1'])
# mon=xt.ParticlesMonitor(start_at_turn=0,
#                         stop_at_turn = n_turns,
#                         num_particles=n_monitored)
# line_ds.insert_element('mon', mon, at_s=tab['s','end'])
# line_ds.build_tracker()
tw_ds=line_ds.twiss(method=method,matrix_stability_tol=5e-3)
MCF=tw_ds['momentum_compaction_factor']

#%% Generating particles grid
# def grid(sigma_z, delta, n_x, n_y):
#     x=np.linspace(-sigma_z,sigma_z, n_r)
#     y=np.linspace(0,delta, n_theta)
#     xx, yy = np.meshgrid(x, y)
#     xx=np.ravel(xx)
#     yy=np.ravel(yy)
#     return(xx,yy)

# n_r = 15
# n_theta = 1
# n_part=n_r*n_theta
# sigma_z = 8e-2 #RMS bunch length [m]
# delta=0
# list_zeta,list_delta=grid(sigma_z, delta, n_r, n_theta)
# particles = line_ds.build_particles(
#     x=0, px=0,
#     y=0, py=0,
#     zeta=list_zeta,
#     delta=list_delta,
#     method=method)

#%% Matched distribution 
rfbucket = xp.longitudinal.rf_bucket.RFBucket(
                            circumference=C/nb_arc,
                            gamma=gamma_inj,
                            mass_kg=muon_mass_kg,
                            charge_coulomb=e_charge,
                            alpha_array=np.atleast_1d(MCF),
                            harmonic_list=np.atleast_1d(h_rf/nb_arc),
                            voltage_list=np.atleast_1d(volt_arc), #*nb_arc
                            phi_offset_list=np.atleast_1d((phase)*np.pi/180),
                            p_increment=energy_increment_per_arc * e_charge / c) #*nb_arc

matcher = xp.longitudinal.rfbucket_matching.RFBucketMatcher(rfbucket=rfbucket,
                          distribution_type=xp.longitudinal.rfbucket_matching.ThermalDistribution,
                          sigma_z=sigma_z,
                        #   epsn_z=4*np.pi*emit_s
                        )

z_particles, delta_particles, _, _ = matcher.generate(macroparticlenumber=n_particles)
particles = line_ds.build_particles(
            zeta=z_particles-rfbucket.z_sfp, 
            delta=delta_particles,
            # delta=1e-3,
            # x=1e-3,
            # x_norm=x_in_sigmas, px_norm=px_in_sigmas,
            # y_norm=y_in_sigmas, py_norm=py_in_sigmas,
            # nemitt_x=norm_emit_x, nemitt_y=norm_emit_y,
            # weight=initial_bunch_intensity/n_macroparticles
            )

# Initial particle distribution
x0=np.array(particles.x)
y0=np.array(particles.y)
z0=np.array(particles.zeta)
delta0=np.array(particles.delta)

#%% Tracking 
line_ds.track(particles, num_turns=n_turns, turn_by_turn_monitor=True)
particles.sort(interleave_lost_particles=True)
rec=line_ds.record_last_track
mask= particles.state > 0

#Plot dynamic aperture
# plt.figure()
# # plt.scatter(list_zeta, list_delta, c=particles.at_turn/nb_arc)
# plt.scatter(z_particles, delta_particles, c=particles.at_turn/nb_arc)
# plt.xlabel('z [m]')
# plt.ylabel('$\delta$')
# cb = plt.colorbar()
# cb.set_label('Lost at turn')
# plt.show()

# Create a figure and subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
# ax1.hist(delta0, bins=20, color='skyblue', edgecolor='black')
# ax1.set_title( 'Delta')
# ax1.set_xlabel('Delta')
# ax1.grid(True)
# ax2.hist(z0, bins=20, color='salmon', edgecolor='black')
# ax2.set_title('Zeta ')
# ax2.set_xlabel('z [mm]')
# ax2.grid(True)
# plt.tight_layout()
# plt.show()

#Plot longitudinal phase space (50st particles on last 400 turns)
plt.figure()
plt.scatter(rec.zeta[:50,-400:], rec.delta[:50,-400:], color='C0',s=2) #[mask][:, -1000:]
plt.xlabel('z [m]')
plt.ylabel('$\delta$')
plt.show()

#Normalisation 
# def to_norm(x, px, y, py):
#     x_norm = x / np.sqrt(beta_x_end)
#     px_norm = x *alf_x_end/np.sqrt(beta_x_end) + np.sqrt(beta_x_end)*px
#     y_norm = y / np.sqrt(beta_y_end)
#     py_norm = y *alf_y_end/np.sqrt(beta_y_end) + np.sqrt(beta_y_end)*py
#     return (x_norm, px_norm, y_norm, py_norm)

# x_norm, px_norm, y_norm, py_norm = to_norm(rec.x[mask], rec.px[mask], rec.y[mask], rec.py[mask])
zeta=rec.zeta
delta=rec.delta

#For separatrix
turn_number=0
mon_data=line_ds['mon']
def calc_separatrix(turn_number,label, color):
    print(turn_number)
    particles_gamma = mon_data.gamma0[:, turn_number][mask][0]
    particle_energy = mon_data.p0c[:, turn_number][mask][0]
    zeta_sep = mon_data.zeta[:, turn_number]
    delta_sep = mon_data.delta[:, turn_number]
    #RF bucket defined here on RCS => careful with consistency!!
    machine_rf_bucket = xp.longitudinal.rf_bucket.RFBucket(
            circumference=C/nb_arc,
            gamma=particles_gamma, mass_kg=muon_mass_kg,
            charge_coulomb=e_charge, alpha_array=[MCF], 
            p_increment=energy_increment_per_arc * e_charge / c, #*nb_arc
            harmonic_list=[h_rf/nb_arc], voltage_list=[volt_arc], #*nb_arc
            phi_offset_list=np.atleast_1d((phase)*np.pi/180),
            z_offset=None)
    ufp = machine_rf_bucket.z_ufp
    xx = np.linspace(machine_rf_bucket.z_left, machine_rf_bucket.z_right, 1000)
    yy = machine_rf_bucket.separatrix(xx)
    # fac = 1e9/c
    fac = 1.0
    plt.figure()
    # plt.scatter(rec.zeta*fac, rec.delta*particle_energy/1e9, color='C0',s=2)    
    plt.scatter(zeta_sep*fac, delta_sep*particle_energy/1e9, s=2)
    plt.plot((xx-machine_rf_bucket.z_sfp)*fac, yy*particle_energy/1e9, color=color, label=label) #
    plt.plot((xx-machine_rf_bucket.z_sfp)*fac, -yy*particle_energy/1e9, color=color)
    plt.xlabel('z [m]')
    plt.ylabel(r'$\Delta E$ [GeV]')
    plt.xlim(-0.06,0.06)
    plt.ylim(-10,10)
    plt.legend()
    plt.show()
    return (machine_rf_bucket)

turns=[0,n_turns-1]
for i in turns:
    calc_separatrix(i,f'Turn {i}','C1')

def calc_emitt(z,delta,E):
    cov=np.cov(z,delta*E)
    emit_evm=np.sqrt(np.linalg.det(cov))
    emit_evs=emit_evm/c
    return(emit_evm*1e-6,emit_evs)

emit_evs=[]
sigma_z_tbt=[]
t_list=np.arange(0,n_turns)/nb_arc
for i in (np.arange(0,n_turns)):
    e_evm, e_evs= calc_emitt(mon_data.zeta[:, i],mon_data.delta[:, i],mon_data.p0c[:,i][mask][0])
    emit_evs.append(e_evs)
    sigma_z_tbt.append(np.std(mon_data.zeta[:, i]))

plt.figure()
plt.plot(t_list[::nb_arc],emit_evs[::nb_arc])
plt.xlabel('Turns')
plt.ylabel('$\epsilon _s \; [eVs]$')
plt.show()
plt.figure()
plt.plot(t_list[::nb_arc],np.array(sigma_z_tbt[::nb_arc])*1e3)
plt.xlabel('Turns')
plt.ylabel('$\sigma _z \; [mm]$')
plt.show()

#%%Plotting phase space, normalised (x,x') and (y,y') 
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))  
# axes[0, 0].scatter(x_norm, px_norm)
# axes[0, 0].set_xlabel('$\~{x} \; [\sqrt{m}]$')
# axes[0, 0].set_ylabel("$\~{x'} \; [\sqrt{m}]$")
# # axes[0, 0].set_ylim(-1e-4, 1e-4)
# # axes[0, 0].set_xlim(-1e-4, 1e-4)
# #axes[0, 0].axis('equal')
# axes[0, 1].scatter(y_norm, py_norm)
# axes[0, 1].set_xlabel('$\~{y} \; [\sqrt{m}]$')
# axes[0, 1].set_ylabel("$\~{y'} \; [\sqrt{m}]$")
# # axes[0, 1].set_ylim(-1e-4, 1e-4)
# # axes[0, 1].set_xlim(-1e-4, 1e-4)
# #axes[0, 1].axis('equal')
# axes[1, 0].scatter(x_norm, y_norm)
# axes[1, 0].set_xlabel('$\~{x} \; [\sqrt{m}]$')
# axes[1, 0].set_ylabel('$\~{y} \; [\sqrt{m}]$')
# # axes[1, 0].set_ylim(-1e-4, 1e-4)
# # axes[1, 0].set_xlim(-1e-4, 1e-4)
# axes[1, 1].scatter(zeta, delta)
# axes[1, 1].set_xlabel('z [m]')
# axes[1, 1].set_ylabel('$\delta$')
# for ax in axes.flat:
#     ax.ticklabel_format(style='sci', scilimits=(-3, 3), axis='both')
# plt.tight_layout()
# plt.show()

#%%Plotting phase space (x,x') and (y,y') 
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))  
# axes[0, 0].scatter(rec.x[mask], rec.px[mask],s=2)
# axes[0, 0].set_xlabel('x [m]')
# axes[0, 0].set_ylabel("x'")
# # axes[0, 0].set_ylim(-1e-4, 1e-4)
# # axes[0, 0].set_xlim(-1e-4, 1e-4)
# #axes[0, 0].axis('equal')
# axes[0, 1].scatter(rec.y[mask], rec.py[mask],s=2)
# axes[0, 1].set_xlabel('y [m]')
# axes[0, 1].set_ylabel("y'")
# # axes[0, 1].set_ylim(-1e-4, 1e-4)
# # axes[0, 1].set_xlim(-1e-4, 1e-4)
# #axes[0, 1].axis('equal')
# axes[1, 0].scatter(rec.x[mask], rec.y[mask],s=2)
# axes[1, 0].set_xlabel('x [m]')
# axes[1, 0].set_ylabel('y [m]')
# # axes[1, 0].set_ylim(-1e-4, 1e-4)
# # axes[1, 0].set_xlim(-1e-4, 1e-4)
# axes[1, 1].scatter(zeta[mask], delta[mask],s=2)
# axes[1, 1].set_xlabel('z [m]')
# axes[1, 1].set_ylabel('$\delta$')
# for ax in axes.flat:
#     ax.ticklabel_format(style='sci', scilimits=(-3, 3), axis='both')
# plt.tight_layout()
# plt.show()

#Print parameters
print('MACHINE')
print(f'E_inj: {E_inj*1e-9:.3f} GeV')
print(f'E_ext: {E_ext*1e-9:.3f} GeV')
print(f'N_turn_rcs: {N_turn_rcs}')
print(f'deltaE_turn: {energy_increment_per_arc*nb_arc*1e-9:.3f} GeV')

print('RF PARAMETERS')
print(f'f_RF: {frequency_rf*1e-9:.3f} GHz')
print(f'Phi_s: {sync_phase} °')
print(f'V_tot: {volt_arc*nb_arc*1e-9:.3f} GV')
print(f'h: {h_rf}')

print('TUNES')
print(f'RCS Qx: {tw_ds["qx"]*nb_arc:.3f}')
print(f'Arc Qx: {tw_ds["qx"]:.3f}')
print(f'RCS Qy: {tw_ds["qy"]*nb_arc:.3f}')
print(f'Arc Qy: {tw_ds["qy"]:.3f}')
print(f'Arc Qs: {tw_ds["qs"]:.4f}')
print(f'RCS Qs: {tw_ds["qs"]*nb_arc:.3f}')
print(f'MCF: {MCF:.5f}')

print('TRACKING')
print(f'n_particles: {n_particles}')
print(f'emit_s_inj: {emit_evs[0]:.5f} eVs')
print(f'sigma_z_inj: {sigma_z_tbt[0]*1e3:.3f} mm')
print(f'emit_s_ext: {emit_evs[-1]:.5f} eVs')
print(f'sigma_z_ext: {sigma_z_tbt[-1]*1e3:.3f} mm')
#%% Optics
# plt.figure()
# plt.plot(tw_ds['s'], tw_ds['betx'], label='$\\beta_x$')
# plt.plot(tw_ds['s'], tw_ds['bety'], label='$\\beta_y$')
# # for i, el in enumerate(line_ds.element_names):
# #     if 'quad' in el:
# #         plt.axvline(x=tw_ds['s'][i], color='grey', linestyle='--')
# #     elif 'sxt' in el:
# #         plt.axvline(x=tw_ds['s'][i], color='red', linestyle='--')
# plt.xlabel('s [m]')
# plt.ylabel('$\\beta_x$, $\\beta_y$ [m]')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(tw_ds['s'], tw_ds['dx'], label='$D_x$')
# plt.plot(tw_ds['s'], tw_ds['dy'], label='$D_y$')
# # for i, el in enumerate(line_ds.element_names):
# #     if 'quad' in el:
# #         plt.axvline(x=tw_ds['s'][i], color='grey', linestyle='--')
# #     elif 'sxt' in el:
# #         plt.axvline(x=tw_ds['s'][i], color='red', linestyle='--')
# plt.xlabel('s [m]')
# plt.ylabel('$D_x$, $D_y$ [m]')
# plt.legend()
# plt.show()

# #Plot on same figure
# fig, ax1 = plt.subplots(figsize=(12, 6))
# beta_plot1, = ax1.plot(tw_ds['s'], tw_ds['betx'], label='$\\beta_x$')
# beta_plot2, = ax1.plot(tw_ds['s'], tw_ds['bety'], label='$\\beta_y$')
# ax1.set_xlabel('s [m]')
# ax1.set_ylabel('$\\beta_x$, $\\beta_y$ [m]')
# ax1.tick_params(axis='y')
# ax2 = ax1.twinx()
# D_plot, = ax2.plot(tw_ds['s'], tw_ds['dx'], color='tab:green', label='$D_x$')
# ax2.set_ylabel('$D_x$ [m]')
# ax2.tick_params(axis='y')
# plots = [beta_plot1, beta_plot2, D_plot]
# labels = [plot.get_label() for plot in plots]
# ax1.legend(plots, labels, loc='upper left')
# plt.show()


