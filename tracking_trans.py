# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:10:49 2024
@author: LS276867

This code allows studies of tracking in the transverse plane. 
Dynamic aperture tools are developped. 
Three functions are defined:
    - grid : create a grib of transverse coordinates
    - to_norm : normalize coordinates with twiss functions
    - plot_pha_spa : plot phase space evolution for a diagonal of the grid
Routine to plot space phase
"""

import numpy as np
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt
import xpart as xp
import json
import sys

#Importing line
file_seq = 'lattice_disp_suppr.json'
line_ds=xt.Line.from_json(file_seq)
tw_ds_bf=line_ds.twiss(method='4d')
method = "4d" 

frequency_ref=1300*1e6
frequency = np.round(frequency_ref*tw_ds_bf.T_rev0)/tw_ds_bf.T_rev0
phase=180
volt_tot = 11.22e9
n_cells = 26
n_cav_cell = 2 
volt=volt_tot/n_cells/n_cav_cell
line_ds['cavity'].frequency=frequency
line_ds['cavity'].lag=phase
line_ds['cavity'].voltage=volt
line_ds['cavity_1'].frequency=frequency
line_ds['cavity_1'].lag=phase
line_ds['cavity_1'].voltage=volt

#Add skew components
#line_ds['sxt_d'].ksl[2]=-0.1
#line_ds['quad_d_2'].ksl[1]=1e-6

#Parameters for new matching
betx_fodo=tw_ds_bf.rows['quad_d_3']['betx'][0]
bety_fodo=tw_ds_bf.rows['quad_d_3']['bety'][0]
dx_fodo=tw_ds_bf.rows['quad_d_3']['dx'][0]
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

tw_ds=line_ds.twiss(method=method)
#Twiss at end of line 
beta_x_end=tw_ds['betx'][-1]
beta_y_end=tw_ds['bety'][-1]
alf_x_end=tw_ds['alfx'][-1]
alf_y_end=tw_ds['alfy'][-1]
#%% Generating particles 
def grid(sigma_z, delta, n_x, n_y):
    x=np.linspace(sigma_z, n_r)
    y=np.linspace(delta, n_theta)
    xx, yy = np.meshgrid(x, y)
    xx=np.ravel(xx)
    yy=np.ravel(yy)
    return(xx,yy)

n_part=5
nemitt_x=9e-6
nemitt_y=9e-6
n_r = 10
n_theta = 10
sigma_max=150
sigma_z = 1e-2 #RMS bunch length [m]
delta=1e-3

# sigma_z=1
# delta=2

list_zeta,list_delta=grid(sigma_z, delta, n_r, n_theta)

#Generating grid
x_normalized, y_normalized, r_xy, theta_xy = xp.generate_2D_polar_grid(
    r_range=(0, sigma_max), 
    theta_range=(0, np.pi/2),
    nr=n_r, ntheta=n_theta)
particles = line_ds.build_particles(
    x_norm=x_normalized, px_norm=0,
    y_norm=y_normalized, py_norm=0,
    nemitt_x=nemitt_x, nemitt_y=nemitt_y, 
    delta=0,
    method=method)

#Matched distribution
# particles = xp.generate_matched_gaussian_bunch(
#          num_particles=n_part,
#          nemitt_x=nemitt_x, nemitt_y=nemitt_y, 
#          sigma_z=sigma_z,
#          line=line_ds,
#          #delta=0,
#          )

# particles = line_ds.build_particles(
#     x=0, px=0,
#     y=0, py=0,
#     zeta=list_zeta,
#     delta=list_delta,
#     method=method)

#Initial particle distribution
x0=np.array(particles.x)
y0=np.array(particles.y)

# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.4, 7))
# plt.title('Particle distribution')
# axes[0].plot(particles.x, particles.px, '.')
# axes[0].set_xlabel('x [mm]')
# axes[0].set_ylabel('px/p0')
# axes[1].plot(particles.y, particles.py, '.')
# axes[1].set_xlabel('y [mm]')
# axes[1].set_ylabel('py/p0')
# plt.tight_layout() 
# plt.show()

# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.4, 7))
# plt.title('Particle distribution')
# axes[0].plot(particles.x, particles.px, '.')
# axes[0].set_xlabel('x')
# axes[0].set_ylabel('px/p0')
# axes[1].plot(particles.y, particles.py, '.')
# axes[1].set_xlabel('y [mm]')
# axes[1].set_ylabel('py')
# plt.tight_layout() 
# plt.show()

#%% Tracking 
n_turns = 1500
line_ds.track(particles, num_turns=n_turns, 
              turn_by_turn_monitor=True,
              #freeze_longitudinal=True
              )
particles.sort(interleave_lost_particles=True)

xx = np.ravel(line_ds.record_last_track.x).reshape((n_r, n_theta, n_turns))
pxx = np.ravel(line_ds.record_last_track.px).reshape((n_r, n_theta, n_turns))
yy = np.ravel(line_ds.record_last_track.y).reshape((n_r, n_theta, n_turns))
pyy = np.ravel(line_ds.record_last_track.py).reshape((n_r, n_theta, n_turns))
zz = np.ravel(line_ds.record_last_track.zeta).reshape((n_r, n_theta, n_turns))
dd = np.ravel(line_ds.record_last_track.delta).reshape((n_r, n_theta, n_turns))

mm = np.all(np.ravel(line_ds.record_last_track.state).reshape((n_r, n_theta, n_turns)), axis=2)

#Print dynamic aperture
plt.figure(1)
plt.scatter(x_normalized, y_normalized, c=particles.at_turn)
plt.xlabel(r'$A_x [\sigma]$')
plt.ylabel(r'$A_y [\sigma]$')
cb = plt.colorbar()
cb.set_label('Lost at turn')

# plt.figure(2)
# plt.pcolormesh(
#     x_normalized.reshape(n_r, n_theta), y_normalized.reshape(n_r, n_theta),
#     particles.at_turn.reshape(n_r, n_theta), shading='gouraud')
# plt.xlabel(r'$A_x [\sigma]$')
# plt.ylabel(r'$A_y [\sigma]$')
# ax = plt.colorbar()
# ax.set_label('Lost at turn')

# plt.figure(3)
# plt.scatter(x0*1e3, y0*1e3, c=particles.at_turn)
# plt.xlabel('x [mm]')
# plt.ylabel('y [mm]')
# cb = plt.colorbar()
# cb.set_label('Lost at turn')
# plt.show()

#Normalisation 
mask= particles.state > 0

def to_norm(x, px, y, py):
    x_norm = x / np.sqrt(beta_x_end)
    px_norm = x *alf_x_end/np.sqrt(beta_x_end) + np.sqrt(beta_x_end)*px
    y_norm = y / np.sqrt(beta_y_end)
    py_norm = y *alf_y_end/np.sqrt(beta_y_end) + np.sqrt(beta_y_end)*py
    return (x_norm, px_norm, y_norm, py_norm)

x_norm=line_ds.record_last_track.x[mask] / np.sqrt(beta_x_end)
px_norm=line_ds.record_last_track.x[mask] *alf_x_end/np.sqrt(beta_x_end) + np.sqrt(beta_x_end)*line_ds.record_last_track.px[mask]
y_norm=line_ds.record_last_track.y[mask] / np.sqrt(beta_y_end)
py_norm=line_ds.record_last_track.y[mask] *alf_y_end/np.sqrt(beta_y_end) + np.sqrt(beta_y_end)*line_ds.record_last_track.py[mask]

# zeta=line_ds.record_last_track.zeta[mask]
# delta=line_ds.record_last_track.delta[mask]

#Plotting phase space, normalised (x,x') and (y,y') 
def plot_pha_spa(n):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))  
    mask = mm[:,n]  
    xx_norm, ppx_norm, yy_norm, ppy_norm = to_norm(xx[:,n], pxx[:,n], yy[:,n], pyy[:,n])

    axes[0, 0].scatter(xx_norm[mask], ppx_norm[mask])
    axes[0, 0].set_xlabel('$\~{x} \; [\sqrt{m}]$')
    axes[0, 0].set_ylabel("$\~{x'} \; [\sqrt{m}]$")
    # axes[0, 0].set_ylim(-1e-4, 1e-4)
    # axes[0, 0].set_xlim(-1e-4, 1e-4)
    #axes[0, 0].axis('equal')
    axes[0, 1].scatter(yy_norm[mask], ppy_norm[mask])
    axes[0, 1].set_xlabel('$\~{y} \; [\sqrt{m}]$')
    axes[0, 1].set_ylabel("$\~{y'} \; [\sqrt{m}]$")
    # axes[0, 1].set_ylim(-1e-4, 1e-4)
    # axes[0, 1].set_xlim(-1e-4, 1e-4)
    #axes[0, 1].axis('equal')
    axes[1, 0].scatter(xx_norm[mask], yy_norm[mask])
    axes[1, 0].set_xlabel('$\~{x} \; [\sqrt{m}]$')
    axes[1, 0].set_ylabel('$\~{y} \; [\sqrt{m}]$')
    # axes[1, 0].set_ylim(-1e-4, 1e-4)
    # axes[1, 0].set_xlim(-1e-4, 1e-4)
    axes[1, 1].scatter(zz[:,n][mask], dd[:,n][mask])
    axes[1, 1].set_xlabel('z [m]')
    axes[1, 1].set_ylabel('$\delta$')
    for ax in axes.flat:
        ax.ticklabel_format(style='sci', scilimits=(-3, 3), axis='both')
    plt.tight_layout()
    plt.show()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))  
axes[0, 0].scatter(x_norm, px_norm)
axes[0, 0].set_xlabel('$\~{x} \; [\sqrt{m}]$')
axes[0, 0].set_ylabel("$\~{x'} \; [\sqrt{m}]$")
# axes[0, 0].set_ylim(-1e-4, 1e-4)
# axes[0, 0].set_xlim(-1e-4, 1e-4)
#axes[0, 0].axis('equal')
axes[0, 1].scatter(y_norm, py_norm)
axes[0, 1].set_xlabel('$\~{y} \; [\sqrt{m}]$')
axes[0, 1].set_ylabel("$\~{y'} \; [\sqrt{m}]$")
# axes[0, 1].set_ylim(-1e-4, 1e-4)
# axes[0, 1].set_xlim(-1e-4, 1e-4)
#axes[0, 1].axis('equal')
axes[1, 0].scatter(x_norm, y_norm)
axes[1, 0].set_xlabel('$\~{x} \; [\sqrt{m}]$')
axes[1, 0].set_ylabel('$\~{y} \; [\sqrt{m}]$')
# axes[1, 0].set_ylim(-1e-4, 1e-4)
# axes[1, 0].set_xlim(-1e-4, 1e-4)
axes[1, 1].scatter(zeta, delta)
axes[1, 1].set_xlabel('z [m]')
axes[1, 1].set_ylabel('$\delta$')
for ax in axes.flat:
    ax.ticklabel_format(style='sci', scilimits=(-3, 3), axis='both')
plt.tight_layout()
plt.show()

# range_n=[0,9]
# for i in range_n:
#     plot_pha_spa(i)

#FFT to determine tune 
ps_x=np.abs(np.fft.fft(x_norm[11]-1j*px_norm[11]))**2
ps_y=np.abs(np.fft.fft(y_norm[11]-1j*py_norm[11]))**2
freq=np.fft.fftfreq(x_norm[11].size, 1)
peak_index_x = np.argmax(ps_x)
peak_index_y = np.argmax(ps_y)
tune_x=freq[peak_index_x]
tune_y=freq[peak_index_y]

# plt.figure()
# plt.plot(freq,ps_x, label='fft_x')
# plt.plot(freq,ps_y, label='fft_y')
# plt.legend()
# plt.title('Tune')
# plt.show()

print('TUNE')
print('Match Qx',tw_ds['qx'])
print('Tracking tune Qx', tune_x)
print('Match Qy',tw_ds['qy'])
print('Tracking tune Qy', tune_y)
nb_arc=26
print('Arc Qs', tw_ds['qs'])
print('RCS Qs', tw_ds['qs']*nb_arc)

#%% Optics
# plt.figure()
# plt.plot(tw_ds['s'], tw_ds['betx'], label='betx')
# plt.plot(tw_ds['s'], tw_ds['bety'], label='bety')
# for i, el in enumerate(line_ds.element_names):
#     if 'quad' in el:
#         plt.axvline(x=tw_ds['s'][i], color='grey', linestyle='--')
#     elif 'sxt' in el:
#         plt.axvline(x=tw_ds['s'][i], color='red', linestyle='--')
# plt.xlabel('s [m]')
# plt.ylabel('betx, bety [m]')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(tw_ds['s'], tw_ds['dx'], label='Dx')
# plt.plot(tw_ds['s'], tw_ds['dy'], label='Dy')
# for i, el in enumerate(line_ds.element_names):
#     if 'quad' in el:
#         plt.axvline(x=tw_ds['s'][i], color='grey', linestyle='--')
#     elif 'sxt' in el:
#         plt.axvline(x=tw_ds['s'][i], color='red', linestyle='--')
# plt.xlabel('s [m]')
# plt.ylabel('Dx, Dy [m]')
# plt.legend()
# plt.show()
