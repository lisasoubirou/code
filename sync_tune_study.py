# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:10:49 2024
@author: LS276867

Study of the synchrotron tune. Shows discrepancy between what is predicted by xsuite and
what we calculate manually. Can be seen visually. 
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
volt_arc=max(energy_increment_per_arc/np.sin(phase*np.pi/180), 100e6)
volt_cav=volt_arc/n_cav_arc
# nemitt_x=3e-6
# nemitt_y=3e-6
emit_s=0.025 #eVs
sigma_z=9.5*1e-3
n_particles=5000
n_monitored=5000

#Importing line
file_seq = 'lattice_disp_suppr.json'
line_ds=xt.Line.from_json(file_seq)
line_ds.particle_ref = xt.Particles(mass0=xp.MUON_MASS_EV, q0=1.,
                                    energy0=xp.MUON_MASS_EV + E_inj,
                                 )
tw_ds_bf=line_ds.twiss(method='4d')
method = "6d" 

#Cavity settings
frequency = np.round(frequency_rf*tw_ds_bf.T_rev0)/tw_ds_bf.T_rev0
line_ds['cavity'].frequency=frequency
line_ds['cavity'].lag=phase
line_ds['cavity'].voltage=volt_cav
line_ds['cavity_1'].frequency=frequency
line_ds['cavity_1'].lag=phase
line_ds['cavity_1'].voltage=volt_cav

#%% Optics
#Twiss
tab = line_ds.get_table()

#Accelerating elements
h_rf = np.round(frequency_rf*tw_ds_bf.T_rev0*nb_arc) #data tw_ds on 1 arc
RF_acc=xt.ReferenceEnergyIncrease(Delta_p0c=energy_increment_per_arc/n_cav_arc)
tw_ds=line_ds.twiss(method=method)
line_ds.discard_tracker()
line_ds.insert_element('RF_acc', RF_acc, at_s=tab['s','cavity'])
line_ds.insert_element('RF_acc_1', RF_acc, at_s=tab['s','cavity_1'])
mon=xt.ParticlesMonitor(start_at_turn=0,
                        stop_at_turn = n_turns,
                        num_particles=1)
mon_dis=xt.ParticlesMonitor(start_at_turn=0,
                        stop_at_turn = n_turns,
                        num_particles=1)
line_ds.insert_element('mon', mon, at_s=tab['s','end'])
line_ds.insert_element('mon_dis',mon_dis, at_s=tab['s','quad_d_3'])
line_ds.build_tracker()

tw_ds=line_ds.twiss(method=method, matrix_stability_tol=5e-3)
MCF=tw_ds['momentum_compaction_factor']

turn_number=0
particle_energy=E_inj
particle_gamma=E_inj/xp.MUON_MASS_EV

machine_rf_bucket = xp.longitudinal.rf_bucket.RFBucket(
        circumference=C/nb_arc,
        gamma=particle_gamma, mass_kg=muon_mass_kg,
        charge_coulomb=e_charge, alpha_array=[MCF], 
        p_increment=energy_increment_per_arc * e_charge / c, #*nb_arc
        harmonic_list=[h_rf/nb_arc], voltage_list=[volt_arc], #*nb_arc
        phi_offset_list=np.atleast_1d((phase)*np.pi/180),
        z_offset=None)
ufp = machine_rf_bucket.z_ufp
xx = np.linspace(machine_rf_bucket.z_left, machine_rf_bucket.z_right, 1000)
yy = machine_rf_bucket.separatrix(xx)

delta=1e-3
fac=30
particles=line_ds.build_particles(
            zeta=(machine_rf_bucket.z_right-machine_rf_bucket.z_sfp)/fac,
            delta=delta
            )

print('Zeta, delta', particles.zeta, particles.delta)
print('X, px', particles.x, particles.px)
print('Y, py', particles.y, particles.py)
line_ds.track(particles, num_turns=n_turns, turn_by_turn_monitor=True)
rec=line_ds.record_last_track

print('TUNES')
print(f'RCS Qx: {tw_ds["qx"]*nb_arc:.3f}')
print(f'Arc Qx: {tw_ds["qx"]:.3f}')
print(f'RCS Qy: {tw_ds["qy"]*nb_arc:.3f}')
print(f'Arc Qy: {tw_ds["qy"]:.3f}')
print(f'Arc Qs: {tw_ds["qs"]:.4f}')
print(f'RCS Qs: {tw_ds["qs"]*nb_arc:.3f}')
print(f'MCF: {MCF:.5f}')

nb_part=0
mon_dis_data=line_ds['mon_dis']
mon=line_ds['mon']
x=mon_dis_data.x[nb_part]
px=mon_dis_data.px[nb_part]
y=mon_dis_data.y[nb_part]
py=mon_dis_data.py[nb_part]
z=mon_dis_data.zeta[nb_part]
ps_x=np.abs(np.fft.fft(x))**2
ps_y=np.abs(np.fft.fft(y))**2
ps_z=np.abs(np.fft.fft(z))**2
freq=np.fft.fftfreq(x.size, 1)
peak_index_x = np.argmax(ps_x)
peak_index_y = np.argmax(ps_y)
peak_index_z = np.argmax(ps_z)
tune_x=freq[peak_index_x]
tune_y=freq[peak_index_y]
tune_z=freq[peak_index_z]

ps_x_pos=ps_x[freq>0]
freq_pos=freq[freq>0]
peak_index_x = np.argmax(ps_x_pos[freq_pos<0.1])
tune_x_sync=freq_pos[freq_pos<0.1][peak_index_x]

print( 'Peak synchrotron in x spectra',tune_x_sync) 
print('Qs from tracking', tune_z)


plt.scatter(rec.zeta, rec.delta*particle_energy/1e9, s=2)
plt.plot((xx-machine_rf_bucket.z_sfp), yy*particle_energy/1e9, color='C1') 
plt.plot((xx-machine_rf_bucket.z_sfp), -yy*particle_energy/1e9, color='C1')
plt.xlabel('z [m]')
plt.ylabel(r'$\Delta E$ [GeV]')
plt.xlim(-0.06,0.06)
plt.ylim(-10,10)
plt.legend()
plt.show()
plt.figure()
plt.plot(freq,ps_x, label='fft_x')
plt.xlim(-0.01,0.01)
plt.legend()
plt.title('Tune')
plt.show()
plt.figure()
plt.plot(freq,ps_z, label='fft_z')
plt.xlim(-0.01,0.01)
plt.legend()
plt.title('Tune')
plt.show()
