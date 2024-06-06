# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:54:27 2022

@author: Achance
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import constants as cons

from gener_pulsed_synchrotron import PulsedSynch

RCS_LE_ref = PulsedSynch("para_RCS_LE.txt")
RCS_ME_ref = PulsedSynch("para_RCS_ME.txt")
RCS_HE_ref = PulsedSynch("para_RCS_HE.txt")

rc("figure", dpi=200)
rc("legend", columnspacing=1.2, handletextpad=0.4, 
   handlelength=2, labelspacing=0.4, borderaxespad=0.4)

# target_ramping = np.arange(0.2,3.1,0.025)*1e-3
# target_ramping_LE = np.arange(0.1,3.1,0.025)*1e-3
# target_ramping_ME = np.arange(0.2,6.1,0.025)*1e-3
# target_ramping_HE = np.arange(0.2,10.1,0.025)*1e-3

# RCS_LE = PulsedSynch("para_RCS_LE.txt", max_gradient=10e9, 
#                      target_ramping=target_ramping_LE,
#                      max_NC_slope=10e4)
# RCS_ME = PulsedSynch("para_RCS_ME.txt", max_gradient=10e9, 
#                      target_ramping=target_ramping_ME,
#                      max_NC_slope=10e4)
# RCS_HE = PulsedSynch("para_RCS_HE.txt", max_gradient=10e9, 
#                      target_ramping=target_ramping_HE,
#                      max_NC_slope=10e4)
# # RCS_LE.print_all()

# fig, ax1 = plt.subplots() 

# ax1.set_xlabel('Ramp time [ms]') 
# ax1.set_ylabel('Total RF voltage [GV]') 
# plot1 = []
# for RCS, RCS_ref, color, label in zip(
#         [RCS_LE, RCS_ME, RCS_HE], 
#         [RCS_LE_ref, RCS_ME_ref, RCS_HE_ref], 
#         ["r-","b-","g-"],
#         ["RCS_LE", "RCS_ME", "RCS_HE"]):  
#     plot1.append(ax1.plot(RCS.ramp_time*1e3, 
#                           RCS.tot_RF_voltage*1e-9, color, label=label)[0])
#     x, y = RCS_ref.ramp_time*1e3, RCS_ref.tot_RF_voltage*1e-9
#     ax1.plot([x, x, 0], [0, y, y], color="k", linestyle="dotted")
#     ax1.plot([x], [y], "o", color="orange")
# # ax1.tick_params(axis ='y', labelcolor = 'red') 

# # Adding Twin Axes
# ax1.set_xlim(0, 3)
# ax1.set_ylim(0, 100)
# ax2 = ax1.twinx() 
  
# ax2.set_ylabel('Average RF gradient [MV/m]', color = 'blue') 
# plot2 = []
# for RCS, RCS_ref, color, label in zip(
#         [RCS_LE, RCS_ME, RCS_HE], 
#         [RCS_LE_ref, RCS_ME_ref, RCS_HE_ref], 
#         ["r--","b--","g--"],
#         ["gradient RCS_LE", "gradient RCS_ME", "gradient RCS_HE"]):  
#     plot2.append(ax2.plot(RCS.ramp_time*1e3, 
#                           RCS.tot_RF_voltage/RCS.RF_length_tot*1e-6, 
#                           color, label=label)[0])
#     x, y = RCS_ref.ramp_time*1e3, RCS_ref.tot_RF_voltage/RCS_ref.RF_length_tot*1e-6
#     ax2.plot([x, x, 3], [0, y, y], color="cyan", linestyle="dotted")
#     ax2.plot([x], [y], "o", color="orange")
# ax2.set_ylim(0, 200)
# ax2.tick_params(axis ='y', labelcolor = 'blue') 

# # Add legends

# lns = plot1 + plot2
# labels = [l.get_label() for l in lns]
# plt.legend(lns, labels, loc=0)
 
# # Show plot

# plt.show()    

# fig, ax1 = plt.subplots() 

# ax1.set_xlabel('Ramp time [ms]') 
# ax1.set_ylabel('NC dipole field slope [kT/s]') 
# plot1 = []
# for RCS, RCS_ref, color, label in zip(
#         [RCS_LE, RCS_ME, RCS_HE], 
#         [RCS_LE_ref, RCS_ME_ref, RCS_HE_ref], 
#         ["r-","b-","g-"],
#         ["RCS_LE", "RCS_ME", "RCS_HE"]):  
#     plot1.append(ax1.plot(RCS.ramp_time*1e3, 
#                           RCS.peak_NC_slope*1e-3, 
#                           color, label=label)[0])
#     x, y = RCS_ref.ramp_time*1e3, RCS_ref.peak_NC_slope*1e-3
#     ax1.plot([x, x, 0], [0, y, y], color="k", linestyle="dotted")
#     ax1.plot([x], [y], "o", color="orange")
# # ax1.tick_params(axis ='y', labelcolor = 'red') 

# # Adding Twin Axes
# ax1.set_xlim(0, 3)
# ax1.set_ylim(0, 10)
# ax2 = ax1.twinx() 
  
# ax2.set_ylabel('Dipole peak power [GW]', color = 'blue') 
# plot2 = []
# for RCS, RCS_ref, color, label in zip(
#         [RCS_LE, RCS_ME, RCS_HE], 
#         [RCS_LE_ref, RCS_ME_ref, RCS_HE_ref], 
#         ["r--","b--","g--"],
#         ["RCS_LE", "RCS_ME", "RCS_HE"]):  
#     plot2.append(ax2.plot(RCS.ramp_time*1e3, 
#                           RCS.peak_NC_power*1e-9, 
#                           color, label="power "+label)[0])
#     x, y = RCS_ref.ramp_time*1e3, RCS_ref.peak_NC_power*1e-9
#     ax2.plot([x, x, 3], [0, y, y], color="cyan", linestyle="dotted")
#     ax2.plot([x], [y], "o", color="orange")
# ax2.set_ylim(0, 100)
# ax2.tick_params(axis ='y', labelcolor = 'blue') 

# # Add legends

# lns = plot1 + plot2
# labels = [l.get_label() for l in lns]
# plt.legend(lns, labels, loc=0)
 
# # Show plot

# plt.show()    

# for RCS, RCS_ref, color, label in zip(
#         [RCS_LE, RCS_ME, RCS_HE], 
#         [RCS_LE_ref, RCS_ME_ref, RCS_HE_ref], 
#         ["r-","b-","g-"],
#         ["RCS_LE", "RCS_ME", "RCS_HE"]):  
#     plt.plot(RCS.ramp_time*1e3, RCS.muon_survival*100, color, label=label)
#     x, y = RCS_ref.ramp_time*1e3, RCS_ref.muon_survival*100 
#     plt.plot([x, x, 0], [0, y, y], color="k", linestyle="dotted")
#     plt.plot([x], [y], "o", color="orange")        
# plt.xlim(0, 3)
# plt.ylim(0, 100)
# plt.xlabel("Ramp time [ms]")
# plt.ylabel("Muon survival [%]")
# plt.legend()
# plt.show()

# for RCS, RCS_ref, color, label in zip(
#         [RCS_LE, RCS_ME, RCS_HE], 
#         [RCS_LE_ref, RCS_ME_ref, RCS_HE_ref], 
#         ["r-","b-","g-"],
#         ["RCS_LE", "RCS_ME", "RCS_HE"]):  
#     plt.plot(RCS.ramp_time*1e3, RCS.inj_Qs, color, label="inj "+label)
#     plt.plot(RCS.ramp_time*1e3, RCS.ext_Qs, color+"-", label="ext "+label)
#     x, y = RCS_ref.ramp_time*1e3, RCS_ref.inj_Qs 
#     plt.plot([x, x, 0], [0, y, y], color="k", linestyle="dotted")
#     plt.plot([x], [y], "o", color="orange")
# plt.xlim(0, 3)
# plt.ylim(0, 2)
# plt.axhline(1/np.pi, ls="-.", color="k")
# plt.text(0.2,1/np.pi+0.05, r"1/$\pi$", fontsize=14)
# plt.xlabel("Ramp time [ms]")
# plt.ylabel(r"Synchrotron tune $Q_s$")
# plt.legend()
# plt.show()

# fig, ax1 = plt.subplots() 

# ax1.set_xlabel('Muon survival [%]') 
# ax1.set_ylabel('Total RF voltage [GV]') 
# plot1 = []
# for RCS, color, label in zip(
#         [RCS_LE, RCS_ME, RCS_HE], ["r-","b-","g-"],
#         ["RCS_LE", "RCS_ME", "RCS_HE"]):  
#     plot1.append(ax1.plot(RCS.muon_survival*100, 
#                           RCS.tot_RF_voltage*1e-9, color, label=label)[0])
# # ax1.tick_params(axis ='y', labelcolor = 'red') 

# # Adding Twin Axes
# ax1.set_xlim(60, 100)
# ax1.set_ylim(0, 100)
# ax2 = ax1.twinx() 
  
# ax2.set_ylabel('Average RF gradient [MV/m]', color = 'blue') 
# plot2 = []
# for RCS, color, label in zip(
#         [RCS_LE, RCS_ME, RCS_HE], ["r--","b--","g--"],
#         ["gradient RCS_LE", "gradient RCS_ME", "gradient RCS_HE"]):  
#     plot2.append(ax2.plot(RCS.muon_survival*100, 
#                           RCS.tot_RF_voltage/RCS.RF_length_tot*1e-6, 
#                           color, label=label)[0])
# ax2.set_ylim(0, 200)
# ax2.tick_params(axis ='y', labelcolor = 'blue') 

# # Add legends

# lns = plot1 + plot2
# labels = [l.get_label() for l in lns]
# plt.legend(lns, labels, loc=0)
 
# # Show plot

# plt.axvline(90, color="k")
# plt.show()    

# fig, ax1 = plt.subplots() 

# ax1.set_xlabel('Muon survival [%]') 
# ax1.set_ylabel('NC dipole field slope [kT/s]') 
# plot1 = []
# for RCS, color, label in zip(
#         [RCS_LE, RCS_ME, RCS_HE], ["r-","b-","g-"],
#         ["RCS_LE", "RCS_ME", "RCS_HE"]):  
#     plot1.append(ax1.plot(RCS.muon_survival*100, 
#                           RCS.peak_NC_slope*1e-3, 
#                           color, label=label)[0])
# # ax1.tick_params(axis ='y', labelcolor = 'red') 

# # Adding Twin Axes
# ax1.set_xlim(60, 100)
# ax1.set_ylim(0, 10)
# ax2 = ax1.twinx() 
  
# ax2.set_ylabel('Dipole peak power [GW]', color = 'blue') 
# plot2 = []
# for RCS, color, label in zip(
#         [RCS_LE, RCS_ME, RCS_HE], ["r--","b--","g--"],
#         ["RCS_LE", "RCS_ME", "RCS_HE"]):  
#     plot2.append(ax2.plot(RCS.muon_survival*100, 
#                           RCS.peak_NC_power*1e-9, 
#                           color, label="power "+label)[0])
# ax2.set_ylim(0, 100)
# ax2.tick_params(axis ='y', labelcolor = 'blue') 

# # Add legends

# lns = plot1 + plot2
# labels = [l.get_label() for l in lns]
# plt.legend(lns, labels, loc=0)
# plt.axvline(90, color="k")
 
# # Show plot

# plt.show()    

# for RCS, color, label in zip(
#         [RCS_LE, RCS_ME, RCS_HE], ["r-","b-","g-"],
#         ["RCS_LE", "RCS_ME", "RCS_HE"]):  
#     plt.plot(RCS.muon_survival*100, RCS.ramp_time*1e3, color, label=label)
# plt.xlim(60, 100)
# plt.ylabel("Ramp time [ms]")
# plt.xlabel("Muon survival [%]")
# plt.axvline(90, color="k")
# plt.legend()
# plt.show()

# for RCS, color, label in zip(
#         [RCS_LE, RCS_ME, RCS_HE], ["r-","b-","g-"],
#         ["RCS_LE", "RCS_ME", "RCS_HE"]):  
#     plt.plot(RCS.muon_survival*100, RCS.inj_Qs, color, label="inj "+label)
#     plt.plot(RCS.muon_survival*100, RCS.ext_Qs, color+"-", label="ext "+label)
# plt.xlim(60, 100)
# plt.ylim(0, 2)
# plt.axhline(1/np.pi, ls="-.", color="k")
# plt.text(65,1/np.pi+0.05, r"1/$\pi$", fontsize=14)
# plt.xlabel("Muon survival [%]")
# plt.ylabel(r"Synchrotron tune $Q_s$")
# plt.axvline(90, color="k")
# plt.legend()
# plt.show()

E_ext_LE = np.linspace(2, 5, 101)*63e9
E_ext_ME = 750e9+(E_ext_LE-300e9)*(10-1.5)/(10+1.5)
RCS_LE = PulsedSynch("para_RCS_LE.txt", ext_E=E_ext_LE, 
                     filling_ratio=0.85+2*np.pi/1.5/cons.c/5990.*(E_ext_LE-300e9),
                     target_ramping=0.01e-3, max_NC_slope=1e5)
RCS_ME = PulsedSynch("para_RCS_ME.txt", inj_E=E_ext_LE, 
                     ext_E=E_ext_ME)
RCS_HE = PulsedSynch("para_RCS_HE.txt", inj_E=E_ext_ME)

fig, ax1 = plt.subplots() 

ax1.set_xlabel('Extraction energy of RCS LE [GeV]') 
ax1.set_ylabel('Total RF voltage [GV]') 
plot1 = []
for RCS, RCS_ref, color, label in zip(
        [RCS_LE, RCS_ME, RCS_HE], 
        [RCS_LE_ref, RCS_ME_ref, RCS_HE_ref], 
        ["r-","b-","g-"],
        ["RCS_LE", "RCS_ME", "RCS_HE"]):  
    plot1.append(ax1.plot(E_ext_LE*1e-9, 
                          RCS.tot_RF_voltage*1e-9, color, label=label)[0])
# ax1.tick_params(axis ='y', labelcolor = 'red') 

# Adding Twin Axes
ax1.set_xlim(E_ext_LE[0]*1e-9, E_ext_LE[-1]*1e-9)
ax1.set_ylim(0, 100)
ax2 = ax1.twinx() 
  
ax2.set_ylabel('Average RF gradient [MV/m]', color = 'blue') 
plot2 = []
for RCS, RCS_ref, color, label in zip(
        [RCS_LE, RCS_ME, RCS_HE], 
        [RCS_LE_ref, RCS_ME_ref, RCS_HE_ref], 
        ["r--","b--","g--"],
        ["gradient RCS_LE", "gradient RCS_ME", "gradient RCS_HE"]):  
    plot2.append(ax2.plot(E_ext_LE*1e-9, 
                          RCS.tot_RF_voltage/RCS.RF_length_tot*1e-6, 
                          color, label=label)[0])
ax2.set_ylim(0, 200)
ax2.tick_params(axis ='y', labelcolor = 'blue') 

# Add legends

lns = plot1 + plot2
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc=0)
 
# Show plot

plt.show()    

fig, ax1 = plt.subplots() 

ax1.set_xlabel('Extraction energy of RCS LE [GeV]') 
ax1.set_ylabel('NC dipole field slope [kT/s]') 
plot1 = []
for RCS, RCS_ref, color, label in zip(
        [RCS_LE, RCS_ME, RCS_HE], 
        [RCS_LE_ref, RCS_ME_ref, RCS_HE_ref], 
        ["r-","b-","g-"],
        ["RCS_LE", "RCS_ME", "RCS_HE"]):  
    plot1.append(ax1.plot(E_ext_LE*1e-9, 
                          RCS.peak_NC_slope*1e-3, 
                          color, label=label)[0])
# ax1.tick_params(axis ='y', labelcolor = 'red') 

# Adding Twin Axes
ax1.set_xlim(E_ext_LE[0]*1e-9, E_ext_LE[-1]*1e-9)
ax1.set_ylim(0, 10)
ax2 = ax1.twinx() 
  
ax2.set_ylabel('Dipole peak power [GW]', color = 'blue') 
plot2 = []
for RCS, RCS_ref, color, label in zip(
        [RCS_LE, RCS_ME, RCS_HE], 
        [RCS_LE_ref, RCS_ME_ref, RCS_HE_ref], 
        ["r--","b--","g--"],
        ["RCS_LE", "RCS_ME", "RCS_HE"]):  
    plot2.append(ax2.plot(E_ext_LE*1e-9, 
                          RCS.peak_NC_power*1e-9, 
                          color, label="power "+label)[0])
ax2.set_ylim(0, 100)
ax2.tick_params(axis ='y', labelcolor = 'blue') 

# Add legends

lns = plot1 + plot2
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc=0)
 
# Show plot

plt.show()    

prod = 1.
for RCS, RCS_ref, color, label in zip(
        [RCS_LE, RCS_ME, RCS_HE], 
        [RCS_LE_ref, RCS_ME_ref, RCS_HE_ref], 
        ["r-","b-","g-"],
        ["RCS_LE", "RCS_ME", "RCS_HE"]):  
    prod *= RCS.muon_survival
    plt.plot(E_ext_LE*1e-9, RCS.muon_survival*100, color, label=label)
plt.plot(E_ext_LE*1e-9, prod*100, "k-", label="total")
plt.xlim(E_ext_LE[0]*1e-9, E_ext_LE[-1]*1e-9)
plt.ylim(0, 100)
plt.xlabel("Extraction energy of RCS LE [GeV]")
plt.ylabel("Muon survival [%]")
plt.legend()
plt.show()

for RCS, RCS_ref, color, label in zip(
        [RCS_LE, RCS_ME, RCS_HE], 
        [RCS_LE_ref, RCS_ME_ref, RCS_HE_ref], 
        ["r-","b-","g-"],
        ["RCS_LE", "RCS_ME", "RCS_HE"]):  
    plt.plot(E_ext_LE*1e-9, RCS.inj_Qs, color, label="inj "+label)
    plt.plot(E_ext_LE*1e-9, RCS.ext_Qs, color+"-", label="ext "+label)
plt.xlim(E_ext_LE[0]*1e-9, E_ext_LE[-1]*1e-9)
plt.ylim(0, 2)
plt.axhline(1/np.pi, ls="-.", color="k")
plt.text(2.2*63,1/np.pi+0.05, r"1/$\pi$", fontsize=14)
plt.xlabel("Extraction energy of RCS LE [GeV]")
plt.ylabel(r"Synchrotron tune $Q_s$")
plt.legend()
plt.show()