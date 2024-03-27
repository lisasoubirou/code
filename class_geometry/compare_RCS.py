# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:01:12 2022

@author: Achance
"""

# with open("para_default_RCS.txt", "w") as fin:
#     fin.write(json.dumps(PulsedSynch.default_values, sort_keys=True,
#                          indent=3, separators=(',', ': ')))
from pathlib import Path
import numpy as np

import scipy.constants as cons
from scipy.interpolate import UnivariateSpline


import matplotlib.pyplot as plt
from matplotlib import rc

from gener_pulsed_synchrotron import PulsedSynch

rc("figure", dpi=200)
rc("legend", columnspacing=1.2, handletextpad=0.4, 
   handlelength=2, labelspacing=0.4, borderaxespad=0.4)

file_ref = "para_RCS_ME.txt"
#dir_out = Path("muon_synchrotron")
dir_out = Path(".")

RCS_LE_ref = PulsedSynch("para_RCS_LE.txt")
RCS_LE_2harm = PulsedSynch("para_RCS_LE_2harm.txt")

RCS_ME_ref = PulsedSynch("para_RCS_ME.txt")
RCS_ME_2harm = PulsedSynch("para_RCS_ME_2harm.txt")
RCS_ME_3harm = PulsedSynch("para_RCS_ME_3harm.txt")


path_length = RCS_ME_2harm.geometry.path_length*RCS_ME_2harm.nb_cell_arcs*2
time = np.linspace(RCS_ME_2harm.dipole_period/2/np.pi*RCS_ME_2harm.inj_phi, 
                   RCS_ME_2harm.dipole_period/2/np.pi*RCS_ME_2harm.ext_phi, 
                   len(path_length))
plt.plot(time*1e3, (path_length-path_length[0])*1e3, "r-")
plt.xlabel("Time [ms]")
plt.ylabel("Path length [mm]")
plt.show()

#Plot trajectory
RCS_ME_2harm.geometry.plot()

tab_E = np.linspace(200, 400, 101)*1e9
Brho = tab_E/cons.c
B0 = 1.8
tot_dipole = Brho*2*np.pi/B0
Lelem_cell = 2*0.3 + 2.5
Lcell = 30. 
nc = np.floor(tot_dipole/(Lcell/2-Lelem_cell))/2
tot_arc = nc*Lcell
filling = tot_arc/5990.

#Survival rate af Beam energy
survival_RCS1 = []
survival_RCS2 = []
survival_tot = []
period_RCS1 = []
period_RCS2 = []
Bslope_RCS1 = []
Bslope_RCS2 = []
totvoltage_RCS1 = []
totvoltage_RCS2 = []
pathlength_RCS2 = []
for E, fill in zip(tab_E, filling):
    RCS_LE_2harm.set_params(
        ext_E = E, filling_ratio=fill, max_B_slope=5000); 
    RCS_LE_2harm.set_params(nb_cell_arc=RCS_LE_2harm.nc_best);
    RCS_ME_2harm.set_params(inj_E = E, max_B_slope=5000); 
    RCS_ME_2harm.set_params(nb_cell_arc=RCS_ME_2harm.nc_best);
    survival_RCS1.append(RCS_LE_2harm.muon_survival)
    survival_RCS2.append(RCS_ME_2harm.muon_survival)
    survival_tot.append(RCS_ME_2harm.muon_survival*RCS_LE_2harm.muon_survival)
    period_RCS1.append(RCS_LE_2harm.dipole_period)
    period_RCS2.append(RCS_ME_2harm.dipole_period)
    Bslope_RCS1.append(RCS_LE_2harm.peak_B_slope)
    Bslope_RCS2.append(RCS_ME_2harm.peak_B_slope)
    totvoltage_RCS1.append(RCS_LE_2harm.tot_RF_voltage)
    totvoltage_RCS2.append(RCS_ME_2harm.tot_RF_voltage)
    pathlength_RCS2.append(RCS_ME_2harm.geometry.path_length_diff_tot)
plt.plot(tab_E*1e-9, np.array(survival_RCS1)*100, "r-", label="RCS1")
plt.plot(tab_E*1e-9, np.array(survival_RCS2)*100, "b-", label="RCS2")
plt.plot(tab_E*1e-9, np.array(survival_tot)*100, "k-", label="total")
plt.xlabel("Beam energy [GeV]")
plt.ylabel("Muon survival [%]")
plt.legend()
plt.show()

#Dipole period, Maximum dipole slope af Beam energy
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel("Beam energy [GeV]")
ax1.set_ylabel('Dipole period [ms]', color=color)
ax1.plot(tab_E*1e-9, np.array(period_RCS1)*1e3, "r-", label="Dipole period RCS1")
ax1.plot(tab_E*1e-9, np.array(period_RCS2)*1e3, "b-", label="Dipole period RCS2")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Maximum dipole slope [kT/s]', color=color)  # we already handled the x-label with ax1
ax2.plot(tab_E*1e-9, np.array(Bslope_RCS1)*1e-3, "r--", label="Peak slope RCS1")
ax2.plot(tab_E*1e-9, np.array(Bslope_RCS2)*1e-3, "b--", label="Peak slope RCS2")
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
ax1.legend()
ax2.legend()
plt.show()

#Total voltage af Beam energy
plt.plot(tab_E*1e-9, np.array(totvoltage_RCS1)*1e-9, "r-", label="RCS1")
plt.plot(tab_E*1e-9, np.array(totvoltage_RCS2)*1e-9, "b-", label="RCS2")
plt.xlabel("Beam energy [GeV]")
plt.ylabel("Total voltage [GV]")
plt.legend()
plt.show()

#Path legth variation af Beam energy
plt.plot(tab_E*1e-9, np.array(pathlength_RCS2)*1e3, "r-", label="RCS1")
plt.xlabel("Beam energy [GeV]")
plt.ylabel("Path length variation [mm]")
plt.show()

#Trajectories
RCS_ME_2harm.geometry.plot()
RCS_ME_3harm.geometry.plot()

#%% Ramp study: B, Energy, Synchronous phase (lin/harm)
#Bfield ramp af time (linear & harmonic ramp)
fig, ax1 = plt.subplots()
RCS_ME_ref.plot_B(ax=ax1, label="linear",
                  dipoles=["BNC"])
RCS_ME_2harm.plot_B(ax=ax1, label="harm dipole",
                  dipoles=["BNC"])
# RCS_ME_3harm.plot_B(ax=ax1, label="2 harm dipoles",
#                   dipoles=["BNC1", "BNC2"])
plt.legend()
#plt.savefig(dir_out / "dipole_field_ME.png")
plt.show()
plt.close()

#Beam energy af time (linea/harmonic)
fig, ax1 = plt.subplots()
RCS_ME_ref.plot_gamma(ax=ax1, label="linear")
RCS_ME_2harm.plot_gamma(ax=ax1, label="harm dipole")
# RCS_ME_3harm.plot_gamma(ax=ax1, label="2 harm dipoles")
#plt.savefig(dir_out / "energy_ME.png")
plt.legend()
plt.show()
plt.close()    

#Synchronous phase af time (linear/harmonic)
fig, ax1 = plt.subplots()
RCS_ME_ref.plot_phis(ax=ax1, label="linear")
RCS_ME_2harm.plot_phis(ax=ax1, label="harm dipole")
# RCS_ME_3harm.plot_phis(ax=ax1, label="2 harm dipoles")
ax1.set_ylim(100, 180)
#plt.savefig(dir_out / "synchronous_phase_ME.png")
plt.legend()
plt.show()
plt.close()
# exit

#%% Init
#Change structure of cell: number of dipole and NC/SC first
tab_nc = np.arange(1, 21)
arr_RCS_ME_3dip_NC = [
    PulsedSynch(file_ref, 
    pattern= ["BNC", "BSC", "BNC"],
    nb_cell_arc = nc) for nc in tab_nc
    ]

arr_RCS_ME_5dip_NC = [
    PulsedSynch(file_ref, 
    pattern= ["BNC", "BSC", "BNC", "BSC", "BNC"],
    nb_cell_arc = nc) for nc in tab_nc
    ]
arr_RCS_ME_7dip_NC = [
    PulsedSynch(file_ref, 
    pattern= ["BNC", "BSC", "BNC", "BSC", "BNC", "BSC", "BNC"],
    nb_cell_arc = nc) for nc in tab_nc
    ]
arr_RCS_ME_3dip_SC = [
    PulsedSynch(file_ref, 
    pattern= ["BSC", "BNC", "BSC"],
    nb_cell_arc = nc) for nc in tab_nc
    ]
arr_RCS_ME_5dip_SC = [
    PulsedSynch(file_ref, 
    pattern= ["BSC", "BNC", "BSC", "BNC", "BSC"],
    nb_cell_arc = nc) for nc in tab_nc
    ]
arr_RCS_ME_7dip_SC = [
    PulsedSynch(file_ref, 
    pattern= ["BSC", "BNC", "BSC", "BNC", "BSC", "BNC", "BSC"],
    nb_cell_arc = nc) for nc in tab_nc
    ]

fac = arr_RCS_ME_7dip_SC[0].nb_cell_arcs*2

#%%
#UnivariateSpline: interpolating piecewise with polynomials
nc_max_3dip = UnivariateSpline(tab_nc, [rcs.QP_dipole_spacing for rcs in arr_RCS_ME_3dip_SC], s=0).roots()[0]
nc_max_5dip = UnivariateSpline(tab_nc, [rcs.QP_dipole_spacing for rcs in arr_RCS_ME_5dip_SC], s=0).roots()[0]
nc_max_7dip = UnivariateSpline(tab_nc, [rcs.QP_dipole_spacing for rcs in arr_RCS_ME_7dip_SC], s=0).roots()[0]

#Total path length difference 
plt.semilogy(tab_nc*fac, [rcs.geometry.path_length_diff_tot for rcs in arr_RCS_ME_3dip_SC], "r-", label="3 dipoles") #log scale in y-axis
plt.semilogy(tab_nc*fac, [rcs.geometry.path_length_diff_tot for rcs in arr_RCS_ME_5dip_SC], "b-", label="5 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.path_length_diff_tot for rcs in arr_RCS_ME_7dip_SC], "g-", label="7 dipoles")
plt.axvline(nc_max_3dip*fac, ls="-", color="r", lw=3)
plt.axvline(nc_max_5dip*fac, ls="-", color="b", lw=3)
plt.axvline(nc_max_7dip*fac, ls="-", color="g", lw=3)
plt.xlabel("Total number of dipole sets")
plt.ylabel("Total path length difference [m]")
plt.legend()
plt.ylim(1e-3, 10)
plt.savefig(dir_out / "path_length_ME_SC.png")
plt.show()

#Horizontal traj difference
plt.semilogy(tab_nc*fac, [rcs.geometry.max_apert*1e3 for rcs in arr_RCS_ME_3dip_SC], "r-", label="3 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.max_apert*1e3 for rcs in arr_RCS_ME_5dip_SC], "b-", label="5 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.max_apert*1e3 for rcs in arr_RCS_ME_7dip_SC], "g-", label="7 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.max_apert_noshift*1e3 for rcs in arr_RCS_ME_3dip_SC], "r--", label="No shift 3 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.max_apert_noshift*1e3 for rcs in arr_RCS_ME_5dip_SC], "b--", label="No shift 5 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.max_apert_noshift*1e3 for rcs in arr_RCS_ME_7dip_SC], "g--", label="No shift 7 dipoles")
plt.axvline(nc_max_3dip*fac, ls="-", color="r", lw=3)
plt.axvline(nc_max_5dip*fac, ls="-", color="b", lw=3)
plt.axvline(nc_max_7dip*fac, ls="-", color="g", lw=3)
plt.xlabel("Total number of dipole sets")
plt.ylabel("Horizontal trajectory difference [mm]")
plt.ylim(1, 1000)
plt.legend()
plt.savefig(dir_out / "traj_difference_ME_SC.png")
plt.show()

#Minimum dipole full aperture
plt.semilogy(tab_nc*fac, [rcs.max_apertx*1e3 for rcs in arr_RCS_ME_3dip_SC], "r-", label="3 dipoles Hor.")
plt.semilogy(tab_nc*fac, [rcs.max_apertx*1e3 for rcs in arr_RCS_ME_5dip_SC], "b-", label="5 dipoles Hor.")
plt.semilogy(tab_nc*fac, [rcs.max_apertx*1e3 for rcs in arr_RCS_ME_7dip_SC], "g-", label="7 dipoles Hor.")
plt.semilogy(tab_nc*fac, [rcs.max_apertx_noshift*1e3 for rcs in arr_RCS_ME_3dip_SC], "r--", label="No shift 3 dipoles")
plt.semilogy(tab_nc*fac, [rcs.max_apertx_noshift*1e3 for rcs in arr_RCS_ME_5dip_SC], "b--", label="No shift 5 dipoles")
plt.semilogy(tab_nc*fac, [rcs.max_apertx_noshift*1e3 for rcs in arr_RCS_ME_7dip_SC], "g--", label="No shift 7 dipoles")
plt.axvline(nc_max_3dip*fac, ls="-", color="r", lw=3)
plt.axvline(nc_max_5dip*fac, ls="-", color="b", lw=3)
plt.axvline(nc_max_7dip*fac, ls="-", color="g", lw=3)
plt.xlabel("Total number of dipole sets")
plt.ylabel("Minimum dipole full aperture [mm]")
plt.ylim(1, 1000)
plt.legend()
plt.savefig(dir_out / "aperture_ME_SC.png")
plt.show()

#Dipole length
plt.semilogy(tab_nc*fac, [rcs.dipole_families["BNC"]["length"] for rcs in arr_RCS_ME_3dip_SC], "r-", label="NC 3 dipoles")
plt.semilogy(tab_nc*fac, [rcs.dipole_families["BNC"]["length"] for rcs in arr_RCS_ME_5dip_SC], "b-", label="NC 5 dipoles")
plt.semilogy(tab_nc*fac, [rcs.dipole_families["BNC"]["length"] for rcs in arr_RCS_ME_7dip_SC], "g-", label="NC 7 dipoles")
plt.semilogy(tab_nc*fac, [rcs.dipole_families["BSC"]["length"] for rcs in arr_RCS_ME_3dip_SC], "r--", label="SC 3 dipoles")
plt.semilogy(tab_nc*fac, [rcs.dipole_families["BSC"]["length"] for rcs in arr_RCS_ME_5dip_SC], "b--", label="SC 5 dipoles")
plt.semilogy(tab_nc*fac, [rcs.dipole_families["BSC"]["length"] for rcs in arr_RCS_ME_7dip_SC], "g--", label="SC 7 dipoles")
plt.axvline(nc_max_3dip*fac, ls="-", color="r", lw=3)
plt.axvline(nc_max_5dip*fac, ls="-", color="b", lw=3)
plt.axvline(nc_max_7dip*fac, ls="-", color="g", lw=3)
plt.xlabel("Total number of dipole sets")
plt.ylabel("Dipole length [m]")
plt.legend()
plt.ylim(0.1, 20)
plt.savefig(dir_out / "dipole_length_ME_SC.png")
plt.show()

#MCF
plt.semilogy(tab_nc*fac, [rcs.geometry.alpha for rcs in arr_RCS_ME_3dip_SC], "r-", label="3 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.alpha for rcs in arr_RCS_ME_5dip_SC], "b-", label="5 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.alpha for rcs in arr_RCS_ME_7dip_SC], "g-", label="7 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.alpha for rcs in arr_RCS_ME_3dip_SC], "r--", label="No shift 3 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.alpha for rcs in arr_RCS_ME_5dip_SC], "b--", label="No shift 5 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.alpha for rcs in arr_RCS_ME_7dip_SC], "g--", label="No shift 7 dipoles")
plt.axvline(nc_max_3dip*fac, ls="-", color="r", lw=3)
plt.axvline(nc_max_5dip*fac, ls="-", color="b", lw=3)
plt.axvline(nc_max_7dip*fac, ls="-", color="g", lw=3)
plt.xlabel("Total number of dipole sets")
plt.ylabel("Momentum compaction")
plt.legend()
plt.savefig(dir_out / "momentum_compaction_ME_SC.png")
plt.show()

#%%
#Total path length difference
nc_max_3dip = UnivariateSpline(tab_nc, [rcs.QP_dipole_spacing for rcs in arr_RCS_ME_3dip_NC], s=0).roots()[0]
nc_max_5dip = UnivariateSpline(tab_nc, [rcs.QP_dipole_spacing for rcs in arr_RCS_ME_5dip_NC], s=0).roots()[0]
nc_max_7dip = UnivariateSpline(tab_nc, [rcs.QP_dipole_spacing for rcs in arr_RCS_ME_7dip_NC], s=0).roots()[0]
fac = arr_RCS_ME_7dip_NC[0].nb_cell_arcs*2
plt.semilogy(tab_nc*fac, [rcs.geometry.path_length_diff_tot for rcs in arr_RCS_ME_3dip_NC], "r-", label="3 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.path_length_diff_tot for rcs in arr_RCS_ME_5dip_NC], "b-", label="5 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.path_length_diff_tot for rcs in arr_RCS_ME_7dip_NC], "g-", label="7 dipoles")
plt.axvline(nc_max_3dip*fac, ls="-", color="r", lw=3)
plt.axvline(nc_max_5dip*fac, ls="-", color="b", lw=3)
plt.axvline(nc_max_7dip*fac, ls="-", color="g", lw=3)
plt.xlabel("Total number of dipole sets")
plt.ylabel("Total path length difference [m]")
plt.ylim(1e-3, 10)
plt.legend()
#plt.savefig(dir_out / "path_length_ME_NC.png")
plt.show()

#Horizontal trajectory difference
plt.semilogy(tab_nc*fac, [rcs.geometry.max_apert*1e3 for rcs in arr_RCS_ME_3dip_NC], "r-", label="3 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.max_apert*1e3 for rcs in arr_RCS_ME_5dip_NC], "b-", label="5 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.max_apert*1e3 for rcs in arr_RCS_ME_7dip_NC], "g-", label="7 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.max_apert_noshift*1e3 for rcs in arr_RCS_ME_3dip_NC], "r--", label="No shift 3 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.max_apert_noshift*1e3 for rcs in arr_RCS_ME_5dip_NC], "b--", label="No shift 5 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.max_apert_noshift*1e3 for rcs in arr_RCS_ME_7dip_NC], "g--", label="No shift 7 dipoles")
plt.axvline(nc_max_3dip*fac, ls="-", color="r", lw=3)
plt.axvline(nc_max_5dip*fac, ls="-", color="b", lw=3)
plt.axvline(nc_max_7dip*fac, ls="-", color="g", lw=3)
plt.xlabel("Total number of dipole sets")
plt.ylabel("Horizontal trajectory difference [mm]")
plt.ylim(1, 1000)
plt.legend()
#plt.savefig(dir_out / "traj_difference_ME_NC.png")
plt.show()

#Minimum dipole full aperture
plt.semilogy(tab_nc*fac, [rcs.max_apertx*1e3 for rcs in arr_RCS_ME_3dip_NC], "r-", label="3 dipoles Hor.")
plt.semilogy(tab_nc*fac, [rcs.max_apertx*1e3 for rcs in arr_RCS_ME_5dip_NC], "b-", label="5 dipoles Hor.")
plt.semilogy(tab_nc*fac, [rcs.max_apertx*1e3 for rcs in arr_RCS_ME_7dip_NC], "g-", label="7 dipoles Hor.")
plt.semilogy(tab_nc*fac, [rcs.max_apertx_noshift*1e3 for rcs in arr_RCS_ME_3dip_NC], "r--", label="No shift 3 dipoles")
plt.semilogy(tab_nc*fac, [rcs.max_apertx_noshift*1e3 for rcs in arr_RCS_ME_5dip_NC], "b--", label="No shift 5 dipoles")
plt.semilogy(tab_nc*fac, [rcs.max_apertx_noshift*1e3 for rcs in arr_RCS_ME_7dip_NC], "g--", label="No shift 7 dipoles")
plt.axvline(nc_max_3dip*fac, ls="-", color="r", lw=3)
plt.axvline(nc_max_5dip*fac, ls="-", color="b", lw=3)
plt.axvline(nc_max_7dip*fac, ls="-", color="g", lw=3)
plt.xlabel("Total number of dipole sets")
plt.ylabel("Minimum dipole full aperture [mm]")
plt.legend()
plt.ylim(1, 1000)
#plt.savefig(dir_out / "aperture_ME_NC.png")
plt.show()

#Dipole length
plt.semilogy(tab_nc*fac, [rcs.dipole_families["BNC"]["length"] for rcs in arr_RCS_ME_3dip_NC], "r-", label="NC 3 dipoles")
plt.semilogy(tab_nc*fac, [rcs.dipole_families["BNC"]["length"] for rcs in arr_RCS_ME_5dip_NC], "b-", label="NC 5 dipoles")
plt.semilogy(tab_nc*fac, [rcs.dipole_families["BNC"]["length"] for rcs in arr_RCS_ME_7dip_NC], "g-", label="NC 7 dipoles")
plt.semilogy(tab_nc*fac, [rcs.dipole_families["BSC"]["length"] for rcs in arr_RCS_ME_3dip_NC], "r--", label="SC 3 dipoles")
plt.semilogy(tab_nc*fac, [rcs.dipole_families["BSC"]["length"] for rcs in arr_RCS_ME_5dip_NC], "b--", label="SC 5 dipoles")
plt.semilogy(tab_nc*fac, [rcs.dipole_families["BSC"]["length"] for rcs in arr_RCS_ME_7dip_NC], "g--", label="SC 7 dipoles")
plt.axvline(nc_max_3dip*fac, ls="-", color="r", lw=3)
plt.axvline(nc_max_5dip*fac, ls="-", color="b", lw=3)
plt.axvline(nc_max_7dip*fac, ls="-", color="g", lw=3)
plt.xlabel("Total number of dipole sets")
plt.ylabel("Dipole length [m]")
plt.ylim(0.1, 20)
plt.legend()
#plt.savefig(dir_out / "dipole_length_ME_NC.png")
plt.show()

#MCF
plt.semilogy(tab_nc*fac, [rcs.geometry.alpha for rcs in arr_RCS_ME_3dip_NC], "r-", label="3 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.alpha for rcs in arr_RCS_ME_5dip_NC], "b-", label="5 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.alpha for rcs in arr_RCS_ME_7dip_NC], "g-", label="7 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.alpha for rcs in arr_RCS_ME_3dip_NC], "r--", label="No shift 3 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.alpha for rcs in arr_RCS_ME_5dip_NC], "b--", label="No shift 5 dipoles")
plt.semilogy(tab_nc*fac, [rcs.geometry.alpha for rcs in arr_RCS_ME_7dip_NC], "g--", label="No shift 7 dipoles")
plt.axvline(nc_max_3dip*fac, ls="-", color="r", lw=3)
plt.axvline(nc_max_5dip*fac, ls="-", color="b", lw=3)
plt.axvline(nc_max_7dip*fac, ls="-", color="g", lw=3)
plt.xlabel("Total number of dipole sets")
plt.ylabel("Momentum compaction")
plt.legend()
#plt.savefig(dir_out / "momentum_compaction_ME_NC.png")
plt.show()


#%% Scan in B_sc field 
dipoles = {
  "BNC": [1.8, "linear"],
	  "BSC": 10.0
  }

tab_B = np.linspace(8, 12, 11)
arr_RCS_ME_3dip_NC = [
    PulsedSynch(file_ref, 
    pattern= ["BNC", "BSC", "BNC"],
    nb_cell_arc = 1,
    dipoles = {
      "BNC": [1.8, "linear"],
	  "BSC": B
      }) for B in tab_B
    ]
arr_RCS_ME_5dip_NC = [
    PulsedSynch(file_ref, 
    pattern= ["BNC", "BSC", "BNC", "BSC", "BNC"],
    nb_cell_arc = 1,
    dipoles = {
      "BNC": [1.8, "linear"],
	  "BSC": B
      }) for B in tab_B
    ]
arr_RCS_ME_7dip_NC = [
    PulsedSynch(file_ref, 
    pattern= ["BNC", "BSC", "BNC", "BSC", "BNC", "BSC", "BNC"],
    nb_cell_arc = 1,
    dipoles = {
      "BNC": [1.8, "linear"],
	  "BSC": B
      }) for B in tab_B
    ]
arr_RCS_ME_3dip_SC = [
    PulsedSynch(file_ref, 
    pattern= ["BSC", "BNC", "BSC"],
    nb_cell_arc = 1,
    dipoles = {
      "BNC": [1.8, "linear"],
	  "BSC": B
      }) for B in tab_B
    ]
arr_RCS_ME_5dip_SC = [
    PulsedSynch(file_ref, 
    pattern= ["BSC", "BNC", "BSC", "BNC", "BSC"],
    nb_cell_arc = 1,
    dipoles = {
      "BNC": [1.8, "linear"],
	  "BSC": B
      }) for B in tab_B
    ]
arr_RCS_ME_7dip_SC = [
    PulsedSynch(file_ref, 
    pattern= ["BSC", "BNC", "BSC", "BNC", "BSC", "BNC", "BSC"],
    nb_cell_arc = 1,
    dipoles = {
      "BNC": [1.8, "linear"],
	  "BSC": B
      }) for B in tab_B
    ]

nc_max_ME_3dip_NC = [
    int((rcs.arc_length-rcs.dipole_length_tot/rcs.nb_arc)/((len(rcs.pattern)-1)*rcs.dipole_spacing+rcs.SSS_length)/2)\
        for rcs in arr_RCS_ME_3dip_NC
        ]
nc_max_ME_5dip_NC = [
    int((rcs.arc_length-rcs.dipole_length_tot/rcs.nb_arc)/((len(rcs.pattern)-1)*rcs.dipole_spacing+rcs.SSS_length)/2)\
        for rcs in arr_RCS_ME_5dip_NC
        ]
nc_max_ME_7dip_NC = [
    int((rcs.arc_length-rcs.dipole_length_tot/rcs.nb_arc)/((len(rcs.pattern)-1)*rcs.dipole_spacing+rcs.SSS_length)/2)\
        for rcs in arr_RCS_ME_7dip_NC
        ]
nc_max_ME_3dip_SC = [
    int((rcs.arc_length-rcs.dipole_length_tot/rcs.nb_arc)/((len(rcs.pattern)-1)*rcs.dipole_spacing+rcs.SSS_length)/2)\
        for rcs in arr_RCS_ME_3dip_SC
        ]
nc_max_ME_5dip_SC = [
    int((rcs.arc_length-rcs.dipole_length_tot/rcs.nb_arc)/((len(rcs.pattern)-1)*rcs.dipole_spacing+rcs.SSS_length)/2)\
        for rcs in arr_RCS_ME_5dip_SC
        ]
nc_max_ME_7dip_SC = [
    int((rcs.arc_length-rcs.dipole_length_tot/rcs.nb_arc)/((len(rcs.pattern)-1)*rcs.dipole_spacing+rcs.SSS_length)/2)\
        for rcs in arr_RCS_ME_7dip_SC
        ]

arr_RCS_ME_3dip_NC = [
    PulsedSynch(file_ref, 
    pattern= ["BNC", "BSC", "BNC"],
    nb_cell_arc = nc,
    dipoles = {
      "BNC": [1.8, "linear"],
	  "BSC": B
      }) for B, nc in zip(tab_B,nc_max_ME_3dip_NC) 
    ]
arr_RCS_ME_5dip_NC = [
    PulsedSynch(file_ref, 
    pattern= ["BNC", "BSC", "BNC", "BSC", "BNC"],
    nb_cell_arc = nc,
    dipoles = {
      "BNC": [1.8, "linear"],
	  "BSC": B
      }) for B, nc in zip(tab_B,nc_max_ME_5dip_NC) 
    ]
arr_RCS_ME_7dip_NC = [
    PulsedSynch(file_ref, 
    pattern= ["BNC", "BSC", "BNC", "BSC", "BNC", "BSC", "BNC"],
    nb_cell_arc = nc,
    dipoles = {
      "BNC": [1.8, "linear"],
	  "BSC": B
      }) for B, nc in zip(tab_B,nc_max_ME_7dip_NC) 
    ]
arr_RCS_ME_3dip_SC = [
    PulsedSynch(file_ref, 
    pattern= ["BSC", "BNC", "BSC"],
    nb_cell_arc = nc,
    dipoles = {
      "BNC": [1.8, "linear"],
	  "BSC": B
      }) for B, nc in zip(tab_B,nc_max_ME_3dip_SC) 
    ]
arr_RCS_ME_5dip_SC = [
    PulsedSynch(file_ref, 
    pattern= ["BSC", "BNC", "BSC", "BNC", "BSC"],
    nb_cell_arc = nc,
    dipoles = {
      "BNC": [1.8, "linear"],
	  "BSC": B
      }) for B, nc in zip(tab_B,nc_max_ME_5dip_SC) 
    ]
arr_RCS_ME_7dip_SC = [
    PulsedSynch(file_ref, 
    pattern= ["BSC", "BNC", "BSC", "BNC", "BSC", "BNC", "BSC"],
    nb_cell_arc = nc,
    dipoles = {
      "BNC": [1.8, "linear"],
	  "BSC": B
      }) for B, nc in zip(tab_B,nc_max_ME_7dip_SC) 
    ]

#Total path length difference (SC)
plt.plot(tab_B, [rcs.geometry.path_length_diff_tot for rcs in arr_RCS_ME_3dip_SC], "r-", label="3 dipoles")
plt.plot(tab_B, [rcs.geometry.path_length_diff_tot for rcs in arr_RCS_ME_5dip_SC], "b-", label="5 dipoles")
plt.plot(tab_B, [rcs.geometry.path_length_diff_tot for rcs in arr_RCS_ME_7dip_SC], "g-", label="7 dipoles")
plt.xlabel("Superconducting dipole field [T]")
plt.ylabel("Total path length difference [m]")
plt.legend()
#plt.savefig(dir_out / "path_length_ME_SC_vsB.png")
plt.show()

#Horizontal trajectory difference (SC)
plt.plot(tab_B, [rcs.geometry.max_apert*1e3 for rcs in arr_RCS_ME_3dip_SC], "r-", label="3 dipoles")
plt.plot(tab_B, [rcs.geometry.max_apert*1e3 for rcs in arr_RCS_ME_5dip_SC], "b-", label="5 dipoles")
plt.plot(tab_B, [rcs.geometry.max_apert*1e3 for rcs in arr_RCS_ME_7dip_SC], "g-", label="7 dipoles")
plt.plot(tab_B, [rcs.geometry.max_apert_noshift*1e3 for rcs in arr_RCS_ME_3dip_SC], "r--", label="No shift 3 dipoles")
plt.plot(tab_B, [rcs.geometry.max_apert_noshift*1e3 for rcs in arr_RCS_ME_5dip_SC], "b--", label="No shift 5 dipoles")
plt.plot(tab_B, [rcs.geometry.max_apert_noshift*1e3 for rcs in arr_RCS_ME_7dip_SC], "g--", label="No shift 7 dipoles")
plt.xlabel("Superconducting dipole field [T]")
plt.ylabel("Horizontal trajectory difference [mm]")
plt.legend()
#plt.savefig(dir_out / "traj_difference_ME_SC_vsB.png")
plt.show()

#Minimum dipole full aperture (SC)
plt.plot(tab_B, [rcs.max_apertx*1e3 for rcs in arr_RCS_ME_3dip_SC], "r-", label="3 dipoles Hor.")
plt.plot(tab_B, [rcs.max_apertx*1e3 for rcs in arr_RCS_ME_5dip_SC], "b-", label="5 dipoles Hor.")
plt.plot(tab_B, [rcs.max_apertx*1e3 for rcs in arr_RCS_ME_7dip_SC], "g-", label="7 dipoles Hor.")
plt.plot(tab_B, [rcs.max_apertx_noshift*1e3 for rcs in arr_RCS_ME_3dip_SC], "r--", label="No shift 3 dipoles")
plt.plot(tab_B, [rcs.max_apertx_noshift*1e3 for rcs in arr_RCS_ME_5dip_SC], "b--", label="No shift 5 dipoles")
plt.plot(tab_B, [rcs.max_apertx_noshift*1e3 for rcs in arr_RCS_ME_7dip_SC], "g--", label="No shift 7 dipoles")
plt.xlabel("Superconducting dipole field [T]")
plt.ylabel("Minimum dipole full aperture [mm]")
plt.legend()
#plt.savefig(dir_out / "aperture_ME_SC_vsB.png")
plt.show()

#Dipole length (SC)
plt.plot(tab_B, [rcs.dipole_families["BNC"]["length"] for rcs in arr_RCS_ME_3dip_SC], "r-", label="NC 3 dipoles")
plt.plot(tab_B, [rcs.dipole_families["BNC"]["length"] for rcs in arr_RCS_ME_5dip_SC], "b-", label="NC 5 dipoles")
plt.plot(tab_B, [rcs.dipole_families["BNC"]["length"] for rcs in arr_RCS_ME_7dip_SC], "g-", label="NC 7 dipoles")
plt.plot(tab_B, [rcs.dipole_families["BSC"]["length"] for rcs in arr_RCS_ME_3dip_SC], "r--", label="SC 3 dipoles")
plt.plot(tab_B, [rcs.dipole_families["BSC"]["length"] for rcs in arr_RCS_ME_5dip_SC], "b--", label="SC 5 dipoles")
plt.plot(tab_B, [rcs.dipole_families["BSC"]["length"] for rcs in arr_RCS_ME_7dip_SC], "g--", label="SC 7 dipoles")
plt.xlabel("Superconducting dipole field [T]")
plt.ylabel("Dipole length [m]")
plt.legend()
#plt.savefig(dir_out / "dipole_length_ME_SC_vsB.png")
plt.show()

#MCF (SC)
plt.semilogy(tab_B, [rcs.geometry.alpha for rcs in arr_RCS_ME_3dip_SC], "r-", label="3 dipoles")
plt.semilogy(tab_B, [rcs.geometry.alpha for rcs in arr_RCS_ME_5dip_SC], "b-", label="5 dipoles")
plt.semilogy(tab_B, [rcs.geometry.alpha for rcs in arr_RCS_ME_7dip_SC], "g-", label="7 dipoles")
plt.semilogy(tab_B, [rcs.geometry.alpha for rcs in arr_RCS_ME_3dip_SC], "r--", label="No shift 3 dipoles")
plt.semilogy(tab_B, [rcs.geometry.alpha for rcs in arr_RCS_ME_5dip_SC], "b--", label="No shift 5 dipoles")
plt.semilogy(tab_B, [rcs.geometry.alpha for rcs in arr_RCS_ME_7dip_SC], "g--", label="No shift 7 dipoles")
plt.xlabel("Superconducting dipole field [T]")
plt.ylabel("Momentum compaction")
plt.legend()
#plt.savefig(dir_out / "momentum_compaction_ME_SC_vsB.png")
plt.show()

#Total path length difference (NC)
plt.semilogy(tab_B, [rcs.geometry.path_length_diff_tot for rcs in arr_RCS_ME_3dip_NC], "r-", label="3 dipoles")
plt.semilogy(tab_B, [rcs.geometry.path_length_diff_tot for rcs in arr_RCS_ME_5dip_NC], "b-", label="5 dipoles")
plt.semilogy(tab_B, [rcs.geometry.path_length_diff_tot for rcs in arr_RCS_ME_7dip_NC], "g-", label="7 dipoles")
plt.xlabel("Superconducting dipole field [T]")
plt.ylabel("Total path length difference [m]")
plt.ylim(1e-3, 10)
plt.legend()
#plt.savefig(dir_out / "path_length_ME_NC_vsB.png")
plt.show()

#Horizontal trajectory difference (NC)
plt.semilogy(tab_B, [rcs.geometry.max_apert*1e3 for rcs in arr_RCS_ME_3dip_NC], "r-", label="3 dipoles")
plt.semilogy(tab_B, [rcs.geometry.max_apert*1e3 for rcs in arr_RCS_ME_5dip_NC], "b-", label="5 dipoles")
plt.semilogy(tab_B, [rcs.geometry.max_apert*1e3 for rcs in arr_RCS_ME_7dip_NC], "g-", label="7 dipoles")
plt.semilogy(tab_B, [rcs.geometry.max_apert_noshift*1e3 for rcs in arr_RCS_ME_3dip_NC], "r--", label="No shift 3 dipoles")
plt.semilogy(tab_B, [rcs.geometry.max_apert_noshift*1e3 for rcs in arr_RCS_ME_5dip_NC], "b--", label="No shift 5 dipoles")
plt.semilogy(tab_B, [rcs.geometry.max_apert_noshift*1e3 for rcs in arr_RCS_ME_7dip_NC], "g--", label="No shift 7 dipoles")
plt.xlabel("Superconducting dipole field [T]")
plt.ylabel("Horizontal trajectory difference [mm]")
plt.ylim(1, 1000)
plt.legend()
#plt.savefig(dir_out / "traj_difference_ME_NC_vsB.png")
plt.show()

#Minimum dipole full aperture (NC)
plt.semilogy(tab_B, [rcs.max_apertx*1e3 for rcs in arr_RCS_ME_3dip_NC], "r-", label="3 dipoles Hor.")
plt.semilogy(tab_B, [rcs.max_apertx*1e3 for rcs in arr_RCS_ME_5dip_NC], "b-", label="5 dipoles Hor.")
plt.semilogy(tab_B, [rcs.max_apertx*1e3 for rcs in arr_RCS_ME_7dip_NC], "g-", label="7 dipoles Hor.")
plt.semilogy(tab_B, [rcs.max_apertx_noshift*1e3 for rcs in arr_RCS_ME_3dip_NC], "r--", label="No shift 3 dipoles")
plt.semilogy(tab_B, [rcs.max_apertx_noshift*1e3 for rcs in arr_RCS_ME_5dip_NC], "b--", label="No shift 5 dipoles")
plt.semilogy(tab_B, [rcs.max_apertx_noshift*1e3 for rcs in arr_RCS_ME_7dip_NC], "g--", label="No shift 7 dipoles")
plt.xlabel("Superconducting dipole field [T]")
plt.ylabel("Minimum dipole full aperture [mm]")
plt.legend()
plt.ylim(1, 1000)
#plt.savefig(dir_out / "aperture_ME_NC_vsB.png")
plt.show()

#Dipole length (NC)
plt.semilogy(tab_B, [rcs.dipole_families["BNC"]["length"] for rcs in arr_RCS_ME_3dip_NC], "r-", label="NC 3 dipoles")
plt.semilogy(tab_B, [rcs.dipole_families["BNC"]["length"] for rcs in arr_RCS_ME_5dip_NC], "b-", label="NC 5 dipoles")
plt.semilogy(tab_B, [rcs.dipole_families["BNC"]["length"] for rcs in arr_RCS_ME_7dip_NC], "g-", label="NC 7 dipoles")
plt.semilogy(tab_B, [rcs.dipole_families["BSC"]["length"] for rcs in arr_RCS_ME_3dip_NC], "r--", label="SC 3 dipoles")
plt.semilogy(tab_B, [rcs.dipole_families["BSC"]["length"] for rcs in arr_RCS_ME_5dip_NC], "b--", label="SC 5 dipoles")
plt.semilogy(tab_B, [rcs.dipole_families["BSC"]["length"] for rcs in arr_RCS_ME_7dip_NC], "g--", label="SC 7 dipoles")
plt.xlabel("Superconducting dipole field [T]")
plt.ylabel("Dipole length [m]")
plt.ylim(0.1, 20)
plt.legend()
#plt.savefig(dir_out / "dipole_length_ME_NC_vsB.png")
plt.show()

#MCF (NC)
plt.semilogy(tab_B, [rcs.geometry.alpha for rcs in arr_RCS_ME_3dip_NC], "r-", label="3 dipoles")
plt.semilogy(tab_B, [rcs.geometry.alpha for rcs in arr_RCS_ME_5dip_NC], "b-", label="5 dipoles")
plt.semilogy(tab_B, [rcs.geometry.alpha for rcs in arr_RCS_ME_7dip_NC], "g-", label="7 dipoles")
plt.semilogy(tab_B, [rcs.geometry.alpha for rcs in arr_RCS_ME_3dip_NC], "r--", label="No shift 3 dipoles")
plt.semilogy(tab_B, [rcs.geometry.alpha for rcs in arr_RCS_ME_5dip_NC], "b--", label="No shift 5 dipoles")
plt.semilogy(tab_B, [rcs.geometry.alpha for rcs in arr_RCS_ME_7dip_NC], "g--", label="No shift 7 dipoles")
plt.xlabel("Superconducting dipole field [T]")
plt.ylabel("Momentum compaction")
plt.legend()
#plt.savefig(dir_out / "momentum_compaction_ME_NC_vsB.png")
plt.show()

RCS_ME_ref = PulsedSynch("para_RCS_ME.txt")
RCS_HE_ref = PulsedSynch("para_RCS_HE.txt")


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

# E_ext_LE = np.linspace(2, 5, 101)*63e9
# E_ext_ME = 750e9+(E_ext_LE-300e9)*(10-1.5)/(10+1.5)
# RCS_LE = PulsedSynch("para_RCS_LE.txt", ext_E=E_ext_LE, 
#                      filling_ratio=0.85+2*np.pi/1.5/cons.c/5990.*(E_ext_LE-300e9),
#                      target_ramping=0.01e-3, max_NC_slope=1e5)
# RCS_ME = PulsedSynch("para_RCS_ME.txt", inj_E=E_ext_LE, 
#                      ext_E=E_ext_ME)
# RCS_HE = PulsedSynch("para_RCS_HE.txt", inj_E=E_ext_ME)

# fig, ax1 = plt.subplots() 

# ax1.set_xlabel('Extraction energy of RCS LE [GeV]') 
# ax1.set_ylabel('Total RF voltage [GV]') 
# plot1 = []
# for RCS, RCS_ref, color, label in zip(
#         [RCS_LE, RCS_ME, RCS_HE], 
#         [RCS_LE_ref, RCS_ME_ref, RCS_HE_ref], 
#         ["r-","b-","g-"],
#         ["RCS_LE", "RCS_ME", "RCS_HE"]):  
#     plot1.append(ax1.plot(E_ext_LE*1e-9, 
#                           RCS.tot_RF_voltage*1e-9, color, label=label)[0])
# # ax1.tick_params(axis ='y', labelcolor = 'red') 

# # Adding Twin Axes
# ax1.set_xlim(E_ext_LE[0]*1e-9, E_ext_LE[-1]*1e-9)
# ax1.set_ylim(0, 100)
# ax2 = ax1.twinx() 
  
# ax2.set_ylabel('Average RF gradient [MV/m]', color = 'blue') 
# plot2 = []
# for RCS, RCS_ref, color, label in zip(
#         [RCS_LE, RCS_ME, RCS_HE], 
#         [RCS_LE_ref, RCS_ME_ref, RCS_HE_ref], 
#         ["r--","b--","g--"],
#         ["gradient RCS_LE", "gradient RCS_ME", "gradient RCS_HE"]):  
#     plot2.append(ax2.plot(E_ext_LE*1e-9, 
#                           RCS.tot_RF_voltage/RCS.RF_length_tot*1e-6, 
#                           color, label=label)[0])
# ax2.set_ylim(0, 200)
# ax2.tick_params(axis ='y', labelcolor = 'blue') 

# # Add legends

# lns = plot1 + plot2
# labels = [l.get_label() for l in lns]
# plt.legend(lns, labels, loc=0)
 
# # Show plot

# plt.show()    

# fig, ax1 = plt.subplots() 

# ax1.set_xlabel('Extraction energy of RCS LE [GeV]') 
# ax1.set_ylabel('NC dipole field slope [kT/s]') 
# plot1 = []
# for RCS, RCS_ref, color, label in zip(
#         [RCS_LE, RCS_ME, RCS_HE], 
#         [RCS_LE_ref, RCS_ME_ref, RCS_HE_ref], 
#         ["r-","b-","g-"],
#         ["RCS_LE", "RCS_ME", "RCS_HE"]):  
#     plot1.append(ax1.plot(E_ext_LE*1e-9, 
#                           RCS.peak_NC_slope*1e-3, 
#                           color, label=label)[0])
# # ax1.tick_params(axis ='y', labelcolor = 'red') 

# # Adding Twin Axes
# ax1.set_xlim(E_ext_LE[0]*1e-9, E_ext_LE[-1]*1e-9)
# ax1.set_ylim(0, 10)
# ax2 = ax1.twinx() 
  
# ax2.set_ylabel('Dipole peak power [GW]', color = 'blue') 
# plot2 = []
# for RCS, RCS_ref, color, label in zip(
#         [RCS_LE, RCS_ME, RCS_HE], 
#         [RCS_LE_ref, RCS_ME_ref, RCS_HE_ref], 
#         ["r--","b--","g--"],
#         ["RCS_LE", "RCS_ME", "RCS_HE"]):  
#     plot2.append(ax2.plot(E_ext_LE*1e-9, 
#                           RCS.peak_NC_power*1e-9, 
#                           color, label="power "+label)[0])
# ax2.set_ylim(0, 100)
# ax2.tick_params(axis ='y', labelcolor = 'blue') 

# # Add legends

# lns = plot1 + plot2
# labels = [l.get_label() for l in lns]
# plt.legend(lns, labels, loc=0)
 
# # Show plot

# plt.show()    

# prod = 1.
# for RCS, RCS_ref, color, label in zip(
#         [RCS_LE, RCS_ME, RCS_HE], 
#         [RCS_LE_ref, RCS_ME_ref, RCS_HE_ref], 
#         ["r-","b-","g-"],
#         ["RCS_LE", "RCS_ME", "RCS_HE"]):  
#     prod *= RCS.muon_survival
#     plt.plot(E_ext_LE*1e-9, RCS.muon_survival*100, color, label=label)
# plt.plot(E_ext_LE*1e-9, prod*100, "k-", label="total")
# plt.xlim(E_ext_LE[0]*1e-9, E_ext_LE[-1]*1e-9)
# plt.ylim(0, 100)
# plt.xlabel("Extraction energy of RCS LE [GeV]")
# plt.ylabel("Muon survival [%]")
# plt.legend()
# plt.show()

# for RCS, RCS_ref, color, label in zip(
#         [RCS_LE, RCS_ME, RCS_HE], 
#         [RCS_LE_ref, RCS_ME_ref, RCS_HE_ref], 
#         ["r-","b-","g-"],
#         ["RCS_LE", "RCS_ME", "RCS_HE"]):  
#     plt.plot(E_ext_LE*1e-9, RCS.inj_Qs, color, label="inj "+label)
#     plt.plot(E_ext_LE*1e-9, RCS.ext_Qs, color+"-", label="ext "+label)
# plt.xlim(E_ext_LE[0]*1e-9, E_ext_LE[-1]*1e-9)
# plt.ylim(0, 2)
# plt.axhline(1/np.pi, ls="-.", color="k")
# plt.text(2.2*63,1/np.pi+0.05, r"1/$\pi$", fontsize=14)
# plt.xlabel("Extraction energy of RCS LE [GeV]")
# plt.ylabel(r"Synchrotron tune $Q_s$")
# plt.legend()
# plt.show()

"""nb_cell_arc = np.arange(12, 35)
RCS_ME = PulsedSynch("para_RCS_ME.txt", nb_cell_arc=nb_cell_arc)
RCS_ME_5 = PulsedSynch("para_RCS_ME.txt", nb_cell_arc=nb_cell_arc,
                       nb_dipole_cell=5)
RCS_ME_7 = PulsedSynch("para_RCS_ME.txt", nb_cell_arc=nb_cell_arc,
                       nb_dipole_cell=7)
plt.semilogy(RCS_ME.nb_cell_arcs, RCS_ME.diff_path, "b-", label=r"$n_{SC}$=1")
plt.semilogy(RCS_ME.nb_cell_arcs, RCS_ME_5.diff_path, "r-", label=r"$n_{SC}$=2")
plt.semilogy(RCS_ME.nb_cell_arcs, RCS_ME_7.diff_path, "g-", label=r"$n_{SC}$=3")
plt.xlabel("Total number of cells")
plt.ylabel(r"Path length difference [m]")
plt.legend()
plt.show()

plt.plot(RCS_ME.nb_cell_arcs, RCS_ME.dipole_spacing, "b-", label=r"$n_{SC}$=1")
plt.plot(RCS_ME.nb_cell_arcs, RCS_ME_5.dipole_spacing, "r-", label=r"$n_{SC}$=2")
plt.plot(RCS_ME.nb_cell_arcs, RCS_ME_7.dipole_spacing, "g-", label=r"$n_{SC}$=3")
plt.xlabel("Total number of cells")
plt.ylabel(r"Distance between dipoles [m]")
plt.legend()
plt.show()

plt.plot(RCS_ME.nb_cell_arcs, RCS_ME.SC_length, "b-", label=r"$n_{SC}$=1 SC")
plt.plot(RCS_ME.nb_cell_arcs, RCS_ME.NC_length, "b--", label=r"$n_{SC}$=1 NC")
plt.plot(RCS_ME.nb_cell_arcs, RCS_ME_5.SC_length, "r-", label=r"$n_{SC}$=2 SC")
plt.plot(RCS_ME.nb_cell_arcs, RCS_ME_5.NC_length, "r--", label=r"$n_{SC}$=2 NC")
plt.plot(RCS_ME.nb_cell_arcs, RCS_ME_7.SC_length, "g-", label=r"$n_{SC}$=3 SC")
plt.plot(RCS_ME.nb_cell_arcs, RCS_ME_7.NC_length, "g--", label=r"$n_{SC}$=3 NC")
plt.xlabel("Total number of cells")
plt.ylabel(r"Dipole length [m]")
plt.legend()
plt.show()

nb_cell_arc = np.arange(12, 35)
RCS_HE = PulsedSynch("para_RCS_HE.txt", nb_cell_arc=nb_cell_arc)
RCS_HE_5 = PulsedSynch("para_RCS_HE.txt", nb_cell_arc=nb_cell_arc,
                       nb_dipole_cell=5)
RCS_HE_7 = PulsedSynch("para_RCS_HE.txt", nb_cell_arc=nb_cell_arc,
                       nb_dipole_cell=7)
plt.semilogy(RCS_ME.nb_cell_arcs, RCS_HE.diff_path, "b-", label=r"$n_{SC}$=1")
plt.semilogy(RCS_ME.nb_cell_arcs, RCS_HE_5.diff_path, "r-", label=r"$n_{SC}$=2")
plt.semilogy(RCS_ME.nb_cell_arcs, RCS_HE_7.diff_path, "g-", label=r"$n_{SC}$=3")
plt.xlabel("Total number of cells")
plt.ylabel(r"Path length difference [m]")
plt.legend()
plt.show()

plt.plot(RCS_ME.nb_cell_arcs, RCS_HE.dipole_spacing, "b-", label=r"$n_{SC}$=1")
plt.plot(RCS_ME.nb_cell_arcs, RCS_HE_5.dipole_spacing, "r-", label=r"$n_{SC}$=2")
plt.plot(RCS_ME.nb_cell_arcs, RCS_HE_7.dipole_spacing, "g-", label=r"$n_{SC}$=3")
plt.xlabel("Total number of cells")
plt.ylabel(r"Distance between dipoles [m]")
plt.legend()
plt.show()

plt.plot(RCS_ME.nb_cell_arcs, RCS_HE.SC_length, "b-", label=r"$n_{SC}$=1 SC")
plt.plot(RCS_ME.nb_cell_arcs, RCS_HE.NC_length, "b--", label=r"$n_{SC}$=1 NC")
plt.plot(RCS_ME.nb_cell_arcs, RCS_HE_5.SC_length, "r-", label=r"$n_{SC}$=2 SC")
plt.plot(RCS_ME.nb_cell_arcs, RCS_HE_5.NC_length, "r--", label=r"$n_{SC}$=2 NC")
plt.plot(RCS_ME.nb_cell_arcs, RCS_HE_7.SC_length, "g-", label=r"$n_{SC}$=3 SC")
plt.plot(RCS_ME.nb_cell_arcs, RCS_HE_7.NC_length, "g--", label=r"$n_{SC}$=3 NC")
plt.xlabel("Total number of cells")
plt.ylabel(r"Dipole length [m]")
plt.legend()
plt.show()"""