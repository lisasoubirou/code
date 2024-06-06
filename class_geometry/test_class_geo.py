import json
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/mnt/c/muco/rcsparameters/rcsparameters')
from geometry.geometry import Geometry

file_input='/mnt/c/muco/code/class_geometry/parameter_files/para_RCS_LHC.txt'
RCS = Geometry(file_input)
print('Input parameters')
print('Circumference  [m]', RCS.C)
print('Injection energy [eV]',RCS.E_inj)
print('Extraction energy [eV]',RCS.E_ext)
print('Length of the straight section  [m]',RCS.LSSS) 
print('Dipole spacing  [m]', RCS.dipole_spacing) 
print('Number of arcs',RCS.nb_arc)
print('Number of insertion',RCS.nb_RF_section)
print('Number of cells per arc',RCS.nb_cell_arc)
print('Dipole properties', RCS.dipoles)
print('Dipole pattern of a half-cell',RCS.pattern)
print('Filling ratio of the arcs', RCS.filling_ratio)
print('Filling ratio of the RF',RCS.filling_ratio_RF)

#Print some geometric parameters computed by the class
print('Some computed parameters')
print('Total arc length [m]', RCS.tot_arc_length)
print('Arc length [m]', RCS.arc_length)
print('Total insertion length [m]', RCS.tot_insertion_length)
print('Insertion length [m]', RCS.insertion_length)
print('Cell length [m]', RCS.cell_length)
print('Total length of dipole', RCS.dipole_length_tot)
print('Dipole filling factor', RCS.filling_ratio_dipole)
print('Dipole parameters', RCS.dipole_families)

#Print some maxima over acceleration time
print('Trajectory parameters')
print('Total trajecory excursion',np.max(RCS.extrema(RCS.t_ext)[1])-np.min(RCS.extrema(RCS.t_inj)[0]))
mask_nc=[item == 'BNC' for item in RCS.pattern]
mask_sc=[item == 'BSC' for item in RCS.pattern]
max_width_dip=RCS.extrema(RCS.t_ext)[1]-RCS.extrema(RCS.t_inj)[0]
print('Max width in SC', np.max(max_width_dip[mask_sc]))
print('Max width in NC',np.max(max_width_dip[mask_nc]))
print('Max path length difference', RCS.max_path_diff)

#Print the results for a given time
print('Some parameters at a given time')
t_print=0.5
y_min, y_max,width,max_apert_noshift,max_apert=RCS.extrema(t_print)
print('Results for t=',t_print)
print('Theta',RCS.theta(t_print))
print('Width', width)
print('Max apert no shift',max_apert_noshift)
print('Path length',RCS.path_length(t_print))
print('Path length tot',RCS.path_length_tot(t_print))

# Plot n_traj between injection and extraction
n_traj=8
t_traj=np.linspace(RCS.t_inj,RCS.t_ext,n_traj)
RCS.plot_traj(t_traj)

#Plottings
n_plot=15
t_np=np.linspace(RCS.t_inj,RCS.t_ext,n_plot)
#Bfield (NC)
plt.figure()
plt.plot(t_np, [RCS.dipole_families['BNC']["B"](t) for t in t_np])
plt.xlabel('Time normalised')
plt.ylabel('B [T]')
plt.show()
#Total Path length diff
plt.figure()
plt.plot( t_np, [(RCS.path_length_tot(t)-RCS.path_length_min)*1e3 for t in t_np])
plt.xlabel('Time normalised')
plt.ylabel('Total path length difference [mm]')
plt.show()
