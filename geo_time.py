# -*- coding: utf-8 -*-
"""
Created on Wed Feb  28 11:11:16 2024
@author: LS276867

Code used to study geometry variation with time. 
Path length variation, ending point variation.
Trajectories are compared between mathematical formalism of geo_class and Xsuite tracking.
Functions: 
    - track : tracking one 1 turn 
    - path_length_plot : plot path_length variation for a given time 
    - z_final : compares ending point
Routines are going with this functions
Printing routines to print results
"""

import numpy as np
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt
import xpart as xp
import sys
sys.path.append('/mnt/c/muco')
from class_geometry.class_geo import Geometry 
from optics_function import make_optics, plot_twiss 

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

file_input='/mnt/c/muco/code/class_geometry/para_RCS_ME.txt'
RCS = Geometry(file_input)
time_frac=0
nb_cell_tot=RCS.nb_cell_arcs
nb_arc=RCS.nb_arc
angle_cell=RCS.cell_angle
angle_arc=2*np.pi/nb_arc
N_turn=55

def track(line, num_turns=1):
    line.build_tracker()
    part=line.build_particles()
    if num_turns==1:
        line.track(part,num_turns=1, turn_by_turn_monitor='ONE_TURN_EBE')
    else:
        line.track(part,num_turns=num_turns, turn_by_turn_monitor=True)
    rec=line.record_last_track
    return (rec)

n_k=21
option='var_k'
# option='var_ref'

# line_cell_inj, line_ds_inj = make_optics(file_input, 0, n_k, option)
# line_cell_ext, line_ds_ext = make_optics(file_input, 1, n_k, option)
# line_cell_mid, line_ds_mid=make_optics(file_input, 0.5, n_k, option)
# line_cell_mid0, line_ds_mid0=make_optics(file_input, 0.1, n_k,option)
# line_cell_mid1, line_ds_mid1=make_optics(file_input, 0.25, n_k,option)
# line_cell_mid2, line_ds_mid2=make_optics(file_input, 0.75, n_k,option)
# rec_cell_inj=track(line_cell_inj)
# rec_cell_ext=track(line_cell_ext)
# rec_cell_mid=track(line_cell_mid)
# rec_cell_mid0=track(line_cell_mid0)
# rec_cell_mid1=track(line_cell_mid1)
# rec_cell_mid2=track(line_cell_mid2)

# # # rec_ds_inj_1turn=track(line_ds_inj)
# # # rec_ds_inj=track(line_ds_inj, num_turns=N_turn*nb_arc)

# line_cell_inj_ref, line_ds_inj_ref = make_optics(file_input, 0, n_k,'var_ref')
# line_cell_ext_ref, line_ds_ext_ref = make_optics(file_input, 1, n_k,'var_ref')
# line_cell_mid_ref, line_ds_mid_ref = make_optics(file_input, 0.5, n_k,'var_ref')



# # #Survey cell
# survey_cell_inj=line_cell_inj.survey(theta0=angle_cell/4)
# survey_cell_ext=line_cell_ext.survey(theta0=angle_cell/4) 
# survey_cell_mid=line_cell_mid.survey(theta0=angle_cell/4)
# survey_cell_mid1=line_cell_mid1.survey(theta0=angle_cell/4)
# survey_cell_mid2=line_cell_mid2.survey(theta0=angle_cell/4)

# survey_cell_inj_ref=line_cell_inj_ref.survey(theta0=angle_cell/4)
# survey_cell_ext_ref=line_cell_ext_ref.survey(theta0=angle_cell/4) 
# survey_cell_mid_ref=line_cell_mid_ref.survey(theta0=angle_cell/4) 


def path_length_plot(time_l):
    path_length_ref=[]
    path_length_k=[]
    line_inj=make_optics(file_input, 0, n_k,'var_k',return_cell='line_cell')
    survey_inj=line_inj.survey(theta0=angle_cell/4)
    for time in time_l:
        # line=make_optics(file_input, time, n_k,'var_ref')[0]
        # survey=line.survey(theta0=angle_cell/4)
        # path=survey['s','_end_point']-survey_cell_inj['s','_end_point']
        # path_length_ref.append(path)
        line=make_optics(file_input, time, n_k,'var_k',return_cell='line_cell')
        survey=line.survey(theta0=angle_cell/4)
        path=survey['s','_end_point']-survey_inj['s','_end_point']
        path_length_k.append(path)
    return(path_length_ref,path_length_k)

def z_final(nk_list):
    zl_ref=[]
    zl_k=[]
    for n_div in nk_list:
        line_inj=make_optics(file_input, 0, n_div,'var_ref')[0]
        survey_inj=line_inj.survey(theta0=angle_cell/4)
        line_ext=make_optics(file_input, 1, n_div,'var_ref')[0]
        survey_ext=line_ext.survey(theta0=angle_cell/4)
        z_fin=survey_ext['Z','_end_point']-survey_inj['Z','_end_point']
        zl_ref.append(z_fin)

        # line_inj=make_optics(file_input, 0, n_div,'var_k')[0]
        # rec1=track(line_inj)
        # line_ext=make_optics(file_input, 1, n_div,'var_k')[0]
        # rec2=track(line_ext)
        # z_fin=rec2.x[0][-1]-rec1.x[0][-1]
        # zl_k.append(z_fin)
    return(zl_ref,zl_k)

# def z_final(nk_list):
#     zl_ref=[]
#     zl_k=[]
#     for n_div in nk_list:
#         # line_inj=make_optics(file_input, 0, n_div,'var_ref')[0]
#         # survey_inj=line_inj.survey(theta0=RCS.cell_angle/4)
#         # line_ext=make_optics(file_input, 1, n_div,'var_ref')[0]
#         # survey_ext=line_ext.survey(theta0=RCS.cell_angle/4)
#         # z_fin=survey_ext['Z','_end_point']-survey_inj['Z','_end_point']
#         # zl_ref.append(z_fin)

#         line_inj=make_optics(file_input, 0, n_div,'var_k')[0]
#         rec1=track(line_inj)
#         line_ext=make_optics(file_input, 1, n_div,'var_k')[0]
#         rec2=track(line_ext)
#         z_fin=rec2.x[0][-1]-rec1.x[0][-1]
#         z_end=rec1.x[0][0]-rec1.x[0][-1]
#         zl_k.append(z_end)
#     return(zl_ref,zl_k)

#Routine to plot path length variation 
t_path=np.linspace(0,1,15)
t_geo_path=np.linspace(0,RCS.t_ext,15)
path_length_geo=[]
for t in t_geo_path:
    path_length_geo.append(RCS.path_length(t)-RCS.path_length(RCS.t_inj))
path_length_ref,path_length_k=path_length_plot(t_path)
plt.figure()
# plt.plot(t_path,np.array(path_length_k)*nb_cell_tot*2*1e3-np.array(path_length_geo)*nb_cell_tot*2*1e3,label='Tracking')
plt.scatter(t_path,np.array(path_length_geo)*nb_cell_tot*2*1e3,label='Geometry',color='red')
plt.xlabel('Time (normalised)')
plt.ylabel('Path length variation [mm]')
plt.legend()
plt.show()

#Routine to plot ending point variation 
# nk_list=[1,11,21,41,51,71,81,101]
# # nk_list=[1]
# zref_final,zk_final=z_final(nk_list)
# plt.figure()
# plt.plot(nk_list,zk_final)
# plt.xlabel('n_kick_dipole')
# plt.ylabel('z_end variation')
# plt.show()
# plt.figure()
# # plt.title('var_ref')
# plt.plot(nk_list,zref_final)
# plt.xlabel('n_kick_dipole')
# plt.ylabel('z_end variation')
# plt.show()

#Survey on an arc
# plt.figure()
# plt.title('Arc')
# plt.plot(survey_ds_inj['Z'], survey_ds_inj['X'], label='inj')
# plt.plot(survey_ds_ext['Z'], survey_ds_ext['X'], label='ext')
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# # plt.ylim(0,10)
# plt.legend()
# plt.show()


# plt.figure()
# # plt.title('var_k: Reference Trajectory from survey')
# # plt.plot(survey_cell_inj['Z'], survey_cell_inj['X'])
# # plt.plot(survey_cell_ext['Z'], survey_cell_ext['X'])
# # plt.plot(survey_cell_mid['Z'], survey_cell_mid['X'])
# # plt.plot(survey_cell_mid1['Z'], survey_cell_mid1['X'])
# # plt.plot(survey_cell_mid2['Z'], survey_cell_mid2['X'])
# # plt.plot(survey_cell_mid0['Z'], survey_cell_mid0['X'])
# plt.xlabel('z [m]')
# plt.ylabel('x [m]')
# # plt.ylim(0,10)
# # plt.legend()
# plt.show()

t_traj=[0,0.1,0.25,0.5,0.75,1]
# plt.figure()
# # plt.title('var_k: Trajectory from tracking')
# plt.plot()
# plt.plot(rec_cell_inj.s[0],rec_cell_inj.x[0],label='Inj')
# plt.plot(rec_cell_ext.s[0],rec_cell_ext.x[0],label='Ext')
# plt.plot(rec_cell_mid.s[0],rec_cell_mid.x[0],label='Mid')
# plt.plot(rec_cell_mid0.s[0],rec_cell_mid0.x[0],label='Mid')
# plt.plot(rec_cell_mid1.s[0],rec_cell_mid1.x[0],label='Mid')
# plt.plot(rec_cell_mid2.s[0],rec_cell_mid2.x[0],label='Mid')
# plt.xlabel('s [m]')
# plt.ylabel('x [m]')
# # plt.legend()
# plt.show()

# tab=line_cell_ext.get_table()
# plt.figure()
# # plt.title('Reconstructed traj: var_ref, var_k')
# plt.plot(survey_cell_inj['s'], np.array(survey_cell_inj['X']+rec_cell_inj.x[0])*1e3)
# plt.plot(survey_cell_ext['s'], np.array(survey_cell_ext['X']+rec_cell_ext.x[0])*1e3)
# plt.plot(survey_cell_mid['s'], np.array(survey_cell_mid['X']+rec_cell_mid.x[0])*1e3)
# plt.plot(survey_cell_inj['s'], np.array(survey_cell_inj['X']+rec_cell_mid0.x[0])*1e3)
# plt.plot(survey_cell_inj['s'], np.array(survey_cell_inj['X']+rec_cell_mid1.x[0])*1e3)
# plt.plot(survey_cell_inj['s'], np.array(survey_cell_inj['X']+rec_cell_mid2.x[0])*1e3)
# # plt.plot([],[],label='Xsuite', color='black')
# plt.plot(survey_cell_inj_ref['s'], survey_cell_inj_ref['X'])
# plt.plot(survey_cell_ext_ref['s'], survey_cell_ext_ref['X'])
# plt.plot(survey_cell_mid_ref['s'], survey_cell_mid_ref['X'])
# plt.axvline(x=tab['s','en1'], color='grey', linestyle='--')
# plt.axvline(x=tab['s','ex1'], color='grey', linestyle='--')
# plt.axvline(x=tab['s','en2'], color='grey', linestyle='--')
# plt.axvline(x=tab['s','ex2'], color='grey', linestyle='--')
# plt.axvline(x=tab['s','en3'], color='grey', linestyle='--')
# plt.axvline(x=tab['s','ex3'], color='grey', linestyle='--')
# plt.text(tab['s','BSC..10'], 0.1, 'SC', horizontalalignment='center')
# plt.text(tab['s','BNC..10'], 0.1, 'NC', horizontalalignment='center')
# plt.text(tab['s','BSC_1..10'], 0.1, 'SC', horizontalalignment='center')
# # for i in t_traj:
# #         plt.scatter(np.real(RCS.zn(i)), np.imag(RCS.zn(i))*1e3,color='black',s=5)
# # plt.scatter([],[],label='Geometry',color='black',s=5)
# plt.xlabel('s [m]')
# plt.ylabel('x [mm]')
# plt.legend()
# plt.show()

# plt.figure()
# # plt.title('Difference of trajectory')
# # plt.plot(survey_cell_inj_ref['s'], survey_cell_inj['X']+rec_inj.x[0]-survey_cell_inj_ref['X'], label='Inj')
# # plt.plot(survey_cell_ext_ref['s'], survey_cell_ext['X']+rec_ext.x[0]-survey_cell_ext_ref['X'], label='Ext')
# plt.xlabel('s [m]')
# plt.ylabel('x [m]')
# plt.legend()
# plt.show()

# print('Path length for ref, var')
# print('Injection', survey_cell_inj_ref['s'][-1], survey_cell_inj['s'][-1])
# print('Inj diff',survey_cell_inj_ref['s'][-1]-survey_cell_inj['s'][-1])
# print('Extraction',survey_cell_ext_ref['s'][-1], survey_cell_ext['s'][-1])
# print('Ext diff',survey_cell_ext_ref['s'][-1]-survey_cell_ext['s'][-1])

# mid_time=(RCS.phi_inj+RCS.phi_ext)/2
# print('SURVEY RESULTS')
# print('INJECTION TIME')
# print('Z',survey_cell_inj['Z','_end_point'])
# print('Path length Xsuite',survey_cell_inj['s','_end_point'])
# print('Path length from geometry',RCS.path_length(RCS.phi_inj)-
#       RCS.LSSS/np.cos(RCS.epsilon(RCS.phi_inj)[0]))
# print('dpath',survey_cell_inj['s','_end_point']-(RCS.path_length(RCS.phi_inj)-
#       RCS.LSSS/np.cos(RCS.epsilon(RCS.phi_inj)[0])))

# # print('0.25 TIME')
# # print('Z',survey_cell_mid1['Z','_end_point'])
# # print('Path length Xsuite',survey_cell_mid1['s','_end_point'])

# # print('0.5 TIME')
# # print('Z',survey_cell_mid['Z','_end_point'])
# # print('Path length Xsuite',survey_cell_mid['s','_end_point'])
# # print('Path length from geometry',RCS.path_length(mid_time)-
# #       RCS.LSSS/np.cos(RCS.epsilon(mid_time)[0]))
# # print('dpath',survey_cell_mid['s','_end_point']- (RCS.path_length(mid_time)-
# #       RCS.LSSS/np.cos(RCS.epsilon(mid_time)[0])))

# # print('0.75 TIME')
# # print('Z',survey_cell_mid2['Z','_end_point'])
# # print('Path length Xsuite',survey_cell_mid2['s','_end_point'])

# print('EXTRACTION TIME')
# print('Z',survey_cell_ext['Z','_end_point'])
# print('Path length Xsuite',survey_cell_ext['s','_end_point'])
# print('Path length from geometry',RCS.path_length(RCS.phi_ext)-
#       RCS.LSSS/np.cos(RCS.epsilon(RCS.phi_ext)[0]))
# print('dpath methods',survey_cell_ext['s','_end_point']-(RCS.path_length(RCS.phi_ext)-
#       RCS.LSSS/np.cos(RCS.epsilon(RCS.phi_ext)[0])))

# print('DZ FROM INJECTION')
# print('t=0.25',survey_cell_mid1['Z','_end_point']-survey_cell_inj['Z','_end_point'])
# print('t=0.5',survey_cell_mid['Z','_end_point']-survey_cell_inj['Z','_end_point'])
# print('t=0.75',survey_cell_mid2['Z','_end_point']-survey_cell_inj['Z','_end_point'])
# print('t=1',survey_cell_ext['Z','_end_point']-survey_cell_inj['Z','_end_point'])

# # print('DPATH FROM INJECTION')
# # print('t=0.25',survey_cell_mid1['s','_end_point']-survey_cell_inj['s','_end_point'])
# # print('t=0.5',survey_cell_mid['s','_end_point']-survey_cell_inj['s','_end_point'])
# # print('t=0.75',survey_cell_mid2['s','_end_point']-survey_cell_inj['s','_end_point'])
# # print('t=1',survey_cell_ext['s','_end_point']-survey_cell_inj['s','_end_point'])

# print('DPATH FROM MID')
# print('t=0.',survey_cell_inj['s','_end_point']-survey_cell_mid['s','_end_point'])
# # print('t=0.25',survey_cell_mid1['s','_end_point']-survey_cell_mid['s','_end_point'])
# # print('t=0.5',survey_cell_mid['s','_end_point']-survey_cell_mid['s','_end_point'])
# # print('t=0.75',survey_cell_mid2['s','_end_point']-survey_cell_mid['s','_end_point'])
# print('t=1',survey_cell_ext['s','_end_point']-survey_cell_mid['s','_end_point'])

