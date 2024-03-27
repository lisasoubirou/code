# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:10:16 2024
@author: LS276867

Study of 1 dipole to try and understanding the difference in trajectories between var_k et var_ref. 
3 cases can be studied using the flags:
    - "thick" : the dipole is not sliced
    - "thin_auto" : the dipole is defined as a thick bend, then sliced by xsuite function
    - "thin": the dipole is defined manually by a sequence of drifts and thin multipole kicks
Functions:
    - make_dip: get geometry and create line_dip
    - track : track on the line when n_turns=1
Printing routines to show results
"""

import numpy as np
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt
import xpart as xp
import sys
sys.path.append('/mnt/c/muco')
from class_geometry.class_geo import Geometry 
import json

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

def make_dip(file_input,time_frac,n_k,dip,option):
    RCS = Geometry(file_input)
    print('Input file', file_input)
    time=time_frac
    energy=RCS.E_inj+(RCS.E_ext-RCS.E_inj)*time_frac


    print('Time (phase)', time)
    print('Energy', energy)

    list_hn=RCS.hn(time) #List of bendings
    list_hn_ref=RCS.hn(t_ref)
    theta=RCS.theta(time) #List of bending angle 
    epsilon=RCS.epsilon(time)
    L_dip_path=RCS.L_dip_path(time)
    L_dip_path_ref=RCS.L_dip_path(t_ref)
    theta_ref=RCS.theta(t_ref)
    pattern=RCS.pattern 

    #Define dipoles manually
    i_BSC=0
    length_sc=RCS.dipole_families['BSC']['length']
    if option == 'var_ref':
        print('Option: var_ref')
        if dip=='thick' or dip=="thin_auto":
            BSC=xt.Bend(k0=list_hn[i_BSC],
                        h=list_hn[i_BSC],
                        length=L_dip_path[0]
                        # length=length_sc
                        )
        elif dip == 'thin':
            i_BSC=pattern.index('BSC')
            BSC=xt.Multipole(length=L_dip_path[0]/n_k,
                            knl=[theta[i_BSC]/n_k], 
                            ksl=[0.0], hxl=theta[i_BSC]/n_k, hyl=0. ) 
            drift_BSC=xt.Drift(length=L_dip_path[0]/2/n_k)
        else:
            raise ValueError("Invalid option: {}".format(dip))
    elif option == 'var_k':
        print('Option: var_k')
        if dip=='thick' or dip=="thin_auto":
            BSC=xt.Bend(k0=list_hn[i_BSC],
                        h=list_hn_ref[i_BSC],
                        length=L_dip_path_ref[0]
                        # length=length_sc
                        )
            print(L_dip_path_ref[0]/2)
        elif dip == 'thin':
            i_BSC=pattern.index('BSC')
            BSC=xt.Multipole(length=L_dip_path_ref[0]/n_k,
                            knl=[theta[i_BSC]/n_k], 
                            ksl=[0.0], hxl=theta_ref[i_BSC]/n_k, hyl=0. ) 
            drift_BSC=xt.Drift(length=L_dip_path_ref[0]/2/n_k)
        else:
            raise ValueError("Invalid option: {}".format(dip))
    else:
        raise ValueError("Invalid option: {}".format(option))
    
    # If we want edges
    eps=RCS.epsilon(t_ref)
    # eps=epsilon
    # hn=RCS.hn(0)
    hn=list_hn*0
    model='full'
    en1=xt.DipoleEdge(k=hn[0], e1=-eps[0], side='entry',model = model)
    ex1=xt.DipoleEdge(k=hn[0], e1=eps[1], side='exit',model =model)

    if dip == 'thick' or dip=="thin_auto":
        BSC_element=[en1,BSC,ex1]
        BSC_name=['en1','dip_bsc','ex1'] 
    elif dip == 'thin':    
        BSC_element=[en1,drift_BSC,  BSC,drift_BSC]+[drift_BSC, BSC, drift_BSC]*(n_k-2) + [
            drift_BSC,BSC,drift_BSC,ex1]
        BSC_name=['en1','drift_bsc', 'dip_bsc','drift_bsc']+['drift_bsc','dip_bsc', 'drift_bsc'
                ]*(n_k-2)+['drift_bsc','dip_bsc','drift_bsc','ex1'] 
    
    line_BSC=xt.Line(
        elements=BSC_element,
        element_names=BSC_name)
    line_BSC.particle_ref = xp.Particles(p0c=energy, #eV
                                        q0=1, mass0=xp.MUON_MASS_EV)
    line_BSC.config.XTRACK_USE_EXACT_DRIFTS = True
    if dip == 'thin_auto':
         line_BSC.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Teapot(n_k))])
    return(line_BSC)

def track(line, num_turns=1):
    line.build_tracker()
    part=line.build_particles()
    if num_turns==1:
        line.track(part,num_turns=1, turn_by_turn_monitor='ONE_TURN_EBE')
    else:
        line.track(part,num_turns=num_turns, turn_by_turn_monitor=True)
    rec=line.record_last_track
    return (rec)

# def compare_methods:

if __name__ == "__main__":
    file_input='/mnt/c/muco/class_geometry/para_RCS_ME.txt'
    time_frac=0
    t_ref=0.5
    RCS = Geometry(file_input)
    n_k=51
    line= make_dip(file_input,time_frac,n_k,'thin_auto','var_k') 
    survey=line.survey(theta0=RCS.cell_angle/4)
    rec=track(line)
    
    line_ref= make_dip(file_input,time_frac,n_k,'thin_auto','var_ref') 
    survey_ref=line_ref.survey(theta0=RCS.cell_angle/4)
    rec_ref=track(line_ref)

    line_thin= make_dip(file_input,time_frac,n_k,'thin','var_k') 
    survey_thin=line_thin.survey(theta0=RCS.cell_angle/4)
    rec_thin=track(line_thin)

    line_thin_ref=make_dip(file_input,time_frac,n_k,'thin','var_ref') 
    survey_thin_ref=line_thin_ref.survey(theta0=RCS.cell_angle/4)
    rec_thin_ref=track(line_thin_ref)

    line_thick=make_dip(file_input,time_frac,n_k,'thick','var_k')
    survey_thick=line_thick.survey(theta0=RCS.cell_angle/4)
    rec_thick=track(line_thick)

    line_thick_ref=make_dip(file_input,time_frac,n_k,'thick','var_ref')
    survey_thick_ref=line_thick_ref.survey(theta0=RCS.cell_angle/4)
    rec_thick_ref=track(line_thick_ref)

    print('COORDINATES AT END OF LINE')
    print('VAR_K THIN AUTO')
    print('x at beg', rec.x[0][0])
    print('px at beg', rec.px[0][0])
    print('x at end', rec.x[0][-1])
    print('px at end', rec.px[0][-1])
    print('VAR_K THIN ')
    print('x at beg', rec_thin.x[0][0])
    print('px at beg', rec_thin.px[0][0])
    print('x at end', rec_thin.x[0][-1])
    print('px at end', rec_thin.px[0][-1])
    print('VAR_K THICK')
    print('x at beg', rec_thick.x[0][0])
    print('px at beg', rec_thick.px[0][0])
    print('x at end', rec_thick.x[0][-1])
    print('px at end', rec_thick.px[0][-1])
    print('VAR_REF THIN AUTO')
    print('x at beg', rec_ref.x[0][0])
    print('px at beg', rec_ref.px[0][0])
    print('x at end', rec_ref.x[0][-1])
    print('px at end', rec_ref.px[0][-1])
    print('VAR_REF THIN ')
    print('x at beg', rec_thin_ref.x[0][0])
    print('px at beg', rec_thin_ref.px[0][0])
    print('x at end', rec_thin_ref.x[0][-1])
    print('px at end', rec_thin_ref.px[0][-1])
    print('VAR_REF THICK')
    print('x at beg', rec_thick_ref.x[0][0])
    print('px at beg', rec_thick_ref.px[0][0])
    print('x at end', rec_thick_ref.x[0][-1])
    print('px at end', rec_thick_ref.px[0][-1])

    #Trajectories
    plt.figure()
    tab=line.get_table()
    plt.plot(survey['s'], survey['X']+rec.x[0], label='rec')
    plt.plot(survey_thick['s'], survey_thick['X']+rec_thick.x[0], label='rec_thick')
    plt.plot(survey_thin['s'], survey_thin['X']+rec_thin.x[0], label='rec_thin')
    # plt.plot(survey_ref['s'], survey_ref['X']+rec_ref.x[0], label='rec_ref')
    # plt.plot(survey_thick_ref['s'], survey_thick_ref['X']+rec_thick_ref.x[0], label='rec_thick_ref')
    plt.xlabel('s [m]')
    plt.ylabel('x [m]')
    plt.legend()
    plt.show()

    # px 
    plt.figure()
    plt.plot(rec.s[0], rec.px[0], label='rec')
    # plt.plot(rec_ref.s[0], rec_ref.px[0], label='rec_ref')
    plt.plot(rec_thick.s[0], rec_thick.px[0], label='rec_thick')
    plt.plot(rec_thin.s[0], rec_thin.px[0], label='rec_thin')
    # plt.plot(rec_thick_ref.s[0], rec_thick_ref.px[0], label='rec_thick_ref')
    plt.ylabel('px')
    plt.xlabel('s [m]')
    plt.legend()
    plt.show()

    # x
    plt.figure()
    plt.plot(rec.s[0], rec.x[0], label='rec')
    # plt.plot(rec_ref.s[0], rec_ref.x[0], label='rec_ref')
    plt.plot(rec_thick.s[0],rec_thick.x[0], label='rec_thick')
    plt.plot(rec_thin.s[0],rec_thin.x[0], label='rec_thin')
    # plt.plot(rec_thick_ref.s[0], rec_thick_ref.x[0], label='rec_thick_ref')
    plt.ylabel('x [m]')
    plt.xlabel('s [m]')
    plt.legend()
    plt.show()

    print('DIFFERENCES BTW METHODS')
    print('Auto thin - thick')
    print('x', rec.x[0][-1]-rec_thick.x[0][-1])
    print('px',rec.px[0][-1]-rec_thick.px[0][-1])
    print('Manually thin - thick')
    print('x',rec_thin.x[0][-1]-rec_thick.x[0][-1])
    print('px',rec_thin.px[0][-1]-rec_thick.px[0][-1])

    angle=RCS.theta(t_ref)-RCS.theta(time_frac)
    print('Angle px - Theoretical angle')
    print('Thin auto', rec.px[0][-1]-angle[0])
    print('Thin manually', rec_thin.px[0][-1]-angle[0])
    print('Thick', rec_thick.px[0][-1]-angle[0])