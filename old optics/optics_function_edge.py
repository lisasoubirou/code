# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:10:16 2024
@author: LS276867

Version of optics_function that was tested to add edges. Now use optics_function.
"""

import numpy as np
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt
import xpart as xp
import sys
# sys.path.append('/mnt/c/muco')
from class_geometry.class_geo import Geometry 
import json

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

#Call class_geo
file_input='/mnt/c/muco/code/class_geometry/para_RCS_ME.txt'
time_frac=0
RCS = Geometry(file_input)

def make_optics(file_input,time_frac,n_k,option):
    RCS = Geometry(file_input)
    print('Input file', file_input)
    time=time_frac
    energy=RCS.E_inj+(RCS.E_ext-RCS.E_inj)*time_frac
    frequency_rf=RCS.RF_freq

    print('Time (phase)', time)
    print('Energy', energy)

    #Get data from class_geo
    Ldd=RCS.dipole_spacing 
    Ls=RCS.LSSS #Length of short straight section
    # Lqd=RCS.QP_dipole_spacing
    Lqd=RCS.L_qp_dip_path(time)
    Lins=RCS.insertion_length
    Lcell=RCS.cell_length
    list_hn=RCS.hn(time) #List of bendings
    theta=RCS.theta(time) #List of bending angle 
    epsilon=-RCS.epsilon(time)
    L_dd_path=RCS.L_dd_path(time) 
    L_dip_path=RCS.L_dip_path(time)
    L_dip_path_ref=RCS.L_dip_path(0)
    theta_ref=RCS.theta(0)
    pattern=RCS.pattern 
    nb_cell_tot=RCS.nb_cell_arcs
    nb_arc=RCS.nb_arc
    n_cav_arc=2

    #RF data
    N_turn_rcs=55
    N_frac=1
    n_turns = N_turn_rcs*nb_arc*N_frac
    energy_increment_per_arc = (RCS.E_ext-RCS.E_inj)/n_turns
    sync_phase=45
    phase=180-sync_phase
    volt_arc=energy_increment_per_arc/np.sin(phase*np.pi/180)
    volt_cav=volt_arc/n_cav_arc

    #Define elements
    mu=np.pi/2 #Phase advance 
    f=RCS.cell_length/(4*np.sin(mu/2)) #Focusing strength of quadrupole
    drift_qd=xt.Drift(length=Lqd)
    drift_dd=xt.Drift(length=L_dd_path[1])
    drift_s=xt.Drift(length=Ls/2)
    drift_s_sxt=xt.Drift(length=Ls/2-Lqd)
    drift_r=xt.Drift(length=Lins/2/2)

    quad_f=xt.Multipole(knl=[0., 1/f], ksl=[0., 0.])
    quad_d=xt.Multipole(knl=[0., -1/f], ksl=[0., 0.])
    quad_f1=xt.Multipole(knl=[0., 1/f], ksl=[0., 0.])
    quad_d1=xt.Multipole(knl=[0., -1/f], ksl=[0., 0.])
    quad_f2=xt.Multipole(knl=[0., 1/f], ksl=[0., 0.])
    quad_d2=xt.Multipole(knl=[0., -1/f], ksl=[0., 0.])
    quad_f3=xt.Multipole(knl=[0., 1/f], ksl=[0., 0.])
    quad_d3=xt.Multipole(knl=[0., -1/f], ksl=[0., 0.])

    s_f=1/(f*0.5)*0
    s_d=-s_f
    sxt_d=xt.Multipole(knl=[0., 0.,s_d], ksl=[0., 0.,0.])
    sxt_f=xt.Multipole(knl=[0., 0.,s_f], ksl=[0., 0.,0.])

    V=0
    freq=0
    phi=0
    cavity=xt.Cavity(voltage=V, frequency=freq, lag=phi)
    # RF_acc=xt.ReferenceEnergyIncrease(Delta_p0c=energy_increment_per_arc/n_cav_arc)
    RF_acc=xt.ReferenceEnergyIncrease(Delta_p0c=0)

    beg = xt.Marker()
    end = xt.Marker()

    #Define dipoles manually
    if option == 'var_ref':
        print('Option: var_ref')
        length_nc=RCS.dipole_families['BNC']['length']
        i_BNC=pattern.index('BNC')
        BNC=xt.Multipole(length=L_dip_path[1]/n_k, 
                        knl=[theta[i_BNC]/n_k], 
                        ksl=[0.0], hxl=theta[i_BNC]/n_k, hyl=0. )   
        drift_BNC=xt.Drift(length=L_dip_path[1]/2/n_k)
        if len(set(pattern))>1:
            length_sc=RCS.dipole_families['BSC']['length']
            i_BSC=pattern.index('BSC')
            BSC=xt.Multipole(length=L_dip_path[0]/n_k,
                            knl=[theta[i_BSC]/n_k], 
                            ksl=[0.0], hxl=theta[i_BSC]/n_k, hyl=0. ) 
            drift_BSC=xt.Drift(length=L_dip_path[0]/2/n_k)
    elif option == 'var_k':
        print('Option: var_k')
        length_nc=RCS.dipole_families['BNC']['length']
        i_BNC=pattern.index('BNC')
        BNC=xt.Multipole(length=L_dip_path_ref[1]/n_k, 
                        knl=[theta[i_BNC]/n_k], 
                        ksl=[0.0], hxl=theta_ref[i_BNC]/n_k, hyl=0. )   
        drift_BNC=xt.Drift(length=L_dip_path_ref[1]/2/n_k)
        if len(set(pattern))>1:
            length_sc=RCS.dipole_families['BSC']['length']
            i_BSC=pattern.index('BSC')
            BSC=xt.Multipole(length=L_dip_path_ref[0]/n_k,
                            knl=[theta[i_BSC]/n_k], 
                            ksl=[0.0], hxl=theta_ref[i_BSC]/n_k, hyl=0. ) 
            drift_BSC=xt.Drift(length=L_dip_path_ref[0]/2/n_k)
    else:
        raise ValueError("Invalid option: {}".format(option))

    eps=-RCS.epsilon(0.5)
    # eps=epsilon
    # hn=RCS.hn(0)
    hn=list_hn
    model='linear'
    en1=xt.DipoleEdge(k=hn[0], e1=eps[0], side='entry',model = model)
    ex1=xt.DipoleEdge(k=hn[0], e1=eps[1], side='exit',model =model)
    en2=xt.DipoleEdge(k=hn[1], e1=eps[1], side='entry',model =model)
    ex2=xt.DipoleEdge(k=hn[1], e1=eps[2], side='exit',model =model)
    en3=xt.DipoleEdge(k=hn[2], e1=eps[2], side='entry',model =model)
    ex3=xt.DipoleEdge(k=hn[2], e1=eps[3], side='exit',model =model)

    #FODO cell
    if n_k==1:
        cell=[drift_qd]+[en1,drift_BSC,BSC,drift_BSC,ex1]+ [drift_dd]+ [en2,drift_BNC, BNC, drift_BNC,ex2]+ [
        drift_dd]+[en3,drift_BSC, BSC, drift_BSC,ex3]+[drift_qd]
        cell_name=['drift_qd']+['en1','drift_BSC','BSC','drift_BSC','ex1']+ ['drift_dd']+ ['en2','drift_BNC', 'BNC', 'drift_BNC','ex2']+ [
        'drift_dd']+['en3','drift_BSC','BSC', 'drift_BSC','ex3']+['drift_qd']
    else:
        cell=[drift_qd]+[en1,drift_BSC,  BSC,drift_BSC]+[drift_BSC, BSC, drift_BSC]*(n_k-2) + [drift_BSC,BSC,drift_BSC,ex1
            ]+[drift_dd]+[
        en2, drift_BNC, BNC,drift_BNC]+ [drift_BNC, BNC, drift_BNC]*(n_k-2) +[drift_BNC,BNC,drift_BNC,ex2]+ [drift_dd
        ]+ [en3,drift_BSC,  BSC,drift_BSC]+[drift_BSC, BSC, drift_BSC]*(n_k-2) + [drift_BSC,BSC,drift_BSC,ex3]+[drift_qd]
        cell_name=['drift_qd']+[
        'en1','drift_bsc', 'dip_bsc','drift_bsc']+['drift_bsc','dip_bsc', 'drift_bsc']*(n_k-2)+['drift_bsc','dip_bsc','drift_bsc','ex1'] + ['drift_dd']+[
        'en2','drift_bnc', 'dip_bnc','drift_bnc']+['drift_bnc','dip_bnc', 'drift_bnc']*(n_k-2)+['drift_bnc','dip_bnc','drift_bnc','ex2']+['drift_dd']+[
        'en3','drift_bsc', 'dip_bsc','drift_bsc']+['drift_bsc','dip_bsc', 'drift_bsc']*(n_k-2)+['drift_bsc','dip_bsc','drift_bsc','ex3']+['drift_qd']
    
    FODO_elements=[quad_f,drift_s]+cell+[drift_s, quad_d,drift_s]+cell+[drift_s]
    FODO_names=['quad_f','drift_s']+cell_name+['drift_s', 'quad_d','drift_s']+cell_name+['drift_s']

    line_cell=xt.Line(
        elements=cell,
        element_names=cell_name)
    line_cell.particle_ref = xp.Particles(p0c=energy, #eV
                                        q0=1, mass0=xp.MUON_MASS_EV)
    line_FODO=xt.Line(
        elements=FODO_elements,
        element_names=FODO_names)
    line_FODO.particle_ref = xp.Particles(p0c=energy, #eV
                                        q0=1, mass0=xp.MUON_MASS_EV)

    tw_FODO = line_FODO.twiss(method='4d') 
    index_qpd=np.where(tw_FODO['name']=='quad_d')[0][0]
    betx_fodo=tw_FODO['betx'][index_qpd]
    bety_fodo=tw_FODO['bety'][index_qpd]     
    dx_fodo=tw_FODO['dx'][index_qpd]     
    # plot_twiss(tw_FODO,line_FODO)

    #Transfer
    tr_elements= [beg]+[quad_f1 ,drift_s]+cell+[drift_s, quad_d1,drift_s]+cell+[drift_s,quad_f2,drift_s
                ]+cell+[drift_s, quad_d2, drift_s]+cell+[drift_s, quad_f3,drift_r, cavity, drift_r, quad_d3]+[end]
    tr_names=['beg']+['quad_f', 'drift_s']+cell_name+['drift_s', 'quad_d','drift_s']+cell_name+['drift_s',
            'quad_f','drift_s']+cell_name+['drift_s', 'quad_d', 'drift_s']+cell_name+['drift_s', 'quad_f',
                        'drift_r','cavity','drift_r', 'quad_d']+['end']

    line_tr=xt.Line(
        elements=tr_elements,
        element_names=tr_names)
    line_tr.particle_ref = xp.Particles(p0c=energy, #eV
                                        q0=1, mass0=xp.MUON_MASS_EV)

    #Matches FODO lattices to the transfer line: on first quad of the transfer == output optics of FODO
    tw_init_FODO = tw_FODO.get_twiss_init(0)
    tw_init_FODO.element_name = 'beg'
    tw_tr=line_tr.twiss(start=line_tr.element_names[0],end='_end_point', method='4d',
                        # ele_init= "drift_ini",
                        init=tw_init_FODO
                        )

    #Init of variables to be varied
    line_tr._init_var_management()
    line_tr.vars['k_f0'] = 1/f
    line_tr.vars['k_d1'] = -1/f
    line_tr.vars['k_f1'] = 1/f
    line_tr.vars['k_d2'] = -1/f
    line_tr.vars['k_f2'] = 1/f
    line_tr.vars['k_d3'] = -1/f/2   #Half quad at the beginning and end of line_ds

    line_tr.element_refs['quad_f'].knl[1] = line_tr.vars['k_f0']
    line_tr.element_refs['quad_d'].knl[1] = line_tr.vars['k_d1']
    line_tr.element_refs['quad_f_1'].knl[1] = line_tr.vars['k_f1']
    line_tr.element_refs['quad_d_1'].knl[1] = line_tr.vars['k_d2']
    line_tr.element_refs['quad_f_2'].knl[1] = line_tr.vars['k_f2']
    line_tr.element_refs['quad_d_2'].knl[1] = line_tr.vars['k_d3']

    # Matching
    match_tr=line_tr.match(
                start='beg',
                end='_end_point', 
                init=tw_init_FODO,
                vary=xt.VaryList(['k_f0','k_d1','k_f1','k_d2','k_f2','k_d3'],
                                step=1e-8,
                                tag='quad'),
                targets=[xt.TargetSet(dx=0,dpx=0,alfx=0,alfy=0, at='end',tol=1e-9)],
                # verbose=True
                )
    print('RESULTS FROM TR MATCHING')
    match_tr.target_status()
    match_tr.vary_status()

    #Calculate twiss after matching
    tw_match=line_tr.twiss(start=line_tr.element_names[0],end='_end_point', method='4d',
                        init=tw_init_FODO
                        )
    # plot_twiss_single(tw_match)

    # Dispersion suppressor lattice 
    ad_2_elements=tr_elements[1:-1]
    ad_2_names=tr_names[1:-1]
    ad_1_elements= tr_elements[1:-1]
    ad_1_elements=ad_1_elements[::-1]
    ad_1_names= tr_names[1:-1]
    ad_1_names=ad_1_names[::-1]

    ds_elements=ad_1_elements+FODO_elements[1:]+FODO_elements*3+ad_2_elements+[end] 
    ds_names=ad_1_names+FODO_names[1:]+FODO_names*3+ad_2_names+['end'] 

    line_ds=xt.Line(
        elements=ds_elements,
        element_names=ds_names)
    line_ds.particle_ref = xp.Particles(p0c=energy, #eV
                                        q0=1, mass0=xp.MUON_MASS_EV)
    
    tab_bf=line_ds.get_table()
    line_ds.insert_element('sxt_d_1',sxt_d,at_s=tab_bf['s','quad_d_2'])
    line_ds.insert_element('sxt_d_2',sxt_d,at_s=tab_bf['s','quad_d_4'])
    line_ds.insert_element('sxt_f_1',sxt_f,at_s=tab_bf['s','quad_f_4'])
    line_ds.insert_element('sxt_f_2',sxt_f,at_s=tab_bf['s','quad_f_6'])
    line_ds.insert_element('RF_acc', RF_acc, at_s=tab_bf['s','cavity']-1e-5)
    line_ds.insert_element('RF_acc_1', RF_acc, at_s=tab_bf['s','cavity_1']+1e-5)

    line_ds._init_var_management()
    # #Set knobs to be varied
    line_ds.vars['k_f0'] = line_ds['quad_f_2'].knl[1]
    line_ds.vars['k_d1'] = line_ds['quad_d_2'].knl[1]
    line_ds.vars['k_f1'] = line_ds['quad_f_1'].knl[1]
    line_ds.vars['k_d2'] = line_ds['quad_d_1'].knl[1]
    line_ds.vars['k_f2'] = line_ds['quad_f'].knl[1]
    line_ds.vars['k_d3'] = line_ds['quad_d'].knl[1]
    line_ds.vars['s_d'] = -1/(f*0.5)*0
    line_ds.vars['s_f'] = 1/(f*0.5)*0

    #Attach knobs to elements
    line_ds.element_refs['quad_f_2'].knl[1] = line_ds.vars['k_f0']
    line_ds.element_refs['quad_d_2'].knl[1] = line_ds.vars['k_d1']
    line_ds.element_refs['quad_f_1'].knl[1] = line_ds.vars['k_f1']
    line_ds.element_refs['quad_d_1'].knl[1] = line_ds.vars['k_d2']
    line_ds.element_refs['quad_f'].knl[1] = line_ds.vars['k_f2']
    line_ds.element_refs['quad_d'].knl[1] = line_ds.vars['k_d3']

    line_ds.element_refs['quad_f_6'].knl[1] = line_ds.vars['k_f0']
    line_ds.element_refs['quad_d_7'].knl[1] = line_ds.vars['k_d1']
    line_ds.element_refs['quad_f_7'].knl[1] = line_ds.vars['k_f1']
    line_ds.element_refs['quad_d_8'].knl[1] = line_ds.vars['k_d2']
    line_ds.element_refs['quad_f_8'].knl[1] = line_ds.vars['k_f2']
    line_ds.element_refs['quad_d_9'].knl[1] = line_ds.vars['k_d3']

    line_ds.element_refs['sxt_d_1'].knl[2] = line_ds.vars['s_d']
    line_ds.element_refs['sxt_f_1'].knl[2] = line_ds.vars['s_f']
    line_ds.element_refs['sxt_d_2'].knl[2] = line_ds.vars['s_d']
    line_ds.element_refs['sxt_f_2'].knl[2] = line_ds.vars['s_f']

    # match_ds_4d=line_ds.match(vary=xt.VaryList(['k_f0','k_d1','k_f1','k_d2','k_f2','k_d3'],
    #                             step=1e-8,
    #                             tag='quad'),
    #             targets=[xt.Target(tar='dx', at='end', value=0., tol=1e-9,tag='DS'),
    #                     xt.TargetSet(betx=betx_fodo,bety=bety_fodo,dx=dx_fodo, at='quad_d_3', tol=1e-9, tag='FODO'),
    #                     xt.TargetSet(qx=2.60, qy=2.23, tol=1e-3, tag='tune')],
    #             #solve=False,
    #             method='4d',
    #             # verbose=True
    #             )
    # match_ds_4d.target_status()
    # match_ds_4d.vary_status()

    # match_ds_4d=line_ds.match(vary=xt.VaryList(['s_d','s_f'],
    #                             step=1e-8,
    #                             tag='sxt'),
    #             targets=[
    #                     xt.TargetSet(qx=2.60, qy=2.23, tol=1e-6, tag='tune'),
    #                     xt.TargetSet(dqx=0, dqy=0, tol=1e-6, tag='chroma')],
    #             # solve=False,
    #             method='4d',
    #             # verbose=True
    #             )

    # print('RESULTS DS MATCH 4D')
    # match_ds_4d.target_status()
    # match_ds_4d.vary_status()
    tw_ds_4d=line_ds.twiss( method='4d')

    #Import line to json
    # line_ds.to_json('lattice_disp_suppr.json') 

   # Cavity settings
    frequency = np.round(frequency_rf*tw_ds_4d.T_rev0)/tw_ds_4d.T_rev0
    line_ds['cavity'].frequency=frequency
    line_ds['cavity'].lag=phase
    line_ds['cavity'].voltage=volt_cav
    line_ds['cavity_1'].frequency=frequency
    line_ds['cavity_1'].lag=phase
    line_ds['cavity_1'].voltage=volt_cav
    tab = line_ds.get_table()

    # match_ds_6d=line_ds.match(vary=xt.VaryList(['k_f0','k_d1','k_f1','k_d2','k_f2','k_d3'],
    #                             step=1e-8,
    #                             tag='quad'),
    #             targets=[xt.TargetSet(dx=0, at='end', tol=1e-9,tag='DS'),
    #                     # xt.TargetSet(betx=betx_fodo,bety=bety_fodo,dx=dx_fodo, at='quad_d_3', tol=1e-9, tag='FODO'),
    #                     xt.TargetSet(qx=2.60, qy=2.23, tol=1e-6, tag='tune')],
    #             method='6d',
    #             matrix_stability_tol=5e-3,
    #             # verbose=True
    #             )
    # match_ds_6d.target_status()
    # match_ds_6d.vary_status()

    # match_ds_6d=line_ds.match(vary=xt.VaryList(['s_d','s_f'],
    #                         step=1e-8,
    #                         tag='sxt'),
    #             targets=[xt.TargetSet(qx=2.60, qy=2.23, tol=1e-6, tag='tune'),
    #                     xt.TargetSet(dqx=0, dqy=0, tol=1e-6, tag='chroma')],
    #             #solve=False,
    #             method='6d',
    #             matrix_stability_tol=5e-3,
    #             #verbose=True
    #             )

    print('RESULTS DS MATCH 6D')
    # match_ds_6d.target_status()
    # match_ds_6d.vary_status()
    tw_ds_6d=line_ds.twiss(method='6d', matrix_stability_tol=5e-3)
    # plot_twiss_single(tw_ds_6d)
    # line_ds.to_json('lattice_disp_suppr_6d.json')

    print('GLOBAL PARAMETER')
    print('MCF=', round(tw_ds_6d['momentum_compaction_factor'], 6))
    print('Qx=', round(tw_ds_6d['qx'], 5))
    print('Qy', round(tw_ds_6d['qy'], 5))
    print('dqx', round(tw_ds_6d['dqx'], 2))
    print('dqy', round(tw_ds_6d['dqy'], 2))
    return(line_cell,line_ds,tw_ds_4d,tw_ds_6d)

def plot_twiss(tw,line):
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)
    plt.plot(tw['s'], tw['betx'], label='betx')
    plt.plot(tw['s'], tw['bety'], label='bety')
    for i, el in enumerate(line.element_names):
        if 'quad' in el:
            plt.axvline(x=tw['s'][i], color='grey', linestyle='--')
        elif 'sxt' in el:
            plt.axvline(x=tw['s'][i], color='red', linestyle='--')
        elif 'cavity' in el:
            plt.axvline(x=tw['s'][i], color='green', linestyle='--')
    plt.xlabel('s [m]')
    plt.ylabel('betx, bety [m]')
    plt.legend(loc='upper right')
    plt.subplot(2, 1, 2)
    plt.plot(tw['s'], tw['dx'], label='Dx')
    plt.plot(tw['s'], tw['dy'], label='Dy')
    for i, el in enumerate(line.element_names):
        if 'quad' in el:
            plt.axvline(x=tw['s'][i], color='grey', linestyle='--')
        elif 'sxt' in el:
            plt.axvline(x=tw['s'][i], color='red', linestyle='--')
        elif 'cavity' in el:
            plt.axvline(x=tw['s'][i], color='green', linestyle='--')
    plt.xlabel('s [m]')
    plt.ylabel('Dx, Dy [m]')
    plt.legend(loc='upper right')

def plot_twiss_single(tw):
    plt.figure()
    plt.plot(tw['s'],tw['betx'], label='betx')
    plt.plot(tw['s'],tw['bety'], label='bety')
    plt.xlabel('s [m]')
    plt.ylabel('betx, bety [m]')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(tw['s'],tw['dx'], label='Dx')
    plt.plot(tw['s'],tw['dy'], label='Dy')
    plt.xlabel('s [m]')
    plt.ylabel('Dx, Dy [m]')
    plt.legend()
    plt.show()

def z_final(nk_list):
    zl_ref=[]
    zl_k=[]
    for n_div in nk_list:
        # line_inj=make_optics(file_input, 0, n_div,'var_ref')[0]
        # survey_inj=line_inj.survey(theta0=RCS.cell_angle/4)
        # line_ext=make_optics(file_input, 1, n_div,'var_ref')[0]
        # survey_ext=line_ext.survey(theta0=RCS.cell_angle/4)
        # z_fin=survey_ext['Z','_end_point']-survey_inj['Z','_end_point']
        # zl_ref.append(z_fin)

        line_inj=make_optics(file_input, 0, n_div,'var_k')[0]
        rec1=track(line_inj)
        line_ext=make_optics(file_input, 1, n_div,'var_k')[0]
        rec2=track(line_ext)
        z_fin=rec2.x[0][-1]-rec1.x[0][-1]
        z_end=rec1.x[0][0]-rec1.x[0][-1]
        zl_k.append(z_end)
    return(zl_ref,zl_k)

def track(line, num_turns=1):
    line.build_tracker()
    part=line.build_particles()
    if num_turns==1:
        line.track(part,num_turns=1, turn_by_turn_monitor='ONE_TURN_EBE')
    else:
        line.track(part,num_turns=num_turns, turn_by_turn_monitor=True)
    rec=line.record_last_track
    return (rec)

if __name__ == "__main__":
    # line_cell, line_ds = make_optics(file_input,time_frac,1,'var_ref')
    # tw_ds=line_ds.twiss(method='6d', matrix_stability_tol=5e-3)
    # nb_arc=26
    # plot_twiss_single(tw_ds)
    # plt.figure()
    # plt.plot(tw_ds.s,tw_ds.zeta*1e3)
    # plt.xlabel('s [m]')
    # plt.ylabel('zeta [mm] ')
    # plt.show()
    # plt.figure()
    # plt.plot(tw_ds.s,tw_ds.delta)
    # plt.xlabel('s [m]')
    # plt.ylabel('delta [10-3] ')
    # plt.show()
    # print('TUNES')
    # print(f'RCS Qx: {tw_ds["qx"]*nb_arc:.3f}')
    # print(f'Arc Qx: {tw_ds["qx"]:.3f}')
    # print(f'RCS Qy: {tw_ds["qy"]*nb_arc:.3f}')
    # print(f'Arc Qy: {tw_ds["qy"]:.3f}')
    # print(f'Arc Qs: {tw_ds["qs"]:.4f}')
    # print(f'RCS Qs: {tw_ds["qs"]*nb_arc:.3f}')
    # print(f'MCF: {tw_ds["momentum_compaction_factor"]:.5f}')
    
    nk_list=[1,21,51,101]
    # nk_list=[1]
    # zref_final,zk_final=z_final(nk_list)
   
    # print(zref_final)
    # plt.figure()
    # plt.plot(nk_list,zk_final)
    # plt.xlabel('n_kick_dipole')
    # plt.ylabel('z_end variation')
    # plt.show()
  
    # plt.figure()
    # plt.scatter(nk_list,zref_final)
    # plt.xlabel('n_kick_dipole')
    # plt.ylabel('z_end variation')
    # plt.show()

    # n_k=101
    # option='var_k'
    # line_cell_inj, line_ds_inj = make_optics(file_input, 0, n_k, option)
    # line_cell_ext, line_ds_ext = make_optics(file_input, 1, n_k, option)
    # survey_cell_inj=line_cell_inj.survey(theta0=RCS.cell_angle/4)
    # survey_cell_ext=line_cell_ext.survey(theta0=RCS.cell_angle/4) 
    # rec_cell_inj=track(line_cell_inj)
    # rec_cell_ext=track(line_cell_ext)
    # plt.figure()
    # # plt.plot(survey_cell_inj['Z'], survey_cell_inj['X'],label='inj')
    # # plt.plot(survey_cell_ext['Z'], survey_cell_ext['X'],label='ext')
    # plt.plot(survey_cell_inj['s'], np.array(survey_cell_inj['X']+rec_cell_inj.x[0])*1e3)
    # plt.plot(survey_cell_ext['s'], np.array(survey_cell_ext['X']+rec_cell_ext.x[0])*1e3)
    # plt.legend()
    # plt.show()

    # # print(zk_final)
    # line_cell_inj_ref, line_ds_inj_ref = make_optics(file_input, 0, n_k,'var_ref')
    # line_cell_ext_ref, line_ds_ext_ref = make_optics(file_input, 1, n_k,'var_ref')
    # line_cell_mid_ref, line_ds_mid_ref = make_optics(file_input, 0.5, n_k,'var_ref')

    # survey_cell_inj_ref=line_cell_inj_ref.survey(theta0=RCS.cell_angle/4)
    # survey_cell_ext_ref=line_cell_ext_ref.survey(theta0=RCS.cell_angle/4) 
    # survey_cell_mid_ref=line_cell_mid_ref.survey(theta0=RCS.cell_angle/4) 

    # plt.figure()
    # # plt.title('Difference of trajectory')
    # tab=line_cell_inj.get_table()
    # plt.plot(survey_cell_inj_ref['s'], survey_cell_inj['X']+rec_cell_inj.x[0]-survey_cell_inj_ref['X'], label='Inj')
    # plt.plot(survey_cell_ext_ref['s'], survey_cell_ext['X']+rec_cell_ext.x[0]-survey_cell_ext_ref['X'], label='Ext')
    # # for i, el in enumerate(line_cell_inj.element_names):
    # #     if 'en' in el:
    # #         plt.axvline(x=tab['s',el], color='grey', linestyle='--')
    # #     if 'ex' in el:
    # #         plt.axvline(x=tab['s',el], color='grey', linestyle='--')
    # plt.xlabel('s [m]')
    # plt.ylabel('x [m]')
    # plt.legend()
    # plt.show()

    # plt.figure()
    # plt.plot(survey_cell_inj['Z'],survey_cell_inj['Z']-survey_cell_mid_ref['Z'], "b.");
    # plt.show()    
    line_cell,line_ds,tw_4d,tw_6d = make_optics(file_input,time_frac,21,'var_k')
