# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:10:16 2024

@author: LS276867
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

def make_optics(file_input,time_frac,n_slice,option):
    RCS = Geometry(file_input)
    print('Input file', file_input)
    time=time_frac
    t_ref=0.5
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
    list_hn_ref=RCS.hn(t_ref) #List of reference bendings
    theta=RCS.theta(time) #List of bending angle 
    epsilon=RCS.epsilon(time)
    L_dd_path=RCS.L_dd_path(time) 
    L_dip_path=RCS.L_dip_path(time)
    L_dip_path_ref=RCS.L_dip_path(t_ref)
    theta_ref=RCS.theta(t_ref)
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
        i_BNC=pattern.index('BNC')
        BNC=xt.Bend(k0=list_hn[i_BNC],
                        h=list_hn[i_BNC],
                        length=L_dip_path[i_BNC]
                        # length=length_sc
                        )   
        if len(set(pattern))>1:
            i_BSC=pattern.index('BSC')
            BSC=xt.Bend(k0=list_hn[i_BSC],
                        h=list_hn[i_BSC],
                        length=L_dip_path[i_BSC]
                        # length=length_sc
                        )
    elif option == 'var_k':
        print('Option: var_k')
        i_BNC=pattern.index('BNC')
        BNC=xt.Bend(k0=list_hn[i_BNC],
                        h=list_hn_ref[i_BNC],
                        length=L_dip_path_ref[i_BNC]
                        # length=length_sc
                        )   
        if len(set(pattern))>1:
            i_BSC=pattern.index('BSC')
            BSC=xt.Bend(k0=list_hn[i_BSC],
                        h=list_hn_ref[i_BSC],
                        length=L_dip_path_ref[i_BSC]
                        # length=length_sc
                        )
    else:
        raise ValueError("Invalid option: {}".format(option))
    
    # If we want edges
    eps=RCS.epsilon(t_ref)
    # eps=epsilon
    # hn=RCS.hn(0)
    hn=list_hn
    model='full'
    en1=xt.DipoleEdge(k=hn[0], e1=-eps[0], side='entry',model = model)
    ex1=xt.DipoleEdge(k=hn[0], e1=eps[1], side='exit',model =model)
    en2=xt.DipoleEdge(k=hn[1], e1=-eps[1], side='entry',model =model)
    ex2=xt.DipoleEdge(k=hn[1], e1=eps[2], side='exit',model =model)
    en3=xt.DipoleEdge(k=hn[2], e1=-eps[2], side='entry',model =model)
    ex3=xt.DipoleEdge(k=hn[2], e1=eps[3], side='exit',model =model)

    cell=[drift_qd]+[en1,BSC,ex1]+ [drift_dd]+ [en2, BNC,ex2]+ [drift_dd]+[en3, BSC,ex3]+[drift_qd]
    cell_name=['drift_qd']+['en1','BSC','ex1']+ ['drift_dd']+ ['en2', 'BNC','ex2']+ ['drift_dd']+['en3','BSC','ex3']+['drift_qd']
       
    FODO_elements=[quad_f,drift_s]+cell+[drift_s, quad_d,drift_s]+cell+[drift_s]
    FODO_names=['quad_f','drift_s']+cell_name+['drift_s', 'quad_d','drift_s']+cell_name+['drift_s']

    line_cell=xt.Line(
        elements=cell,
        element_names=cell_name)
    line_cell.particle_ref = xp.Particles(p0c=energy, #eV
                                        q0=1, mass0=xp.MUON_MASS_EV)
    line_cell.config.XTRACK_USE_EXACT_DRIFTS = True
    if n_slice > 0:
        line_cell.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Teapot(n_slice))])


    line_FODO=xt.Line(
        elements=FODO_elements,
        element_names=FODO_names)
    line_FODO.particle_ref = xp.Particles(p0c=energy, #eV
                                        q0=1, mass0=xp.MUON_MASS_EV)
    line_FODO.config.XTRACK_USE_EXACT_DRIFTS = True

    tw_FODO = line_FODO.twiss(method='4d') 
    betx_qd_fodo=tw_FODO['betx', 'quad_d']
    bety_qd_fodo=tw_FODO['bety', 'quad_d'] 
    dx_qd_fodo=tw_FODO['dx', 'quad_d']  
    betx_qf_fodo=tw_FODO['betx', 'quad_f']
    bety_qf_fodo=tw_FODO['bety', 'quad_f'] 
    dx_qf_fodo=tw_FODO['dx', 'quad_f'] 
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
    line_tr.config.XTRACK_USE_EXACT_DRIFTS = True

    #Matches FODO lattices to the transfer line: on first quad of the transfer == output optics of FODO
    tw_init_FODO = tw_FODO.get_twiss_init(0)
    tw_init_FODO.element_name = 'beg'
    tw_tr=line_tr.twiss(start=line_tr.element_names[0],end='_end_point', method='4d',
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

    tab_bf=line_ds.get_table()
    line_ds.insert_element('sxt_d_1',sxt_d,at_s=tab_bf['s','quad_d_2'])
    line_ds.insert_element('sxt_d_2',sxt_d,at_s=tab_bf['s','quad_d_4'])
    line_ds.insert_element('sxt_f_1',sxt_f,at_s=tab_bf['s','quad_f_4'])
    line_ds.insert_element('sxt_f_2',sxt_f,at_s=tab_bf['s','quad_f_6'])
    line_ds.insert_element('RF_acc', RF_acc, at_s=tab_bf['s','cavity']-1e-5)
    line_ds.insert_element('RF_acc_1', RF_acc, at_s=tab_bf['s','cavity_1']+1e-5)

    line_ds.particle_ref = xp.Particles(p0c=energy, #eV
                                        q0=1, mass0=xp.MUON_MASS_EV)
    line_ds.config.XTRACK_USE_EXACT_DRIFTS = True
    if n_slice > 0:
        line_ds.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Teapot(n_slice))])

    # tw_ds_bf=line_ds.twiss( method='4d')

    line_ds._init_var_management()
    #Set knobs to be varied
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

    match_ds_4d=line_ds.match(vary=xt.VaryList(['k_f0','k_d1','k_f1','k_d2','k_f2','k_d3'],
                                step=1e-8,
                                tag='quad'),
                targets=[
                        xt.Target(tar='dx', at='end', value=0., tol=1e-9,tag='DS'),
                        xt.TargetSet(betx=betx_qd_fodo,bety=bety_qd_fodo, at='quad_d_3', tol=1e-9, tag='FODO'),
                        xt.TargetSet(qx=2.60, qy=2.23, tol=1e-3, tag='tune')
                        ],
                #solve=False,
                method='4d',
                # verbose=True
                )
    print('MATCHING QUAD DS 4D')
    match_ds_4d.target_status()
    match_ds_4d.vary_status()

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
    # print('MATCHING SXT DS 4D')
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

    h_rf = np.round(frequency_rf*tw_ds_4d.T_rev0*nb_arc) #data tw_ds on 1 arc
    Delta_p0c=energy_increment_per_arc/n_cav_arc
    dz_acc=xt.ZetaShift(dzeta=-0.000497805162122214/2)
    line_ds.discard_tracker()
    # line_ds.insert_element('dz_acc', dz_acc, at_s=tab['s','cavity'])
    line_ds['RF_acc'].Delta_p0c=Delta_p0c 
    line_ds['RF_acc_1'].Delta_p0c=Delta_p0c 
    # line_ds.insert_element('dz_acc_1', dz_acc, at_s=tab['s','cavity_1'])
    line_ds.build_tracker()

    match_ds_6d=line_ds.match(vary=xt.VaryList(['k_f0','k_d1','k_f1','k_d2','k_f2','k_d3'],
                                step=1e-8,
                                tag='quad'),
                targets=[xt.TargetSet(dx=0, at='end', tol=1e-9,tag='DS'),
                        # xt.TargetSet(betx=betx_fodo,bety=bety_fodo,dx=dx_fodo, at='quad_d_3', tol=1e-9, tag='FODO'),
                        xt.TargetSet(qx=2.60, qy=2.23, tol=1e-6, tag='tune')],
                method='6d',
                matrix_stability_tol=5e-3,
                # verbose=True
                )
    print('MATCHING QUAD DS 6D')
    match_ds_6d.target_status()
    match_ds_6d.vary_status()

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
    # print('RESULTS SXT DS MATCH 6D')
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

    return(line_cell,line_ds)#tw_ds_4d,tw_ds_6d) 

def plot_twiss(tw,line):
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)
    plt.plot(tw['s'], tw['betx'], label='betx')
    plt.plot(tw['s'], tw['bety'], label='bety')
    for i, el in enumerate(line.element_names):
        if 'quad' in el:
            plt.axvline(x=tw['s'][i], color='grey', linestyle='--',alpha=0.7)
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
            plt.axvline(x=tw['s'][i], color='grey', linestyle='--',alpha=0.7)
        if 'sxt' in el:
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

def plot_twiss_diff(tw_6d,tw_4d,line):
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)
    plt.plot(tw_6d['s'], tw_6d['betx']-tw_4d['betx'], label='betx')
    plt.plot(tw_6d['s'], tw_6d['bety']-tw_4d['bety'], label='bety')
    for i, el in enumerate(line.element_names):
        if 'quad' in el:
            plt.axvline(x=tw_6d['s'][i], color='grey', linestyle='--',alpha=0.7)
        elif 'sxt' in el:
            plt.axvline(x=tw_6d['s'][i], color='red', linestyle='--')
        elif 'cavity' in el:
            plt.axvline(x=tw_6d['s'][i], color='green', linestyle='--')
    plt.xlabel('s [m]')
    plt.ylabel('betx, bety [m]')
    plt.legend(loc='upper right')
    plt.subplot(2, 1, 2)
    plt.plot(tw_6d['s'], tw_6d['dx']-tw_4d['dx'], label='Dx')
    plt.plot(tw_6d['s'], tw_6d['dy']-tw_4d['dy'], label='Dy')
    for i, el in enumerate(line.element_names):
        if 'quad' in el:
            plt.axvline(x=tw_6d['s'][i], color='grey', linestyle='--',alpha=0.7)
        if 'sxt' in el:
            plt.axvline(x=tw_6d['s'][i], color='red', linestyle='--')
        elif 'cavity' in el:
            plt.axvline(x=tw_6d['s'][i], color='green', linestyle='--')
    plt.ylim(None)        
    plt.xlabel('s [m]')
    plt.ylabel('Dx, Dy [m]')
    plt.legend(loc='upper right')
    plt.show()

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
    #Call class_geo
    file_input='/mnt/c/muco/code/class_geometry/para_RCS_ME.txt'
    RCS = Geometry(file_input)
    t_frac=0
    n_k=0
    method='var_ref'

    line_cell,line_ds= make_optics(file_input,t_frac,n_k,method) 
    survey_cell=line_cell.survey(theta0=RCS.cell_angle/4)
    rec_cell=track(line_cell)
    # tw_4d=line_ds.twiss(method='4d')
    # tw_6d=line_ds.twiss(method='6d', matrix_stability_tol=5e-3)
    # plot_twiss_single(tw_4d)
    # plot_twiss_single(tw_6d)
    # plot_twiss_diff(tw_6d,tw_4d,line_ds)

    nb_arc=26
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
    # print(f'RCS Qx: {tw_6d["qx"]*nb_arc:.3f}')
    # print(f'Arc Qx: {tw_6d["qx"]:.3f}')
    # print(f'RCS Qy: {tw_6d["qy"]*nb_arc:.3f}')
    # print(f'Arc Qy: {tw_6d["qy"]:.3f}')
    # print(f'Arc Qs: {tw_6d["qs"]:.4f}')
    # print(f'RCS Qs: {tw_6d["qs"]*nb_arc:.3f}')
    # print(f'MCF: {tw_6d["momentum_compaction_factor"]:.5f}')

    # Plot trajectory
    plt.figure()
    tab=line_cell.get_table()
    plt.plot(survey_cell['s'], survey_cell['X']+rec_cell.x[0], label='Inj')
    for i, el in enumerate(line_cell.element_names):
        if 'en' in el:
            plt.axvline(x=tab['s',el], color='grey', linestyle='--')
        if 'ex' in el:
            plt.axvline(x=tab['s',el], color='grey', linestyle='--')
    plt.xlabel('s [m]')
    plt.ylabel('x [m]')
    plt.legend()
    plt.show()