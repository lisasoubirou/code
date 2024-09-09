# -*- coding: utf-8 -*-
"""
Created on Wed July  24  2024

@author: LS276867
"""

import numpy as np
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt
import xpart as xp
import sys
import xplt

sys.path.append('/mnt/c/muco')
from rcsparameters.geometry.geometry import Geometry, load_file
from rcsparameters.geometry.cell import Cell
# from ramping_module import ramping
# from track_function import track
import json

plt.rc('axes', labelsize=12)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

step_quad=1e-8


def fodo(RCS,time_frac, option, n_slice):
    time=time_frac
    t_ref=0.5
    energy=RCS.E_inj+(RCS.E_ext-RCS.E_inj)*time_frac

    #Get data from class_geo
    nb_arc=RCS.nb_arc
    nb_cell_rcs=RCS.nb_cell_rcs
    nb_cell_arc=RCS.nb_cell_arc
    pattern=RCS.pattern
    Ldd=RCS.dipole_spacing 
    Ls=RCS.LSSS #Length of short straight section
    Lins_cell=RCS.insertion_length #+RCS.L_extra_arc
    if RCS.bloc == 2:
        Lins_cell= Lins_cell+Ldd
    list_hn=RCS.hn(time) #List of bendings
    list_hn_ref=RCS.hn(t_ref) #List of reference bendings
    theta=RCS.theta(time) #List of bending angle 
    theta_ref=RCS.theta(t_ref)
    epsilon=RCS.epsilon(time)
    eps_ref=RCS.epsilon(t_ref)
    L_dip_path=RCS.L_dip_path(time)
    L_dip_path_ref=RCS.L_dip_path(t_ref)
    L_dd_path=RCS.L_dd_path(time)
    L_dd_path_ref=RCS.L_dd_path(t_ref)
    L_qd_path=RCS.L_qp_dip_path(time)
    # L_drift_ext=(RCS.L_extra_arc-2*Ls)/(4*nb_cell_arc*(1+1/nb_cell_arc))
    t_cell= (RCS.insertion_length+RCS.L_extra_arc*(1+1/RCS.nb_cell_arc)-RCS.cell_length)/((1+1/RCS.nb_cell_arc)*(RCS.L_extra_arc))
    L_quad_cell=t_cell*RCS.L_extra_arc/nb_cell_arc/4
    if (1-t_cell)*RCS.L_extra_arc-2*L_quad_cell-2*RCS.LSSS < 0:
        print("Distributing extra length is not enough to egalize cell length and RF length")
        print("We adjust at best")
        t_cell=(1-2*Ls/RCS.L_extra_arc)/(1+1/2/RCS.nb_cell_arc)
    L_quad_cell=t_cell*RCS.L_extra_arc/nb_cell_arc/4

    Lins_cell=Lins_cell+(1-t_cell)*RCS.L_extra_arc-2*L_quad_cell-2*RCS.LSSS
    #RF data
    N_turn_rcs=55
    n_turns = N_turn_rcs*nb_cell_arc
    energy_increment_per_arc = (RCS.E_ext-RCS.E_inj)/n_turns
    sync_phase=45
    phase=180-sync_phase
    volt_cell=energy_increment_per_arc/np.sin(phase*np.pi/180)

    #Define elements: drifts, quads, sextupoles
    mu=np.pi/2 #Phase advance 
    f=(RCS.cell_length+(t_cell-1)*RCS.L_extra_arc/RCS.nb_cell_arc)*1/(4*np.sin(mu/2)) #Focusing strength of quadrupole
    drift_ext=xt.Drift(length=L_quad_cell)
    drift_ins=xt.Drift(length=Lins_cell/2)

    quad_f=xt.Quadrupole(k1=1/f/Ls, length=Ls) 
    quad_d=xt.Quadrupole(k1=-1/f/Ls, length=Ls) 
    f_ins=(RCS.insertion_length+(1-t_cell)*RCS.L_extra_arc)/(4*np.sin(mu/2)) #Focusing strength of quadrupole 
    # quad_f_ins=xt.Quadrupole(k1=1/f_ins/Ls, length=Ls)
    quad_d_half=xt.Quadrupole(k1=-1/f/Ls, length=Ls/2)
    step_quad=1e-8

    sxt_d=xt.Sextupole(k2=0)
    sxt_f=xt.Sextupole(k2=0)

    step_quad=1e-8
    step_sxt=1e-5

    #Cavity settings
    cavity=xt.Cavity(voltage=volt_cell/2, frequency=1.3e9, lag=phase)
    RF_acc=xt.ReferenceEnergyIncrease(Delta_p0c=energy_increment_per_arc/2)
    dz_acc=xt.ZetaShift(dzeta=0)

    beg = xt.Marker()
    end = xt.Marker()

    model_edge='full'
    hcell_el=[]
    hcell_name=[]
    for id, item in enumerate(pattern):
        if id == 0 or RCS.type=='normal' or item != prev_item :
            hcell_el.append(xt.Drift(length=L_dd_path[id]))
            hcell_name.append('drift_dd')   
            hcell_el.append(xt.DipoleEdge(k=list_hn[id], e1=-eps_ref[id], side='entry',model = model_edge))
            hcell_name.append('en'+f'{id}')
        if option == 'var_ref':
            hcell_el.append(xt.Bend(k0=list_hn[id], h=list_hn[id],length=L_dip_path[id]))
        elif option == 'var_k':
            hcell_el.append(xt.Bend(k0=list_hn[id], h=list_hn_ref[id], length=L_dip_path_ref[id]))
        else:
            raise ValueError("Invalid option: {}".format(option))
        if item == 'BSC':
            hcell_name.append('BSC')
        elif item == 'BNC':
            hcell_name.append('BNC')
        if id == len(pattern)-1 or RCS.type=='normal' or item != pattern[id+1] :
            hcell_el.append(xt.DipoleEdge(k=list_hn[id], e1=eps_ref[id+1], side='exit',model = model_edge))
            hcell_name.append('ex'+f'{id}')
        prev_item=item
    hcell_el.append(xt.Drift(length=L_qd_path))
    hcell_name.append('drift_dd')

    FODO_el = [drift_ext, quad_f, sxt_f, drift_ext]+hcell_el+[drift_ext, quad_d, sxt_d, drift_ext]+hcell_el
    FODO_names = ['drift_ext','quad_f','sxt_f','drift_ext']+hcell_name+['drift_ext','quad_d','sxt_d','drift_ext']+hcell_name 
    FODO_arc=[beg, quad_d_half, RF_acc, cavity, drift_ins] + FODO_el*int(nb_cell_arc) + [drift_ext, quad_f,drift_ext, drift_ins, cavity, RF_acc, quad_d_half, end]
    FODO_arc_names=['marker_beg','quad_d_half', 'RF_acc', 'cavity', 'drift_ins'] + FODO_names*int(nb_cell_arc) + ['drift_ext','quad_f','drift_ext', 'drift_ins','cavity','RF_acc','quad_d_half', 'marker_end']

    # line_cell=xt.Line(elements=FODO_el*int(nb_cell_arc), element_names=FODO_names*int(nb_cell_arc))
    # line_cell.particle_ref = xp.Particles(p0c=energy, q0=1, mass0=xp.MUON_MASS_EV)
    # tw_cell=line_cell.twiss(method='4d')

    # line_FODO=xt.Line(elements=FODO_arc, element_names=FODO_arc_names)
    # line_FODO.particle_ref = xp.Particles(p0c=energy, q0=1, mass0=xp.MUON_MASS_EV)
    # if n_slice > 0:
    #     line_FODO.slice_thick_elements(slicing_strategies=[xt.Strategy(slicing=xt.Teapot(n_slice))])
    
    # line_FODO._init_var_management()
    # line_FODO.vars['kf'] = line_FODO['quad_f'].k1
    # line_FODO.vars['kd'] = line_FODO['quad_d'].k1
    # line_FODO.vars['kf1'] = line_FODO['quad_f'].k1
    # line_FODO.vars['kd1'] = line_FODO['quad_d'].k1
    # line_FODO.vars['kf2'] = line_FODO['quad_f'].k1
    # line_FODO.vars['kd2'] = line_FODO['quad_d'].k1
    # line_FODO.vars['kd3'] = line_FODO['quad_d'].k1
    # line_FODO.vars['kf3'] = line_FODO['quad_f'].k1


    # line_FODO.element_refs['quad_f'].k1 = line_FODO.vars['kf']
    # line_FODO.element_refs['quad_f_6'].k1 = line_FODO.vars['kf']

    # line_FODO.element_refs['quad_f_1'].k1 = line_FODO.vars['kf1']
    # line_FODO.element_refs['quad_f_5'].k1 = line_FODO.vars['kf1']

    # line_FODO.element_refs['quad_f_2'].k1 = line_FODO.vars['kf2']
    # line_FODO.element_refs['quad_f_4'].k1 = line_FODO.vars['kf2']
    # line_FODO.element_refs['quad_f_3'].k1 = line_FODO.vars['kf3']

    
    # line_FODO.element_refs['quad_d_half'].k1 = line_FODO.vars['kd']
    # line_FODO.element_refs['quad_d_half_1'].k1 = line_FODO.vars['kd']

    # line_FODO.element_refs['quad_d'].k1 = line_FODO.vars['kd1']
    # line_FODO.element_refs['quad_d_5'].k1 = line_FODO.vars['kd1']

    # line_FODO.element_refs['quad_d_1'].k1 = line_FODO.vars['kd2']
    # line_FODO.element_refs['quad_d_4'].k1 = line_FODO.vars['kd2']

    # line_FODO.element_refs['quad_d_2'].k1 = line_FODO.vars['kd3']
    # line_FODO.element_refs['quad_d_3'].k1 = line_FODO.vars['kd3']

    # # print(tw_cell['dx','quad_f'],tw['dx','quad_d'])

    # tab=line_FODO.get_table()
    # for quad in tab.rows[tab.element_type == 'Quadrupole'].name:
    #     if 'quad_f' in quad:
    #         line_FODO.element_refs[quad].k1 = line_FODO.vars['kf']
    #     if 'quad_d' in quad:
    #         line_FODO.element_refs[quad].k1 = line_FODO.vars['kd']
    # match_ds_4d=line_FODO.match(vary=xt.VaryList(['kf','kd'],  #'kf1','kd1','kf2','kd2','kd3'
    #                             step=step_quad,
    #                             tag='quad'),
    #             targets=[
    #                     # xt.TargetSet(
    #                     #             betx=tw_cell['betx','quad_f'],
    #                     #             bety=tw_cell['bety','quad_f'], 
    #                     #             dx=tw_cell['dx','quad_f'], at='quad_f_3', tol=1e-9, tag='FODO'),
    #                     xt.Target(tar='dx', at='marker_end', value=0, tol=1e-6, tag='fodo'),
    #                     # xt.TargetSet(qx=2.6067, qy=2.252, tol=1e-3, tag='tune')
    #                     ],
    #             solve=False,
    #             method='4d',
    #             # verbose=True,
    #             matrix_stability_tol=0.11
    #             )
    # # print('MATCHING QUAD DS 4D')
    # match_ds_4d.step(20)
    # match_ds_4d.target_status()
    # match_ds_4d.vary_status()
    # # # tw_ds_4d=line_ds.twiss( method='4d')

    # line_tot=xt.Line(
    # elements=FODO_arc*int(rcs_opt.nb_arc),
    # element_names=FODO_arc_names*int(rcs_opt.nb_arc))
    # line_tot.particle_ref = xp.Particles(p0c=energy, q0=1, mass0=xp.MUON_MASS_EV)
    # return (line_cell,line_FODO,line_tot)
    return(FODO_el, FODO_names, FODO_arc, FODO_arc_names)

def plot_layout(line):
    sv = line.survey()
    plot = xplt.FloorPlot(
        sv,
        line,
        projection="ZX",
        boxes={  # adjust box style for element names matching regex
            "BSC": dict(color="maroon",width=5),
            "BNC": dict(color="blue",width=5),
            "quad_d" : dict(color="green",width=10),
            "quad_f" : dict(color="green",width=10),
            "sxt_d" : dict(color="purple",width=15),
            "sxt_f" : dict(color="purple",width=15),
        }
)
    plot.legend(loc="lower left")
    # plot.ax.set_xlim(0, 200)


def plot_twiss(tw):
    fig1 = plt.figure(1, figsize=(6.4, 4.8*1.5))
    bet = plt.subplot(2,1,1)
    disp = plt.subplot(2,1,2, sharex=bet)

    bet.plot(tw.s, tw.betx, label=r'$\beta_x$')
    bet.plot(tw.s, tw.bety,label=r'$\beta_y$')
    bet.set_ylabel(r'$\beta_{x,y}$ [m]')
    # bet.axvspan(tw['s','marker_beg'], tw['s','quad_f'],
    #                 color='b', alpha=0.1, linewidth=0)
    # bet.axvspan(tw['s','drift_ins_1'], tw['s','marker_end'],
    #                 color='b', alpha=0.1, linewidth=0)
    bet.legend()

    # spbet.legend()
    disp.plot(tw.s, tw.dx,label=r'$D_x$')
    disp.plot(tw.s, tw.dy,label=r'$D_y$')
    disp.set_ylabel(r'$D_{x,y}$ [m]')
    disp.legend()
    # disp.axvspan(tw['s','marker_beg'], tw['s','quad_f'],
    #                 color='b', alpha=0.1, linewidth=0)
    # disp.axvspan(tw['s','drift_ins_1'], tw['s','marker_end'],
    #                 color='b', alpha=0.1, linewidth=0)
    fig1.subplots_adjust(left=.15, right=.92, hspace=.27)
    plt.show()

def apert_1b_optics(tw, rcs):
    beta_max = np.max(tw.betx)
    disp_max = np.max(tw.dx)
    beam_size = rcs.sigma_delta*(disp_max) + rcs.n_sigma*np.sqrt(rcs.emittance_inj*beta_max)
    mask_nc = [item == 'BNC' for item in rcs.pattern]
    mask_sc = [item == 'BSC' for item in rcs.pattern]
    max_width_dip = rcs.extrema(rcs.t_ext)[1]-rcs.extrema(rcs.t_inj)[0]
    width_sc = np.max(max_width_dip[mask_sc])
    width_nc = np.max(max_width_dip[mask_nc])

    aperture_sc= width_sc + (beam_size + rcs.dx_qp)*2
    aperture_nc= width_nc + (beam_size + rcs.dx_qp)*2
    return (aperture_nc, aperture_sc)

def apert_2b_optics(tw, rcs):
    beta_max = np.max(tw.betx)
    disp_max = np.max(tw.dx)
    beam_size = rcs.sigma_delta*(disp_max) + rcs.n_sigma*np.sqrt(rcs.emittance_inj*beta_max)
    nd=len(rcs.pattern)
    n_bloc=len(rcs.pattern)//2
    y_rot_inj=rcs.rot_coord(rcs,n_bloc,0)[1]
    y_rot_ext=rcs.rot_coord(rcs,n_bloc,1)[1]
    y_max=np.array([np.max(y_rot_ext[id*11:(id+1)*11]) for id in range(nd//2)])
    y_min=np.array([np.min(y_rot_inj[id*11:(id+1)*11]) for id in range(nd//2)])
    max_width_dip=y_max-y_min
    mask_nc=[item == 'BNC' for item in rcs.pattern[0:n_bloc]]
    mask_sc=[item == 'BSC' for item in rcs.pattern[0:n_bloc]]
    width_sc=np.max(max_width_dip[mask_sc])
    width_nc=np.max(max_width_dip[mask_nc])
    aperture_sc= width_sc + (beam_size + rcs.dx_qp)*2
    aperture_nc= width_nc + (beam_size + rcs.dx_qp)*2
    return (aperture_nc, aperture_sc)

def size_beam_diam(tw, rcs):
    beta_max = np.max(tw.betx)
    disp_max = np.max(tw.dx)
    beam_size = 2*(rcs.sigma_delta*(disp_max) + rcs.n_sigma*np.sqrt(rcs.emittance_inj*beta_max))
    return beam_size

file_input='/mnt/c/muco/code/class_geometry/parameter_files/para_RCS_FE.txt'
kw_geo=load_file(file_input)
# kw_geo['pattern']=['BSC','BNC','BSC','BSC','BNC','BSC'] #,'BNC','BSC']
print(kw_geo)
# kw_geo['bloc']=2
# kw_geo['nb_arc']=3
default_para_RCS1={
        'n_sigma':6, #Nb of sigma of beam size that the QP aperture must fit
        'sigma_delta':5e-3, #Max dispersion in energy of the beam
        'emitt_n':25e-6, #Normalized emittance
        'n_cav':3000, #Total number of cavity : 700, 380, 540, 3000 
        'l_cav':1.2474, #Length of 1 cavity
        'mu':np.pi/2, #Phase advance per cell
        'apert_min':30e-3, #Min aperture size (diam) to be taken for QP
        'dx':10e-3, #Extra radius length to be taken to fit vacuum chambers
        'Bpole':1, #Max field on QP pole
    }
rcs = Cell(file_input_geo=kw_geo,file_input_opt=default_para_RCS1)
rcs.nb_cell_arc=9
time=0
n_slice=0
option='var_ref'

FODO_el, FODO_names, arc_el,arc_names=fodo(RCS=rcs, time_frac=time, option=option, n_slice=n_slice)
line_FODO=xt.Line(elements=FODO_el, element_names=FODO_names)
line_FODO.particle_ref = xp.Particles(p0c=rcs.E_inj+(rcs.E_ext-rcs.E_inj)*time, q0=1, mass0=xp.MUON_MASS_EV)
tw_FODO=line_FODO.twiss(method='4d',matrix_stability_tol=0.11)
tw_init_FODO = tw_FODO.get_twiss_init(0)
plot_twiss(tw_FODO)

line=xt.Line(elements=arc_el[len(arc_el)//2:], element_names=arc_names[len(arc_el)//2:])
line.particle_ref = xp.Particles(p0c=rcs.E_inj+(rcs.E_ext-rcs.E_inj)*time, q0=1, mass0=xp.MUON_MASS_EV)


match_tr=line.match(
            start=0,
            end='_end_point', 
            init=tw_init_FODO,
            vary=xt.VaryList(['k_f0','k_d1','k_f1','k_d2','k_f2','k_d3'],
                            step=step_quad,
                            tag='quad'),
            targets=[xt.TargetSet(dx=0,dpx=0,alfx=0,alfy=0, at='end',tol=1e-9)],
            # verbose=True
            )
print('RESULTS FROM TR MATCHING')
match_tr.target_status()
match_tr.vary_status()

# line_cell, line,line_tot=fodo(RCS=rcs, time_frac=time, option=option, n_slice=n_slice)
# tw=line.twiss(method='6d',matrix_stability_tol=0.11)
# print('GLOBAL PARAMETER')
# print('MCF', round(tw['momentum_compaction_factor'], 6))
# print('Qx', round(tw['qx'], 5))
# print('Qy', round(tw['qy'], 5))
# print('dqx', round(tw['dqx'], 2))
# print('dqy', round(tw['dqy'], 2))
# if rcs_opt.type == 'hybrid':
#     if rcs_opt.bloc == 1:
#         aperture_nc, aperture_sc = apert_1b_optics(tw, rcs_opt)
#     elif rcs_opt.bloc == 2:
#         aperture_nc, aperture_sc = apert_2b_optics(tw, rcs_opt)
#     print('Apert NC', aperture_nc)
#     print('Apert SC', aperture_sc)
#     print('Length SC', rcs_opt.dipole_families['BSC']['length'])
# print('Length NC', rcs_opt.dipole_families['BNC']['length'])
# print('Size beam parameter', size_beam_diam(tw, rcs_opt), 'm')

# plot_twiss(line_cell.twiss(method='4d',matrix_stability_tol=0.11))
# plot_twiss(tw)

line.build_tracker()
plot_layout(line)

# line_tot.build_tracker()
# survey=line_tot.survey()
# print(survey)
# plt.figure()
# plt.plot(survey['Z'], survey['X'])
# plt.axis('equal')
# # for i, el in enumerate(line.element_names):
# #     if 'dip' in el:
# #         plt.axvline(x=survey['Z'][i], color='grey', linestyle='--')
# #     elif 'quad' in el:
# #         plt.axvline(x=survey['Z'][i], color='red', linestyle='--')
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.show()