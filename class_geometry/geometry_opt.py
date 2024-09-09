import numpy as np
import matplotlib.pyplot as plt
import json

from rcsparameters.geometry.geometry import Geometry
from sympy import solve, Poly, Symbol
import matplotlib
from matplotlib.colors import Normalize
from scipy import optimize
import pandas as pd

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize

file_input='/mnt/c/muco/code/class_geometry/parameter_files/para_RCS_ME.txt'
E_muon=105.66e6
emitt_n=25e-6
sigma_delta=10e-3
n_sigma=2*6
dbeta=0.20
sigma_inc=1.1
tol_dspur_lhc=1.3e-2
dx=10e-3
apert_min=30e-3
RCS = Geometry(file_input,dipole_spacing=0.4)
mu=np.pi/2
G_max=40
B_pole=1
l_cav=1.2474

def rot_coord(RCS,n_bloc,x):
    theta=RCS.cell_angle/8
    coor_init=RCS.zn(x)[1:(1+n_bloc*11)]
    coor=coor_init-coor_init[0]
    x,y=np.real(coor),np.imag(coor)
    x_rot=x*np.cos(theta)+y*np.sin(theta)
    y_rot=-x*np.sin(theta)+y*np.cos(theta)
    return(x_rot, y_rot)

def solve_nc_opt(RCS, G_max, mu):
    L_arc=RCS.arc_length
    L_nc=RCS.dipole_families['BNC']['tot_length']/RCS.nb_arc
    L_sc=RCS.dipole_families['BSC']['tot_length']/RCS.nb_arc
    L_dd=RCS.dipole_spacing
    nd=len(RCS.pattern)
    Brho_ext=RCS.ext_Brho

    delta=(L_dd*(nd+1))**2+ 8*Brho_ext*np.sin(mu/2)*(L_arc-L_nc-L_sc)/G_max/L_arc
    nc_max=int((-(nd+1)*L_dd + np.sqrt(delta))/(8*Brho_ext*np.sin(mu/2)/G_max/L_arc))
    L_qp_min=4*Brho_ext*np.sin(mu/2)/G_max/L_arc*nc_max
    RCS_opt = Geometry(file_input,dipole_spacing=RCS.dipole_spacing, nb_cell_arc=nc_max, LSSS=L_qp_min, pattern=RCS.pattern)

    print('Length of 1 quad',np.round(L_qp_min,2), '[m]')
    print('Number max of n_c, analytical', nc_max,'class', RCS_opt.nb_cell_arc)
    print('Cell length',np.round(RCS_opt.cell_length,2),'[m]')
    return (nc_max, L_qp_min, RCS_opt)

def solve_nc_opt_phi(RCS,mu):
    emitt=emitt_n/RCS.inj_gamma
    L_dd=RCS.dipole_spacing
    L_arc=RCS.arc_length
    L_nc=RCS.dipole_families['BNC']['tot_length']/RCS.nb_arc
    L_sc=RCS.dipole_families['BSC']['tot_length']/RCS.nb_arc
    nd=len(RCS.pattern)
    A=n_sigma*np.sqrt(RCS.arc_length*(1+np.sin(mu/2))/np.sin(mu)*emitt)
    B=sigma_delta*RCS.arc_length*2*np.pi*(1+np.sin(mu/2)/2)/(4*(np.sin(mu/2))**2)
    C=4*RCS.ext_Brho*np.sin(mu/2)/RCS.arc_length/B_pole
    D=(nd+1)*L_dd
    E=(L_arc-L_nc-L_sc)/2
    function= lambda x: -(C*((A/x**0.5) + B/x**2 + dx)*x**2 + D*x - E)
    # x = Symbol('x')
    # solve(Poly(C*((A/x**0.5) + B/x**2 + dx)*x**2 + D*x - E), x)
    result=optimize.minimize_scalar(function, bounds=(0,10))
    return result

def compute_beam_size(RCS, mu):
    emitt=emitt_n/RCS.inj_gamma
    beta_max=RCS.cell_length*(1+np.sin(mu/2))/np.sin(mu)
    D_max=RCS.cell_length*RCS.cell_angle*(1+np.sin(mu/2)/2)/(4*(np.sin(mu/2))**2)
    # D_spur=D_max*(sigma_inc-1) + sigma_inc*D_max*np.sqrt(1+dbeta)*tol_dspur_lhc*np.sqrt(beta_max)
    D_spur=0
    dbeta=0
    print('Twiss function, beta, D, D_spur', beta_max, D_max, D_spur)
    beam_size_rms=n_sigma*np.sqrt(emitt*beta_max+(D_max*sigma_delta)**2) 
    beam_size_mx=sigma_delta*(D_max+D_spur) + n_sigma*np.sqrt(emitt*beta_max*(1+dbeta))
    print(f'Beam size {n_sigma} sigma RMS', np.round(beam_size_rms,4), '[m]')
    print(f'Beam size {n_sigma} sigma MADX', beam_size_mx)
    beam_size=beam_size_mx
    return (beam_size)

def compute_apert_1bloc(RCS,dx):
    beam_size=compute_beam_size(RCS, mu)
    mask_nc=[item == 'BNC' for item in RCS.pattern]
    mask_sc=[item == 'BSC' for item in RCS.pattern]
    max_width_dip=RCS.extrema(RCS.t_ext)[1]-RCS.extrema(RCS.t_inj)[0]
    width_sc=np.max(max_width_dip[mask_sc])
    width_nc=np.max(max_width_dip[mask_nc])

    aperture_sc=width_sc+beam_size+dx
    aperture_nc=width_nc+beam_size+dx
    print('1bloc, Excursion SC', np.round(width_sc,3), 'Aperture SC', np.round(aperture_sc,3), '[m]')
    print('1bloc, Excursion NC', np.round(width_nc,3), 'Aperture NC', np.round(aperture_nc,3), '[m]')
    # RCS.plot_traj()
    return (width_nc,width_sc,aperture_nc,aperture_sc)

def compute_apert_2bloc(RCS,dx):
    beam_size=compute_beam_size(RCS, mu)
    nd=len(RCS.pattern)
    n_bloc=len(RCS.pattern)//2
    x,y_rot_inj=rot_coord(RCS,n_bloc,0)
    x,y_rot_ext=rot_coord(RCS,n_bloc,1)
    y_max=np.array([np.max(y_rot_ext[id*11:(id+1)*11]) for id in range(nd//2)])
    y_min=np.array([np.min(y_rot_inj[id*11:(id+1)*11]) for id in range(nd//2)])
    max_width_dip=y_max-y_min
    mask_nc=[item == 'BNC' for item in RCS.pattern[0:n_bloc]]
    mask_sc=[item == 'BSC' for item in RCS.pattern[0:n_bloc]]
    width_sc=np.max(max_width_dip[mask_sc])
    width_nc=np.max(max_width_dip[mask_nc])
    aperture_sc=width_sc+beam_size+dx
    aperture_nc=width_nc+beam_size+dx
    print('2bloc, Excursion SC', np.round(width_sc,3), 'Aperture SC', np.round(aperture_sc,3), '[m]')
    print('2bloc, Excursion NC', np.round(width_nc,3), 'Aperture NC', np.round(aperture_nc,3), '[m]')
    # RCS.plot_traj()
    plt.figure()
    plt.title('Trajectory in 1 bloc')
    for i in np.linspace(0.,1.,8):
        x_rot, y_rot = rot_coord(RCS,n_bloc,i)
        plt.plot(x_rot, y_rot)
        for z1, z2 in zip(RCS.z_begin(1)[0:n_bloc], RCS.z_end(1)[0:n_bloc]):
            x1, x2 = np.real(z1), np.real(z2)
            plt.axvline(x1,ls="--",color="k")
            plt.axvline(x2,ls="--",color="k")
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
    plt.show()
    return (width_nc,width_sc,aperture_nc,aperture_sc)

# #Test for different pattern: optimisation of the cell, get beam size and excursion 
# list_pattern=[['BSC','BNC','BSC'],['BSC','BNC','BSC','BNC','BSC'],
#               ['BSC','BNC','BSC','BNC','BSC','BNC','BSC'], # 1bloc
#               ['BSC','BNC','BSC','BSC','BNC','BSC'],
#               ['BSC','BNC','BSC','BNC','BSC','BSC','BNC','BSC','BNC','BSC']] #2 bloc
# name_pattern=['3dip 1bloc','5dip 1bloc','7dip 1bloc','3dip 2bloc','5dip 2bloc']
# results=np.zeros((len(list_pattern),8))
# for i_pat, pattern in enumerate(list_pattern):
#     RCS= Geometry(file_input, dipole_spacing=0.4, pattern=pattern)
#     print('Pattern', pattern)
#     nc_max, L_qp_min, RCS_opt = solve_nc_opt(RCS, G_max, mu)
#     beam_size=compute_beam_size(RCS_opt,mu)
#     if "1bloc" in name_pattern[i_pat]:
#         width_nc,width_sc,aperture_nc,aperture_sc=compute_apert_1bloc(RCS_opt,beam_size,dx)
#     elif "2bloc" in name_pattern[i_pat]:
#         width_nc,width_sc,aperture_nc,aperture_sc=compute_apert_2bloc(RCS_opt,beam_size,dx)
#     results[i_pat,0]=beam_size
#     results[i_pat,1]=width_sc
#     results[i_pat,2]=width_nc
#     results[i_pat,3]=aperture_sc
#     results[i_pat,4]=aperture_nc
#     results[i_pat,5]=RCS_opt.dipole_families['BSC']['length']
#     results[i_pat,6]=RCS_opt.dipole_families['BNC']['length']
#     results[i_pat,7]=nc_max

# num=np.arange(0,5,1)
# plt.figure()
# plt.scatter(num,results[:,7])
# plt.xticks(ticks=num,labels=name_pattern)
# plt.xlabel('Pattern')
# plt.ylabel('$n_c$ max')
# plt.show()

# plt.figure()
# plt.scatter(num,results[:,0]*1e3, label='beam size',color='tab:green', marker='*',s=100)
# plt.xticks(ticks=num,labels=name_pattern)
# plt.xlabel('Pattern')
# plt.ylabel('Beam size [mm]')
# plt.legend(loc='upper left')
# plt.ylim(ymin=0)
# plt.show()

# plt.figure()
# plt.scatter(num,results[:,1]*1e3, label='width sc', color='tab:blue')
# # plt.scatter(num,results[:,2]*1e3, label='width nc',color='tab:orange')
# plt.scatter(num,results[:,3]*1e3, label='aperture sc', marker='s',color='blue')
# # plt.scatter(num,results[:,4]*1e3, label='aperture nc', marker='s',color='tab:orange')
# # plt.scatter(num,results[:,0]*1e3, label='beam size',color='tab:green', marker='*')
# plt.xticks(ticks=num,labels=name_pattern)
# plt.xlabel('Pattern')
# plt.ylabel('Width SC [mm]')
# plt.legend() #loc='upper left'
# plt.ylim(ymin=0)
# plt.show()

# plt.figure()
# # plt.scatter(num,results[:,1]*1e3, label='width sc', color='tab:blue')
# plt.scatter(num,results[:,2]*1e3, label='width nc',color='tab:orange')
# # plt.scatter(num,results[:,3]*1e3, label='aperture sc', marker='s',color='tab:blue')
# plt.scatter(num,results[:,4]*1e3, label='aperture nc', marker='s',color='red')
# # plt.scatter(num,results[:,0]*1e3, label='beam size',color='tab:green', marker='*')
# plt.xticks(ticks=num,labels=name_pattern)
# plt.xlabel('Pattern')
# plt.ylabel('Width NC [mm]')
# plt.legend()
# plt.ylim(ymin=0)
# plt.show()

# plt.figure()
# plt.scatter(num,results[:,5], label='length sc', color='tab:blue')
# plt.scatter(num,results[:,6], label='length nc', color='tab:orange')
# plt.xticks(ticks=num,labels=name_pattern)
# plt.xlabel('Pattern')
# plt.ylabel('Lengh dipole [m]')
# plt.legend()
# plt.ylim(ymin=0)
# plt.show()

#Minimum QP length without taking into account the aperture
# RCS = Geometry(file_input, dipole_spacing=0.4)
# L_cell_min=10
# L_cell_max=RCS.arc_length
# L_cell = np.linspace(L_cell_min, L_cell_max, 30)
# L_qp_min = 4 * RCS.ext_Brho * np.sin(mu / 2) / G_max / L_cell
# plt.figure()
# plt.plot(L_cell, L_qp_min,color='k')
# plt.xlabel('Cell length [m]')
# plt.ylabel('Minimum QP length [m]')
# plt.show()

RCS_name=['ME', 'HE','LHC']
n_cav=[380,540,3000]
# RCS_name=['CERN_RCS1', 'CERN_RCS2','CERN_RCS3']
# n_cav=[21.4e9/30e6/l_cav,83.6e9/30e6/l_cav,45.1e9/30e6/l_cav]
rcs=1
file_input='/mnt/c/muco/code/class_geometry/parameter_files/para_RCS_'+ RCS_name[rcs]+'.txt'
n_cav=n_cav[rcs] 
L_cell_min=10
L_cell_max=RCS.arc_length
L_cell = np.linspace(L_cell_min, 300, 30)
#Minimum QP length taking into account the aperture
def solve_nc_opt_apert(file_input, n_cav, mu, L_cell, Bpole=1):
    RCS = Geometry(file_input, dipole_spacing=0.4)
    L_cav_tot=n_cav*l_cav
    f_arc=1-L_cav_tot/RCS.C
    RCS = Geometry(file_input, dipole_spacing=0.4, filling_ratio=f_arc)
    lquad, size_beam=Lmin_quad_apert(L_cell, Bp=RCS.ext_Brho, Larc=RCS.tot_arc_length, E_ext=RCS.E_ext, mu=mu)
    lquad_min=min(lquad)
    lcell_min=L_cell[np.argmin(lquad)]
    nc_g=np.floor(RCS.arc_length/L_cell[np.argmin(lquad)])+1
    RCS_inter=Geometry(file_input, dipole_spacing=0.4, filling_ratio=f_arc, LSSS=lquad_min, nb_cell_arc=nc_g)
    l_quad_opt=lquad[np.argmin(np.absolute(L_cell-RCS_inter.cell_length))]
    RCS_opt=Geometry(file_input, dipole_spacing=0.4, filling_ratio=f_arc, LSSS=l_quad_opt, nb_cell_arc=nc_g)
    return (lquad, size_beam, RCS_opt, RCS_opt.nb_cell_arc, l_quad_opt)

def Lmin_quad_apert(Lc, Bp, Larc, E_ext, mu=np.pi/2, Bpole=1, eps_n=25e-6, n=6, delta=5e-3, dx=10e-3):
    gamma=E_ext/E_muon
    eps=eps_n/gamma
    beta=Lc*(1+np.sin(mu/2))/np.sin(mu)
    disp=2*np.pi*Lc**2/Larc*(1+1/2*np.sin(mu/2))/(4*(np.sin(mu/2))**2)
    size=n*np.sqrt(beta*eps)+ delta*disp + dx
    size[size < apert_min/2] = apert_min/2
    phi=size
    return(4*np.sin(mu/2)/Lc*Bp/Bpole*phi,phi)

# n_cav=3000
lquad, size_beam, RCS_opt, nc_opt, l_quad_opt=solve_nc_opt_apert(file_input, n_cav, mu, L_cell)
lquad_max=(RCS.arc_length-RCS.dipole_length_tot/RCS.nb_arc-2*(RCS.nd+1)*RCS.dipole_spacing*np.floor(RCS.arc_length/L_cell))/2/np.floor(RCS.arc_length/L_cell)
mask=lquad_max-lquad>0
lquad, size_beam=lquad[mask], size_beam[mask]
L_cell=L_cell[mask]
f_arc=RCS_opt.filling_ratio
nb_cell=np.floor(RCS_opt.arc_length/L_cell).astype(int)
cell_length_test=np.zeros(len(lquad))
nc_max=np.zeros(len(lquad))
width_dip=np.zeros([len(lquad),4])
L_cell_plot=L_cell
for ic, (lqp, nb_c) in enumerate(zip(lquad,nb_cell)):
    RCS=Geometry(file_input, dipole_spacing=0.4, filling_ratio=f_arc, LSSS=lqp, nb_cell_arc=nb_c)
    nc_max[ic]=int(RCS.nb_cell_arc)
    cell_length_test[ic]=float(RCS.cell_length)
    # width_dip[ic, 0],width_dip[ic, 1],width_dip[ic, 2],width_dip[ic, 3]=compute_apert_1bloc(RCS,dx)

MCF=(RCS_opt.cell_angle/2)**2*(1/(np.sin(mu/2))**2 - 1/12)*RCS_opt.filling_ratio

plt.figure()
plt.plot(L_cell,lquad,color='k')
plt.scatter(L_cell[np.argmin(lquad)],min(lquad), color='tab:orange', label='Minimum Lqp')
plt.scatter(RCS_opt.cell_length, l_quad_opt, color='tab:blue', label='Optimum Lqp')
plt.xlabel('Cell length [m]')
plt.ylabel('Minimum QP length [m]')
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.legend()
plt.show()

plt.figure()
plt.plot(L_cell, 2*size_beam*1e3,color='k')
plt.scatter(L_cell[np.argmin(lquad)],2*1e3*size_beam[np.argmin(lquad)], color='tab:orange',label='Minimum Lqp')
plt.scatter(RCS_opt.cell_length,2*1e3*size_beam[np.argmin(np.absolute(L_cell-RCS_opt.cell_length))],color='tab:blue', label='Optimum Lqp' )
plt.xlabel('Cell length [m]')
plt.ylabel('Aperture QP [mm]')
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.legend()
plt.show()

plt.figure()
plt.scatter(L_cell_plot,nc_max)
plt.xlabel('Cell length [m]')
plt.ylabel('Number of cell per arc')
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.show()

# plt.figure()
# plt.scatter(pd.unique(cell_length_test),pd.unique(width_dip[:,2])*1e2)
# y_leg=plt.ylim()[1]
# # for lc, nc in zip(pd.unique(cell_length_test), pd.unique(nc_max)):
#     # plt.axvline(x=lc, linestyle='dashed',alpha=0.8, color='grey')
#     # plt.text(lc,y_leg*1.02,f'{int(RCS_opt.arc_length/lc)}', ha='center', va='center', fontsize=11, color='black',fontdict={'fontname':'DejaVu Sans'})
# plt.text(15,y_leg*1.02,'$n_c$ = ', ha='center', va='center', fontsize=11, color='black',fontdict={'fontname':'DejaVu Sans'})
# plt.xlabel('Cell length [m]')
# plt.ylabel('Aperture NC dipole [cm]')
# plt.ylim(ymin=0)
# plt.xlim(xmin=0)
# plt.show()

# plt.figure()
# plt.scatter(pd.unique(cell_length_test)[1:],pd.unique(width_dip[:,3])[1:]*1e2)
# y_leg=plt.ylim()[1]
# # for lc, nc in zip(pd.unique(cell_length_test), pd.unique(nc_max)):
#     # plt.axvline(x=lc, linestyle='dashed',alpha=0.8, color='grey')
#     # plt.text(lc,y_leg*1.02,f'{int(RCS_opt.arc_length/lc)}', ha='center', va='center', fontsize=11, color='black',fontdict={'fontname':'DejaVu Sans'})
# plt.text(15,y_leg*1.02,'$n_c$ = ', ha='center', va='center', fontsize=11, color='black',fontdict={'fontname':'DejaVu Sans'})
# plt.xlabel('Cell length [m]')
# plt.ylabel('Aperture SC dipole [cm]')
# plt.ylim(ymin=0)
# plt.xlim(xmin=0)
# plt.show()

print('RESULTS')
print('nc_opt', nc_opt)
print('cell length opt', RCS_opt.cell_length)
print('quad length', l_quad_opt)
print('filling ratio of arc', RCS_opt.filling_ratio)
print('MCF', MCF)