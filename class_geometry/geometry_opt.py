import numpy as np
import matplotlib.pyplot as plt
import json

from rcsparameters.chain.chain import RCSChain
from rcsparameters.geometry.geometry import Geometry
import matplotlib
from matplotlib.colors import Normalize

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize

file_input='/mnt/c/muco/code/class_geometry/parameter_files/para_RCS_ME.txt'
emitt_n=25e-6
sigma_delta=10e-3
n_sigma=2*6
dbeta=0.20
sigma_inc=1.1
tol_dspur_lhc=1.3e-2

def rot_coord(RCS,n_bloc,x):
    theta=RCS.cell_angle/8
    coor_init=RCS.zn(x)[1:(1+n_bloc*11)]
    coor=coor_init-coor_init[0]
    x,y=np.real(coor),np.imag(coor)
    x_rot=x*np.cos(theta)+y*np.sin(theta)
    y_rot=-x*np.sin(theta)+y*np.cos(theta)
    return(x_rot, y_rot)

# RCS = Geometry(file_input,dipole_spacing=0.4)
mu=np.pi/2
G_max=40

list_pattern=[['BSC','BNC','BSC'],['BSC','BNC','BSC','BNC','BSC'],
              ['BSC','BNC','BSC','BNC','BSC','BNC','BSC'], # 1bloc
              ['BSC','BNC','BSC','BSC','BNC','BSC'],
              ['BSC','BNC','BSC','BNC','BSC','BSC','BNC','BSC','BNC','BSC']] #2 bloc
name_pattern=['3dip 1bloc','5dip 1bloc','7dip 1bloc','3dip 2bloc','5dip 2bloc']

results=np.zeros((len(list_pattern),8))
for i_pat, pattern in enumerate(list_pattern):
    RCS= Geometry(file_input, dipole_spacing=0.4, pattern=pattern)
    print('Pattern', pattern)
    L_arc=RCS.arc_length
    L_cell=RCS.cell_length
    L_nc=RCS.dipole_families['BNC']['tot_length']/RCS.nb_arc
    L_sc=RCS.dipole_families['BSC']['tot_length']/RCS.nb_arc
    L_dd=RCS.dipole_spacing
    nd=len(RCS.pattern)
    Brho_ext=RCS.ext_Brho
    nb_cell_arc=RCS.nb_cell_arc
    emitt=emitt_n/RCS.inj_gamma

    delta=(L_dd*(nd+1))**2+ 8*Brho_ext*np.sin(mu/2)*(L_arc-L_nc-L_sc)/G_max/L_arc
    nc_max=int((-(nd+1)*L_dd + np.sqrt(delta))/(8*Brho_ext*np.sin(mu/2)/G_max/L_arc))
    L_qp_min=4*Brho_ext*np.sin(mu/2)/G_max/L_arc*nc_max
    RCS_opt = Geometry(file_input,dipole_spacing=0.4, nb_cell_arc=nc_max, LSSS=L_qp_min,pattern=pattern)

    print('Length of 1 quad',np.round(L_qp_min,2), '[m]')
    print('Number max of n_c, analytical', nc_max,'class', RCS_opt.nb_cell_arc)
    print('Cell length',np.round(RCS_opt.cell_length,2),'[m]')

    beta_max=RCS_opt.cell_length*(1+np.sin(mu/2))/np.sin(mu)
    D_max=RCS_opt.cell_length*RCS_opt.cell_angle*(1+np.sin(mu/2)/2)/(4*(np.sin(mu/2))**2)
    # D_spur=D_max*(sigma_inc-1) + sigma_inc*D_max*np.sqrt(1+dbeta)*tol_dspur_lhc*np.sqrt(beta_max)
    D_spur=0
    dbeta=0
    print('Twiss function, beta, D, D_spur', beta_max, D_max, D_spur)
    
    beam_size_rms=n_sigma*np.sqrt(emitt*beta_max+(D_max*sigma_delta)**2) 
    beam_size_mx=sigma_delta*(D_max+D_spur) + n_sigma*np.sqrt(emitt*beta_max*(1+dbeta))
    print(f'Beam size {n_sigma} sigma RMS', np.round(beam_size_rms,4), '[m]')
    print(f'Beam size {n_sigma} sigma MADX', beam_size_mx)
    beam_size=beam_size_mx

    if "1bloc" in name_pattern[i_pat]:
        mask_nc=[item == 'BNC' for item in RCS.pattern]
        mask_sc=[item == 'BSC' for item in RCS.pattern]
        max_width_dip=RCS.extrema(RCS.t_ext)[1]-RCS.extrema(RCS.t_inj)[0]
        width_sc=np.max(max_width_dip[mask_sc])
        width_nc=np.max(max_width_dip[mask_nc])

        aperture_sc=width_sc+beam_size
        aperture_nc=width_nc+beam_size
        print('1bloc, Excursion SC', np.round(width_sc,3), 'Aperture SC', np.round(aperture_sc,3), '[m]')
        print('1bloc, Excursion NC', np.round(width_nc,3), 'Aperture NC', np.round(aperture_nc,3), '[m]')
        RCS_opt.plot_traj()

    elif "2bloc" in name_pattern[i_pat]:
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
        aperture_sc=width_sc+beam_size
        aperture_nc=width_nc+beam_size
        print('2bloc, Excursion SC', np.round(width_sc,3), 'Aperture SC', np.round(aperture_sc,3), '[m]')
        print('2bloc, Excursion NC', np.round(width_nc,3), 'Aperture NC', np.round(aperture_nc,3), '[m]')
        RCS_opt.plot_traj()
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

    results[i_pat,0]=beam_size
    results[i_pat,1]=width_sc
    results[i_pat,2]=width_nc
    results[i_pat,3]=aperture_sc
    results[i_pat,4]=aperture_nc
    results[i_pat,5]=RCS_opt.dipole_families['BSC']['length']
    results[i_pat,6]=RCS_opt.dipole_families['BNC']['length']
    results[i_pat,7]=nc_max

num=np.arange(0,5,1)
plt.figure()
plt.scatter(num,results[:,7])
plt.xticks(ticks=num,labels=name_pattern)
plt.xlabel('Pattern')
plt.ylabel('$n_c$ max')
plt.show()

plt.figure()
plt.scatter(num,results[:,0]*1e3, label='beam size',color='tab:green', marker='*',s=100)
plt.xticks(ticks=num,labels=name_pattern)
plt.xlabel('Pattern')
plt.ylabel('Beam size [mm]')
plt.legend(loc='upper left')
plt.ylim(ymin=0)
plt.show()

plt.figure()
plt.scatter(num,results[:,1]*1e3, label='width sc', color='tab:blue')
# plt.scatter(num,results[:,2]*1e3, label='width nc',color='tab:orange')
plt.scatter(num,results[:,3]*1e3, label='aperture sc', marker='s',color='blue')
# plt.scatter(num,results[:,4]*1e3, label='aperture nc', marker='s',color='tab:orange')
# plt.scatter(num,results[:,0]*1e3, label='beam size',color='tab:green', marker='*')
plt.xticks(ticks=num,labels=name_pattern)
plt.xlabel('Pattern')
plt.ylabel('Width SC [mm]')
plt.legend(loc='upper left')
plt.ylim(ymin=0)
plt.show()

plt.figure()
# plt.scatter(num,results[:,1]*1e3, label='width sc', color='tab:blue')
plt.scatter(num,results[:,2]*1e3, label='width nc',color='tab:orange')
# plt.scatter(num,results[:,3]*1e3, label='aperture sc', marker='s',color='tab:blue')
plt.scatter(num,results[:,4]*1e3, label='aperture nc', marker='s',color='red')
# plt.scatter(num,results[:,0]*1e3, label='beam size',color='tab:green', marker='*')
plt.xticks(ticks=num,labels=name_pattern)
plt.xlabel('Pattern')
plt.ylabel('Width NC [mm]')
plt.legend(loc='upper left')
plt.ylim(ymin=0)
plt.show()

plt.figure()
plt.scatter(num,results[:,5], label='length sc', color='tab:blue')
plt.scatter(num,results[:,6], label='length nc', color='tab:orange')
plt.xticks(ticks=num,labels=name_pattern)
plt.xlabel('Pattern')
plt.ylabel('Lengh dipole [m]')
plt.legend()
plt.ylim(ymin=0)
plt.show()

# RCS = Geometry(file_input, dipole_spacing=0.4)
# L_cell_min=10
# L_cell_max=200
# L_cell = np.linspace(L_cell_min, L_cell_max, 30)
# L_qp_min = 4 * RCS.ext_Brho * np.sin(mu / 2) / G_max / L_cell
# L_cell_lsss=4 * RCS.ext_Brho * np.sin(mu / 2) / G_max / RCS.LSSS
# plt.figure()
# plt.plot(L_cell, L_qp_min,color='k')
# y_leg=plt.ylim()[1]*1.05
# L_previous=L_cell_min
# cmap=plt.get_cmap('viridis')
# norm = Normalize(vmin=1, vmax=RCS.nb_cell_arc)
# for nc in (np.linspace(RCS.nb_cell_arc, 1, RCS.nb_cell_arc)):
#     color = cmap(norm(nc))
#     RCSt = Geometry(file_input, dipole_spacing=0.4, nb_cell_arc=nc)
#     midpoint = (L_previous + RCSt.cell_length) / 2
#     plt.axvspan(L_previous,RCSt.cell_length , alpha=0.5, color=color)
#     plt.text(midpoint,y_leg*0.98, f'{int(nc)}', ha='center', va='center', fontdict={'fontname':'DejaVu Sans'}, fontsize=10, color='black')
#     L_previous=RCSt.cell_length
# plt.text(L_cell_min,y_leg*0.98,'nc =', ha='center', va='center', fontsize=10, color='black',fontdict={'fontname':'DejaVu Sans'})
# plt.axhline(RCS.LSSS, color='grey', linestyle='dotted')
# plt.scatter(L_cell_lsss,RCS.LSSS, marker='+', color='red')
# plt.xlabel('Cell length [m]')
# plt.ylabel('Minimum QP length [m]')
# plt.show()

# RCS = Geometry(file_input, dipole_spacing=0.4)
L_cell_min=10
L_cell_max=200
L_cell = np.linspace(L_cell_min, L_cell_max, 30)
# L_qp_min = 4 * RCS.ext_Brho * np.sin(mu / 2) / G_max / L_cell
# plt.figure()
# plt.plot(L_cell, 2*L_qp_min/L_cell*100, color='k')
# y_leg=plt.ylim()[1]*1.05
# L_previous=L_cell_min
# cmap=plt.get_cmap('viridis')
# norm = Normalize(vmin=1, vmax=RCS.nb_cell_arc)
# for nc in (np.linspace(RCS.nb_cell_arc, 1, RCS.nb_cell_arc)):
#     color = cmap(norm(nc))
#     RCSt = Geometry(file_input, dipole_spacing=0.4, nb_cell_arc=nc)
#     midpoint = (L_previous + RCSt.cell_length) / 2
#     plt.axvspan(L_previous,RCSt.cell_length , alpha=0.5, color=color)
#     plt.text(midpoint,y_leg*0.98, f'{int(nc)}', ha='center', va='center', fontdict={'fontname':'DejaVu Sans'}, fontsize=10, color='black')
#     L_previous=RCSt.cell_length
# plt.text(L_cell_min,y_leg*0.98,'nc =', ha='center', va='center', fontsize=10, color='black',fontdict={'fontname':'DejaVu Sans'})
# plt.xlabel('Cell length [m]')
# plt.ylabel('Relative QP length [%]')
# plt.show()

# RCS = Geometry(file_input, dipole_spacing=0.4)
# Bp=RCS.ext_Brho
# E_muon=105.66e6
# E_ext=RCS.E_ext
# def Lmin_quad(Lc, mu=np.pi/2, Bp=Bp, Bpole=1, eps_n=25e-6, n=10, delta=10e-3, Larc=RCS.tot_arc_length, dx=10e-3, E_ext=E_ext):
#     gamma=E_ext/E_muon
#     eps=eps_n/gamma
#     beta=Lc*(1+np.sin(mu/2))/np.sin(mu)
#     disp=2*np.pi*Lc**2/Larc*(1+1/2*np.sin(mu/2))/(4*(np.sin(mu/2))**2)
#     size=n*np.sqrt(beta*eps)+ delta*disp + dx
#     phi=size
#     return(4*np.sin(mu/2)/Lc*Bp/Bpole*phi,phi)
# lquad,size_beam=Lmin_quad(L_cell)
# plt.plot(L_cell,lquad, color='k')
# y_leg=plt.ylim()[1]*1.05
# L_previous=L_cell_min
# cmap=plt.get_cmap('viridis')
# norm = Normalize(vmin=1, vmax=RCS.nb_cell_arc)
# for nc in (np.linspace(RCS.nb_cell_arc, 1, RCS.nb_cell_arc)):
#     color = cmap(norm(nc))
#     RCSt = Geometry(file_input, dipole_spacing=0.4, nb_cell_arc=nc)
#     midpoint = (L_previous + RCSt.cell_length) / 2
#     plt.axvspan(L_previous,RCSt.cell_length , alpha=0.5, color=color)
#     plt.text(midpoint,y_leg*0.98, f'{int(nc)}', ha='center', va='center', fontdict={'fontname':'DejaVu Sans'}, fontsize=10, color='black')
#     L_previous=RCSt.cell_length
#     print(L_previous)
# plt.text(L_cell_min,y_leg*0.98,'nc =', ha='center', va='center', fontsize=10, color='black',fontdict={'fontname':'DejaVu Sans'})
# plt.scatter(L_cell[np.argmin(lquad)],min(lquad), color='red')
# plt.axhline(RCS.LSSS, color='k', linestyle='dotted')
# plt.xlabel('Cell length [m]')
# plt.ylabel('Minimum QP length [m]')
# plt.ylim(0,plt.ylim()[1])
# plt.show()
# plt.figure()
# plt.plot(L_cell, size_beam*1e2)
# plt.xlabel('Cell length [m]')
# plt.ylabel('Aperture [cm]')
# plt.show()

RCS = Geometry(file_input, dipole_spacing=0.4)
Bp=RCS.ext_Brho
E_muon=105.66e6
E_ext=RCS.E_ext
L_arc=RCS.tot_arc_length
def Lmin_quad(Lc, mu=np.pi/2, Bp=Bp, Bpole=1, eps_n=25e-6, n=6, delta=5e-3, Larc=L_arc, dx=10e-3, E_ext=E_ext):
    gamma=E_ext/E_muon
    eps=eps_n/gamma
    beta=Lc*(1+np.sin(mu/2))/np.sin(mu)
    disp=2*np.pi*Lc**2/Larc*(1+1/2*np.sin(mu/2))/(4*(np.sin(mu/2))**2)
    size=n*np.sqrt(beta*eps)+ delta*disp + dx
    phi=size
    return(4*np.sin(mu/2)/Lc*Bp/Bpole*phi,phi)
lquad,size_beam=Lmin_quad(L_cell)
plt.plot(L_cell,lquad)
plt.scatter(L_cell[np.argmin(lquad)],min(lquad), color='red')
plt.xlabel('Cell length [m]')
plt.ylabel('Minimum QP length [m]')
plt.ylim(0,plt.ylim()[1])
plt.show()
plt.figure()
plt.plot(L_cell, size_beam*1e3)
plt.xlabel('Cell length [m]')
plt.ylabel('Aperture [mm]')
plt.ylim(ymin=0)
plt.show()

# mu=np.pi/2
# G_max=40
# L_arc=RCS_opt.arc_length
# L_cell=RCS_opt.cell_length
# L_nc=RCS_opt.dipole_families['BNC']['tot_length']/RCS_opt.nb_arc
# L_sc=RCS_opt.dipole_families['BSC']['tot_length']/RCS_opt.nb_arc
# L_dd=RCS_opt.dipole_spacing
# nd=len(RCS_opt.pattern)
# Brho_ext=RCS_opt.ext_Brho
# nb_cell_arc=RCS_opt.nb_cell_arc

# delta=(L_dd*(nd+1))**2+ 8*Brho_ext*np.sin(mu/2)*(L_arc-L_nc-L_sc)/G_max/L_arc
# nc_max=int((-(nd+1)*L_dd + np.sqrt(delta))/(8*Brho_ext*np.sin(mu/2)/G_max/L_arc))
# L_qp_min=4*Brho_ext*np.sin(mu/2)/G_max/L_arc*nc_max
# print(L_qp_min)