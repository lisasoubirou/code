# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:27:51 2022

@author: Achance
"""

import yaml
import json
import numpy as np
from pathlib import Path

import scipy.constants as cons
from scipy.optimize import brentq, fminbound, minimize
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline

from multibend_python_general import MultiBend

class PulsedSynch(object):
    default_values = {
     "hybrid": 0,
     "length": 5990,
     "inj_E": 	63e9,
     "ext_E": 	300e9,
     "RF_freq":  1300.011e6,
     "max_gradient": 30e6,
     "RF_phase": 	135,
     "dipoles":
         {"B1": 10., 
          "B2": (1.5, "linear")
          },
     "pattern": ["B1", "B2", "B1"],
     "nb_traj": 10,
     "max_B_slope": 10000,
     "rep_rate": 5,
     "target_ramping": 	0.3e-3,
     "inj_phi": -90,
     "ext_phi":  90,
     "nb_arc": 6,
     "dipole_spacing": 0.3,
     "QP_dipole_spacing": None,
     "nb_RF_section": 6,
     "nb_cell_arc": 12,
     "phase_advance": 90,
     "filling_ratio": 0.85,
     "filling_ratio_RF": 0.4,
     "nb_cell_insertion": 2,
     "QP_length": 1,
     "SX_length": 0.5,
     "SSS_length": 2.0,
     "hor_norm_emittance": 25e-6,
     "vert_norm_emittance": 25e-6,
     "long_norm_emittance": 16e-3,
     "ref_B": 1.7,
     "dipole_height": 40e-3,
     "dipole_width": 80e-3,
     "nb_sigma": 6,
     "nb_sigma_E": 3,
     "max_sigE": 3e-3,
     "dipole_histeresis": 6.5e-3,
     "dipole_eddy": 0.15e-3,
     "core_section": 0.08,
     "core_mass": 612,
     "pow_blowup": 0.5,
     }
    
    def __init__(self, file_input = None, **kw):
        self.muon_mass = cons.physical_constants[
            "muon mass energy equivalent in MeV"][0]*1e6
        self.muon_lifetime = 2.1969811e-6
        kw_def = dict(PulsedSynch.default_values)
        for attr, val in kw_def.items():
            setattr(self, attr, val)
        file_input = np.array(file_input, ndmin=1)
        file_input = np.append(file_input, [dict(kw)])
        for fin in file_input:
            if fin is None:
                data = {}
            elif isinstance(fin, dict):
                data = fin
            elif Path(fin).exists():
                try:
                    with open(fin, "r") as f:
                        data = json.load(f)
                        print(f"Load json file {fin}")
                except:
                    try:
                        with open(fin, "r") as f:
                            data = yaml.safe_load(f)
                            print(f"Load yaml file {fin}")
                    except:
                        raise yaml.parser.ParserError(f"{fin} is an invalid file")    
            else:
                raise FileExistsError(f"{fin} does not exist")
            for attr, val in data.items():
                if attr in kw_def:
                    setattr(self, attr, val)
                else:
                    raise KeyError("{} is an invalid key value".format(attr))
        self.inj_phi *= np.pi/180.
        self.ext_phi *= np.pi/180.
        self.RF_phase *= np.pi/180.
        self.phase_advance *= np.pi/180.
        self.set_params()

    def set_params(self, **kw):
        for attr, val in kw.items():
            setattr(self, attr, val)
        self.nb_dipole_cell = len(self.pattern)
        self.tab_func = []
        list_key = sorted(list(self.dipoles.keys()))

        def funcB(i):
            key = list_key[i]
            val = self.dipoles[key]
            if isinstance(val, (float, int)):
                return lambda x: val
            elif isinstance(val, (tuple, list, np.ndarray)):
                if val[-1] == "linear":
                    if len(val) == 2:
                        Bmax = val[0]
                        return lambda x: (1-2*np.abs(np.mod(x+np.pi/2, 2*np.pi)-np.pi)/np.pi)*Bmax
                    elif len(val == 3):
                        mean = 0.5*(val[0]+val[1])
                        diff = np.abs(0.5*(val[1]-val[0]))
                        return lambda x: (2.*np.abs(np.mod(x-np.pi/2, 2*np.pi))/np.pi-1.)*diff+mean
                    else:
                        raise ValueError(f"The dipole family {key} is not valid !")
                else:
                    tab = np.array(val)
                    return lambda x: np.sum([v*np.sin(i*x) for i,v in enumerate(tab)])
            elif isinstance(val, str):
                return eval(val)
            else:
                raise ValueError(f"The dipole family {key} is not valid !")

        def funcdB(i):
            key = list_key[i]
            val = self.dipoles[key]
            if isinstance(val, (float, int)):
                return lambda x: 0.
            elif isinstance(val, (tuple, list, np.ndarray)):
                if val[-1] == "linear":
                    if len(val) == 2:
                        Bmax = val[0]
                        return lambda x: 2*np.sign(np.mod(x-np.pi/2, 2*np.pi)-np.pi)/np.pi*Bmax
                    elif len(val == 3):
                        diff = np.abs(0.5*(val[1]-val[0]))
                        return lambda x: 2*np.sign(np.mod(x-np.pi/2, 2*np.pi))/np.pi*diff
                    else:
                        raise ValueError(f"The dipole family {key} is not valid !")
                else:
                    tab = np.array(val)
                    return lambda x: np.sum([i*v*np.cos(i*x) for i,v in enumerate(tab)])
            elif isinstance(val, str):
                func = eval(val)
                phi = np.linspace(-np.pi, np.pi, 1001)
                return UnivariateSpline(phi, func(phi), s=0).derivative()
            else:
                raise ValueError(f"The dipole family {key} is not valid !")

        self.dipole_families = dict()
        for i_key, key in enumerate(list_key):
            self.dipole_families[key] = dict()
            self.dipole_families[key]["B"] = funcB(i_key)
            self.dipole_families[key]["dB"] = funcdB(i_key)
            
        self.n_dipole_families = len(self.dipole_families)
        for key in self.pattern:
            if key not in self.dipole_families.keys():
                raise ValueError(f"{key} is not defined")
        self.init_all()

    def init_all(self):
        self.init_params()
        self.init_lattice()
        self.init_magnet()
        self.init_RF()

    def init_params(self):
        self.inj_gamma = self.inj_E/self.muon_mass
        self.ext_gamma = self.ext_E/self.muon_mass
        self.inj_Brho = np.sqrt(self.inj_gamma**2-1)*self.muon_mass/cons.c
        self.ext_Brho = np.sqrt(self.ext_gamma**2-1)*self.muon_mass/cons.c
        self.rev_time = self.length/cons.c
        self.rev_freq = 1./self.rev_time        
        self.tot_arc_length = self.filling_ratio*self.length
        self.tot_insertion_length = self.length-self.tot_arc_length
        self.arc_length = self.tot_arc_length/self.nb_arc
        self.insertion_length = self.tot_insertion_length/self.nb_arc
        self.RF_length_tot = self.filling_ratio_RF*self.insertion_length*self.nb_RF_section
        self.nb_cell_arcs = self.nb_cell_arc*self.nb_arc
        self.cell_length = self.arc_length/self.nb_cell_arc
        self.cell_angle = 2*np.pi/self.nb_cell_arcs
        
        e0 = np.sin(self.cell_angle/4)
        if self.n_dipole_families == 1:
            key = list(self.dipole_families.keys())[0]
            dip = self.dipole_families[key]
            Bext = dip["B"](self.ext_phi)
            L0 = self.ext_Brho/Bext*2*e0
            self.dipole_length_tot = 2*self.nb_cell_arcs*L0
            dip["length_half_cell"] = L0
            dip["tot_length"] = self.dipole_length_tot
            dip["nb_half_cell"] = len(self.pattern)
            dip["nb_tot"] = dip["nb_half_cell"]*2*self.nb_cell_arcs
            dip["length"] = self.dipole_length_tot/dip["nb_tot"]
            dip["BL"] = lambda x: dip["length_half_cell"]*dip["B"](x)

            self.list_a = np.ones(dip["nb_half_cell"])/dip["nb_half_cell"]
            self.list_b = np.ones((self.nb_traj, len(self.pattern)))
            f = lambda x: dip["B"](x)*L0- 2*self.inj_Brho*e0
            self.inj_phi = brentq(f, -np.pi/2, self.ext_phi)
            self.func_Brho = lambda x: L0*dip["B"](x)/2./e0
            self.func_dBrho = lambda x: L0*dip["dB"](x)/2./e0
        else:
            A = np.ones((2, self.n_dipole_families))
            list_key = sorted(self.dipole_families.keys())
            list_dip = [self.dipole_families[key] for key in list_key]
            A[0] = [dip["B"](self.inj_phi)/2./e0 for dip in list_dip]
            A[1] = [dip["B"](self.ext_phi)/2./e0 for dip in list_dip]
            B = np.ones(2)
            B[0] = self.inj_Brho
            B[1] = self.ext_Brho
            if np.abs(np.linalg.det(A[:,:2])) > 1e-6 and self.n_dipole_families == 2:
                invA = np.linalg.inv(A[:,:2])
                Lsol = np.dot(invA, B)
            else:
                list_phi = np.linspace(self.inj_phi, self.ext_phi, 101)[1:-1]
                C = np.ones((len(list_phi), self.n_dipole_families))
                for i, phi in enumerate(list_phi):
                    C[i] = [dip["B"](phi)/2./e0 for dip in list_dip]
                D = self.inj_Brho+(self.ext_Brho-self.inj_Brho)*(list_phi-self.inj_phi)/(self.ext_phi-self.inj_phi)
                constraint = ({'type': 'eq', 
                              'fun': lambda x:  np.sum((np.dot(A, x) - B)**2)},)
                bnds = [(0, None)]*self.n_dipole_families
                res = minimize(lambda x: np.sum((np.dot(C, x) - D)**2), 
                                np.ones(self.n_dipole_families)/self.n_dipole_families, 
                                method='SLSQP', bounds=bnds,
                                constraints=constraint)
                if not res.success:
                    raise ValueError("No solution for this set of dipole families")
                Lsol = res.x
                # print(A, list_key)
                # invA = np.linalg.inv(A[:,:2])
                # Adata = C[:,2:]-np.dot(np.dot(C[:,:2], invA), A[:,2:])
                # Bdata = D-np.dot(np.dot(C[:,:2], invA), np.ones(2))
                # Lsol = np.zeros(self.n_dipole_families)
                # Lsol[2:] = np.linalg.lstsq(Adata, Bdata, rcond=None)[0]
                # Lsol[:2] = np.dot(invA, np.ones(2)-np.dot(A[:,2:], Lsol[2:]))
            # else:
            #     Lsol = np.dot(invA, np.ones(2))
            L0 = np.sum(Lsol)
            self.dipole_length_tot = 2*self.nb_cell_arcs*L0
            pattern = list(self.pattern)
            for i, dip in enumerate(list_dip):
                dip["length_half_cell"] = Lsol[i]
                dip["tot_length"] = 2*self.nb_cell_arcs*Lsol[i]
                dip["nb_half_cell"] = pattern.count(list_key[i])
                dip["length"] = Lsol[i]/dip["nb_half_cell"]
                dip["nb_tot"] = dip["nb_half_cell"]*2*self.nb_cell_arcs
                dip["BL"] = lambda x: dip["length_half_cell"]*dip["B"](x)
            self.func_Brho = lambda x: np.sum([l*dip["B"](x) for l, dip in zip(Lsol, list_dip)])/2./e0
            self.func_dBrho = lambda x: np.sum([l*dip["dB"](x) for l, dip in zip(Lsol, list_dip)])/2./e0
            # self.func_Brho = lambda x: np.sum([dip["BL"](x) for l, dip in zip(Lsol, list_dip)])/2./e0
            self.list_a = np.ones(len(self.pattern))
            for i, key in enumerate(self.pattern):
                dip = self.dipole_families[key]
                self.list_a[i] = dip["length_half_cell"]/L0/dip["nb_half_cell"]
            self.list_b = np.ones((self.nb_traj, len(self.pattern)))
            for i, phi in enumerate(np.linspace(self.inj_phi, self.ext_phi, self.nb_traj)):
                Brho = self.func_Brho(phi)
                # tab_Bl = [dip["length_half_cell"]*dip["B"](phi) for dip in list_dip]
                # sum_Bl = np.sum(tab_Bl)
                for j, key in enumerate(self.pattern):
                    dip = self.dipole_families[key]
                    # Bl = dip["length_half_cell"]*dip["B"](phi)
                    self.list_b[i, j] = L0*dip["B"](phi)/Brho/2/e0
        self.func_gamma = lambda x: np.sqrt((self.func_Brho(x)/self.muon_mass*cons.c)**2+1.)
        self.dipole_length = self.dipole_length_tot/self.nb_cell_arcs
        if self.dipole_spacing is not None:
            self.QP_dipole_spacing = (
                self.cell_length/2-L0-(len(self.pattern)-1)*self.dipole_spacing-self.SSS_length)/2.
            self.nc_best = int((self.arc_length-self.dipole_length_tot/self.nb_arc)/((len(self.pattern)-1)*self.dipole_spacing+self.SSS_length)/2)
        elif self.QP_dipole_spacing is not None:
            self.dipole_spacing = (self.cell_length/2-L0-2*self.QP_dipole_spacing-self.SSS_length)/(len(self.pattern)-1)
        else:
            spacing = (self.cell_length/2-L0-self.SSS_length)/(len(self.pattern)+1)
            self.dipole_spacing = spacing
            self.QP_dipole_spacing = spacing
            
        self.geometry = MultiBend(
            Ld = self.dipole_length_tot,
            nc = self.nb_cell_arcs,
            a = self.list_a,
            b = self.list_b,
            nsub = 10,
            Ldd = self.dipole_spacing,
            Lqq = 0.15,
            Lbq= self.QP_dipole_spacing,
            LQP = self.QP_length,
            LSX = self.SX_length,
            LSSS= self.SSS_length,
            Larc = self.tot_arc_length,
            mux = self.phase_advance
            )
            
        self.max_voltage = self.RF_length_tot*self.max_gradient
        self.max_energy_gain = self.max_voltage*np.sin(self.RF_phase)
        self.max_dgamma_dt = self.max_energy_gain/self.muon_mass/self.rev_time
        # self.max_dgamma_dphi = -fminbound(lambda x: -np.abs(self.func_dBrho(x))*cons.c/self.muon_mass, 
        #                             self.inj_phi, self.ext_phi, 
        #                             full_output=True)[1]
        self.max_dgamma_dphi = (self.ext_gamma-self.inj_gamma)/(self.ext_phi-self.inj_phi)
        self.min_period = 2*np.pi*self.max_dgamma_dphi/self.max_dgamma_dt
        self.max_dB_dphi = 0.
        for key, dip in self.dipole_families.items():
            dip["B_max"] = -fminbound(lambda x: -np.abs(dip["B"](x)), 
                                        -np.pi, np.pi, 
                                        full_output=True)[1]
            dip["dB_max"] = -fminbound(lambda x: -np.abs(dip["dB"](x)), 
                                        -np.pi, np.pi, 
                                        full_output=True)[1]
            self.max_dB_dphi = max(self.max_dB_dphi, dip["dB_max"])
        self.min_period = max(self.min_period, 2*np.pi*self.max_dB_dphi/self.max_B_slope)         
        if self.target_ramping is None:
            self.dipole_period = self.min_period
        else:
            self.dipole_period = max(self.min_period, 
                                     2*np.pi/(self.ext_phi-self.inj_phi)*self.target_ramping)
        self.ramp_time = (self.ext_phi-self.inj_phi)/2/np.pi*self.dipole_period
        self.nb_turns = self.ramp_time*self.rev_freq
        self.peak_B_slope = self.max_dB_dphi*2*np.pi/self.dipole_period
        dgamma_dt = 2*np.pi*self.max_dgamma_dphi/self.dipole_period
        self.energy_gain = dgamma_dt*self.muon_mass*self.rev_time
        self.tot_RF_voltage = self.energy_gain/np.sin(self.RF_phase)
        self.func_phis = lambda x: np.pi-np.arcsin(self.func_dBrho(x)*cons.c*2*np.pi/self.dipole_period*self.rev_time/self.tot_RF_voltage)
        self.acceleration_time = quad(lambda x: 1./self.func_gamma(x), self.inj_phi, self.ext_phi)[0]*self.dipole_period/2/np.pi
        self.muon_survival = np.exp(-self.acceleration_time/self.muon_lifetime)
        self.sagitta = self.geometry.max_apert
        self.diff_path = self.geometry.path_length_diff
        self.diff_path_tot = self.geometry.path_length_diff_tot

    def init_lattice(self):
        mu = self.phase_advance
        phi = 2*np.pi/self.nb_cell_arcs
        self.alfa = (phi/2)**2*(1/np.sin(mu)**2-1./12)
        self.gamma_tr = 1./np.sqrt(self.alfa)
        self.energy_tr = self.gamma_tr*self.muon_mass
        self.inj_QP_gradient = 4*np.sin(mu/2)/self.cell_length*self.inj_Brho/self.QP_length
        self.ext_QP_gradient = 4*np.sin(mu/2)/self.cell_length*self.ext_Brho/self.QP_length
        self.max_betx = (1+np.sin(mu/2))/np.sin(mu)*self.cell_length
        self.max_Dx = (1+0.5*np.sin(mu/2))/4/np.sin(mu/2)**2*self.cell_length*phi
        self.sigx = lambda x: np.sqrt(self.max_betx*self.hor_norm_emittance/self.func_gamma(x))
        self.sigy = lambda x: np.sqrt(self.max_betx*self.vert_norm_emittance/self.func_gamma(x))
        self.max_sigx = np.sqrt(self.max_betx*self.hor_norm_emittance/self.inj_gamma)
        self.max_sigy = np.sqrt(self.max_betx*self.vert_norm_emittance/self.inj_gamma)
        self.max_apertx = 2*self.nb_sigma*self.max_sigx+self.max_sigE*self.max_Dx + self.geometry.max_apert
        self.max_apertx_noshift = 2*self.nb_sigma*self.max_sigx+self.max_sigE*self.max_Dx + self.geometry.max_apert_noshift
        self.max_aperty = 2*self.nb_sigma*self.max_sigy

    def init_magnet(self):
        self.dipole_frequency = 1./self.dipole_period
        self.duty_factor = self.rep_rate*self.dipole_period
        self.core_loss_ref = self.dipole_frequency*(
            self.dipole_histeresis+self.dipole_eddy*self.dipole_frequency)
        self.max_NC_field = 0.
        self.max_SC_field = 0.
        self.NC_length_tot = 0.
        self.SC_length_tot = 0.
        for key, dip in self.dipole_families.items():
            if dip["B_max"] < 3.:
                self.max_NC_field = max(self.max_NC_field, dip["B_max"])                
                self.NC_length_tot += dip["tot_length"]
            else:
                self.max_SC_field = max(self.max_SC_field, dip["B_max"])                
                self.SC_length_tot += dip["tot_length"]            
        self.core_loss_max = self.core_loss_ref*self.max_NC_field/self.ref_B
        self.core_loss_ave = self.core_loss_max*self.duty_factor
        self.core_loss_meter = self.core_mass*self.core_loss_ave
        self.core_loss_tot = self.core_loss_meter*self.NC_length_tot
        self.volume_gap = self.dipole_height*self.dipole_width*self.NC_length_tot
        self.NC_energy_tot = self.max_NC_field**2/2/cons.mu_0*self.volume_gap
        self.peak_NC_power = self.peak_B_slope**2*self.dipole_period/cons.mu_0*self.volume_gap
        
    def init_RF(self):
        self.RF_wavelength = cons.c/self.RF_freq
        self.inj_long_rms_emittance = self.long_norm_emittance/self.inj_gamma*2*np.pi/self.RF_wavelength
        self.RF_harmonic = self.RF_freq/self.rev_freq
        self.energy_gain_guess = self.energy_gain
        mid_E = (self.inj_E+self.ext_E)/2
        self.separatrix_factor = np.sqrt(
            2*self.energy_gain_guess/self.length*mid_E*self.RF_wavelength/np.pi/self.alfa
            )/mid_E
        def phi_bounds(phis):
            phi1 = np.pi-2*phis
            S = np.sin(phis)
            C = np.cos(phis)
            ham = 2*C - phi1*S
            f = lambda x: S*(x-np.sin(x))-C*(1-np.cos(x)) + ham
            phi2 = brentq(f, -np.pi, 0)
            return (phi1, phi2)
        
        def surf_separatrix(phis):
            phi1, phi2 = phi_bounds(phis)
            S = np.sin(phis)
            C = np.cos(phis)/S
            f = lambda x: np.sqrt(C*(1+np.cos(x))+x-np.sin(x)+2*phis-np.pi)
            return (phi1, phi2, 2*quad(f, phi2, phi1)[0])
                
        blowup = (mid_E/self.inj_E)**self.pow_blowup
        target = blowup*self.nb_sigma_E**2*self.inj_long_rms_emittance/self.separatrix_factor
        def func_phi(tar):
            f = lambda x: surf_separatrix(x)[-1]-tar
            sol_phi = np.pi-brentq(f, 1e-6, np.pi/2)
            return sol_phi
        # sol_phi = np.vectorize(func_phi)(target)
        sol_phi = self.RF_phase
        self.RF_phase_guess = sol_phi*180./np.pi
        self.voltage_guess = self.energy_gain_guess/np.sin(sol_phi)
        self.energy_acceptance = np.sqrt(
            -2*self.energy_gain_guess/self.length*(
                np.cos(sol_phi)-(np.pi/2-sol_phi)*np.sin(sol_phi)
                )*mid_E*self.RF_wavelength/np.pi/self.alfa)/mid_E        
        inj_eta = 1/self.inj_gamma**2-self.alfa
        ext_eta = 1/self.ext_gamma**2-self.alfa
        self.inj_Qs = np.sqrt(
            self.RF_harmonic*inj_eta/2/np.pi/self.inj_E*self.tot_RF_voltage*np.cos(self.RF_phase))
        self.ext_Qs = np.sqrt(
            self.RF_harmonic*ext_eta/2/np.pi/self.ext_E*self.tot_RF_voltage*np.cos(self.RF_phase))
        
    def print_input(self):
        print(f"     {'Input parameters':50s}")
        print(f"{'Hybrid RCS':56s} {self.hybrid:d}")
        print(f"{'Circumference':50s} {self.length/1000:18.6e} km")
        print(f"{'Injection total energy':50s} {self.inj_E/1e9:18.6e} GeV")
        print(f"{'Extraction total energy':50s} {self.ext_E/1e9:18.6e} GeV")
        print(f"{'RF frequency':50s} {self.RF_freq/1e6:18.6e} MHz")
        print(f"{'Max average RF gradient':50s} {self.max_gradient/1e6:18.6e} MV/m")
        print(f"{'RF phase':50s} {self.RF_phase*180/np.pi:18.6e} deg")
        for key, dip in self.dipole_families.items():
            Bmax = dip["B_max"]
            print(f"Max magnetic field in dipole {key:21s} {Bmax:18.6e} T")
            dBmax = dip["dB_max"]*2*np.pi/self.dipole_period
            print(f"Max field slope in dipole {key:24s} {dBmax:18.6e} T/s")
        print(f"{'Repetition rate':50s} {self.rep_rate:18.6e} Hz")
        if self.target_ramping is not None:
            print(f"{'Target Ramping time':50s} {self.target_ramping*1000:18.6e} ms")
        else:
            print(f"{'Target Ramping time':64s} None")
        for key, dip in self.dipole_families.items():
            Binj = dip["B"](self.inj_phi)
            Bext = dip["B"](self.ext_phi)
            print(f"Magnetic field at injection  in dipole {key:11s} {Binj:18.6e} T")
            print(f"Magnetic field at extraction in dipole {key:11s} {Bext:18.6e} T")
        print(f"{'Number of arcs':56s} {self.nb_arc:d}")
        print(f"{'Number of cells per arc':56s} {self.nb_cell_arc:d}")
        print(f"{'Number of dipoles per cell':56s} {self.nb_dipole_cell:d}")
        for key, dip in self.dipole_families.items():
            ndip = dip["nb_half_cell"]
            print(f"Number of dipole {key} per half-cell {ndip:d}")
        print(f"{'Phase advance per arc cell':50s} {self.phase_advance:18.6f}")
        print(f"{'Number of RF insertions':56s} {self.nb_RF_section:d}")
        print(f"{'Length of RF insertions':50s} {self.insertion_length:18.6e} m")
        print(f"{'Quadrupole length':50s} {self.QP_length:18.6e} m")
        print(f"{'Normalized horizontal emittance':50s} {self.hor_norm_emittance*1e6:18.6e} um")
        print(f"{'Normalized vertical emittance':50s} {self.vert_norm_emittance*1e6:18.6e} um")
        print(f"{'Maximum energy spread':50s} {self.max_sigE:18.6e}")


    def print_params(self):
        print(f"     {'RCS general parameters':50s}")
        print(f"{'Circumference':50s} {self.length/1000:18.6e} km")
        print(f"{'Injection total energy':50s} {self.inj_E/1e9:18.6e} GeV")
        print(f"{'Extraction total energy':50s} {self.ext_E/1e9:18.6e} GeV")
        print(f"{'Injection gamma':50s} {self.inj_gamma:18.6e}")
        print(f"{'Extraction gamma':50s} {self.ext_gamma:18.6e}")
        print(f"{'Injection magnetic rigidity':50s} {self.inj_Brho:18.6e} T.m")
        print(f"{'Extraction magnetic rigidity':50s} {self.ext_Brho:18.6e} T.m")
        print(f"{'Mean revolution time':50s} {self.rev_time*1e6:18.6e} us")
        print(f"{'Mean revolution frequency':50s} {self.rev_freq*1e3:18.6e} kHz")
        print(f"{'Total Insertion length':50s} {self.tot_insertion_length*1e3:18.6e} km")
        print(f"{'Total RF length':50s} {self.RF_length_tot*1e3:18.6e} km")
        print(f"{'Arc length':50s} {self.arc_length*1e3:18.6e} km")
        print(f"{'Number of FODO cells':56s} {self.nb_cell_arcs:d}")
        print(f"{'Length of FODO cell':50s} {self.cell_length:18.6e} m")
        for key, dip in self.dipole_families.items():
            l_tot = dip["tot_length"]
            l = dip["length"]
            txt = f"Total length of the dipole {key:11s}"
            print(f"{txt:50s} {l_tot:18.6e} m")
            txt = f"Length of the dipole {key:11s}"
            print(f"{txt:50s} {l:18.6e} m")
        print(f"{'Distance between dipoles':50s} {self.dipole_spacing:18.6e} m")
        print(f"{'Ratio arc/total length':50s} {self.filling_ratio*100:18.6e} %")
        a = (self.NC_length_tot+self.SC_length_tot)/self.length*100
        print(f"{'Filling factor':50s} {a:18.6e} %")
        print(f"{'Max allowed voltage':50s} {self.max_voltage*1e-9:18.6e} GV")
        print(f"{'Max allowed gain per turn':50s} {self.max_energy_gain*1e-9:18.6e} GeV")
        print(f"{'Ramp time':50s} {self.ramp_time*1e3:18.6e} ms")
        print(f"{'Number of turns':50s} {self.nb_turns:18.6e}")
        print(f"{'Peak field slope':50s} {self.peak_B_slope:18.6e} T/s")
        print(f"{'Acceleration time in beam frame':50s} {self.acceleration_time*1e6:18.6e} us")
        print(f"{'Muon survival':50s} {self.muon_survival*100:18.6e} %")
        print(f"{'Max distance between injection/extraction obit':50s} {self.sagitta*1e3:18.6e} mm")

    def print_lattice(self):
        print(f"     {'Lattice properties':50s}")
        print(f"{'Transition energy':50s} {self.energy_tr*1e-9:18.6e} GeV")
        print(f"{'Momentum compaction':50s} {self.alfa:18.6e}")
        print(f"{'Quadrupole field strength at injection':50s} {self.inj_QP_gradient:18.6e} T/m")
        print(f"{'Quadrupole field strength at extraction':50s} {self.ext_QP_gradient:18.6e} T/m")
        print(f"{'Minimum beam stay clear':50s} {self.nb_sigma:18.6e} sigma")
        print(f"{'Maximum beta in FODO cell':50s} {self.max_betx:18.6e} m")
        print(f"{'Maximum dispersion in FODO cell':50s} {self.max_Dx:18.6e} m")
        print(f"{'Maximum hor. RMS size in FODO cell':50s} {self.max_sigx*1000:18.6e} mm")
        print(f"{'Maximum vert. RMS size in FODO cell':50s} {self.max_sigy*1000:18.6e} mm")
        print(f"{'Maximum hor. beam size in FODO cell':50s} {self.max_apertx*1000:18.6e} mm")
        print(f"{'Maximum vert. beam size in FODO cell':50s} {self.max_aperty*1000:18.6e} mm")

    def print_magnet(self):
        print(f"     {'NC Magnet properties':50s}")
        print(f"{'Max field':50s} {self.max_NC_field:18.6e} T")
        print(f"{'Reference induction':50s} {self.ref_B:18.6e} T")
        print(f"{'Minimum height':50s} {2*self.max_aperty*1000:18.6e} mm")
        width = (self.sagitta+2*self.max_apertx)
        print(f"{'Minimum width':50s} {1000*width:18.6e} mm")
        # print(f"{'Dipole length':50s} {self.NC_length:18.6e} m")
        print(f"{'Max field slope':50s} {self.peak_B_slope:18.6e} T/s")
        print(f"{'Ramping time':50s} {self.ramp_time*1000:18.6e} ms")
        print(f"{'Repetition rate':50s} {self.rep_rate:18.6e} Hz")
        print(f"{'Harmonic frequency':50s} {self.dipole_frequency:18.6e} Hz")
        print(f"{'Duty factor':50s} {self.duty_factor:18.6e}")
        print(f"{'Total NC magnetic energy':50s} {self.NC_energy_tot*1e-6:18.6e} MJ")
        print(f"{'Total NC peak power':50s} {self.peak_NC_power*1e-9:18.6e} GW")
        print(f"{'Total core loss':50s} {self.core_loss_tot/1000:18.6e} kW")
        print(f"{'Hysteretic loss coefficient':50s} {self.dipole_histeresis*1000:18.6e} mJ/kg")
        print(f"{'Eddy current coefficient':50s} {self.dipole_eddy*1000:18.6e} mJ.s/kg")
        print(f"{'Core loss @ Bref':50s} {self.core_loss_ref:18.6e} W/kg")
        print(f"{'Core loss @ Bmax':50s} {self.core_loss_max:18.6e} W/kg")
        print(f"{'Average core loss':50s} {self.core_loss_ave:18.6e} W/kg")
        print(f"{'Core cross-section':50s} {self.core_section:18.6e} m3/m")
        print(f"{'Core mass per meter':50s} {self.core_mass:18.6e} kg/m")
        print(f"{'Core loss per meter':50s} {self.core_loss_meter:18.6e} W/m")
        print(f"{'Total length NC magnet':50s} {self.NC_length_tot/1000:18.6e} km")
        print(f"{'Total core loss':50s} {self.core_loss_tot/1000:18.6e} kW")

    def print_RF(self):        
        print(f"     {'RF properties':50s}")
        print(f"{'Initial RMS long emittance':50s} {self.inj_long_rms_emittance:18.6e}")
        print(f"{'RF wavelength':50s} {self.RF_wavelength:18.6e} m")
        print(f"{'RF harmonic':50s} {self.RF_harmonic:18.6e}")
        print(f"{'Peak gain per turn':50s} {self.energy_gain*1e-9:18.6e} GeV")
        print(f"{'Peak voltage':50s} {self.tot_RF_voltage*1e-9:18.6e} GeV")
        print(f"{'Maximum guess gain per turn':50s} {self.energy_gain_guess*1e-9:18.6e} GeV")
        print(f"{'Maximum guess voltage':50s} {self.voltage_guess*1e-9:18.6e} GeV")
        print(f"{'RF guess phase':50s} {self.RF_phase_guess:18.6e} deg")
        print(f"{'RF guess energy acceptance':50s} { self.energy_acceptance:18.6e}")        
        print(f"{'Injection synchrotron tune':50s} { self.inj_Qs:18.6e}")        
        print(f"{'Extraction synchrotron tune':50s} { self.ext_Qs:18.6e}")        
        
    def print_all(self):
        self.print_input()
        print()
        self.print_params()
        print()
        self.print_lattice()
        print()
        self.print_magnet()
        print()
        self.print_RF()

    def plot_B(self, ax=None, bounds=None, dipoles=None, label="",
               time=True):
        if dipoles is None:
            dipoles = sorted(list(self.dipole_families.keys()))
        if ax is None:
            fig, ax = plt.subplots()
        if bounds is None:
            bounds = [-np.pi, np.pi, None, None]            
        elif bounds[0] is None:
            bounds[0] = -np.pi
        elif bounds[1] is None:
            bounds[1] = np.pi        
        if time:
            xlabel = "Time [ms]"
            fac = self.dipole_period/2/np.pi*1e3
        else:
            xlabel = "Dipole pulsation [deg]"
            fac = 180./np.pi            
        tab_phi = np.linspace(bounds[0], bounds[1], 101)
        for key in dipoles:
            dip = self.dipole_families[key]["B"]
            tab_B = np.array([dip(phi) for phi in tab_phi])
            ax.plot(tab_phi*fac, tab_B,
                    label=f"{label} {key}")
        ax.axvline(self.inj_phi*fac, ls="--", color="black")
        ax.axvline(self.ext_phi*fac, ls="--", color="black")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Dipole field [T]")
        ax.set_xlim(bounds[0]*fac, bounds[1]*fac)
        ax.set_ylim(bounds[2], bounds[3])
        
    def plot_gamma(self, ax=None, bounds=None, label=None, time=True):
        if ax is None:
            fig, ax = plt.subplots()
        if bounds is None:
            bounds = [self.inj_phi, self.ext_phi, None, None]            
        elif bounds[0] is None:
            bounds[0] = self.inj_phi
        elif bounds[1] is None:
            bounds[1] = self.ext_phi        
        if time:
            xlabel = "Time [ms]"
            fac = self.dipole_period/2/np.pi*1e3
        else:
            xlabel = "Dipole pulsation [deg]"
            fac = 180./np.pi            
        tab_phi = np.linspace(bounds[0], bounds[1], 101)
        tab_gamma = np.array([self.func_gamma(phi) for phi in tab_phi])
        if label is None:
            ax.plot(tab_phi*fac, tab_gamma*self.muon_mass*1e-9)
        else:
            ax.plot(tab_phi*fac, tab_gamma*self.muon_mass*1e-9, label=label)
        ax.axvline(self.inj_phi*fac, ls="--", color="black")
        ax.axvline(self.ext_phi*fac, ls="--", color="black")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Beam energy [GeV]")
        ax.set_xlim(bounds[0]*fac,bounds[1]*fac)
        ax.set_ylim(bounds[2], bounds[3])

    def plot_phis(self, ax=None, bounds=None, label=None, time=True):
        if ax is None:
            fig, ax = plt.subplots()
        if bounds is None:
            bounds = [self.inj_phi, self.ext_phi, None, None]            
        elif bounds[0] is None:
            bounds[0] = self.inj_phi
        elif bounds[1] is None:
            bounds[1] = self.ext_phi        
        if time:
            xlabel = "Time [ms]"
            fac = self.dipole_period/2/np.pi*1e3
        else:
            xlabel = "Dipole pulsation [deg]"
            fac = 180./np.pi            
        tab_phi = np.linspace(bounds[0], bounds[1], 101)
        tab_phis = np.array([self.func_phis(phi) for phi in tab_phi])
        if label is None:
            ax.plot(tab_phi*fac, tab_phis*180/np.pi)
        else:
            ax.plot(tab_phi*fac, tab_phis*180/np.pi, label=label)
        ax.axhline(self.RF_phase*180/np.pi, ls="--", color="b")
        ax.axvline(self.inj_phi*fac, ls="--", color="black")
        ax.axvline(self.ext_phi*fac, ls="--", color="black")
        ax.set_xlim(bounds[0]*fac, bounds[1]*fac)
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Synchronous phase [deg]")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('file_input', nargs='?', type=str, default=None,
                        help="the input file")
   
    for key, val in PulsedSynch.default_values.items():
        parser.add_argument("--"+key, type=type(val))
     
    args = parser.parse_args()
    kw = dict()
    for key, val in vars(args).items():
        if val is not None: kw[key] = val
    print(kw)
    RCS = PulsedSynch(**kw)
    RCS.print_all()
    exit
