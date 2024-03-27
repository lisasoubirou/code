# -*- coding: utf-8 -*-
"""
This class aims at computing geometric parameters of the RCS from basic parameters of the 
acceleration chain.
The following parameters can be chosen: 
    nb_arc : # of arcs per RCS
    nb_RF_section :  # of insertion containing RF
    nb_cell_arc : # of cells per arc => if non physical, will be re-adjusted 
    pattern : dipole pattern of a half-cell (example: ["BSC","BNC","BSC"])
    filling_ratio : filling ratio of the arcs (remaining is for insertion) 
    filling_ratio_RF : filling ratio of the RF in the insertion
    LSSS : length of the straight section in between the cells, for quadrupole
    dipole_spacing: length between the dipole of a same half-cell
    
@author: Lisa Soubirou
"""
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import warnings
import scipy.constants as cons
from scipy.optimize import minimize, minimize_scalar
from pathlib import Path

class Geometry:
    default_values = {
    "C": 5990,
    "E_inj": 	63e9,
    "E_ext": 	300e9,
    "t_inj": 0,
    "t_ext": 1,
    "LSSS": 2.5,
    "dipole_spacing": 0.3,  
    "nb_cell_arc": 12, #nc
    "nb_RF_section": 6,
    "nb_arc": 6,
    "dipoles":
        {"B1": 10., 
         "B2": (1.5, "linear")
         },
    "pattern": ["B1", "B2", "B1"],
    "filling_ratio": 0.85,
    "filling_ratio_RF": 0.4,
    "RF_freq": 1300011000.0
    }
        
    def __init__(self, file_input, **kw): 
        #Default parameters
        kw_def = dict(Geometry.default_values)
        for attr, val in kw_def.items():
            setattr(self, attr, val)
        #Input file
        file_input = np.array(file_input, ndmin=1)
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
        #Parameters from dico
        for attr, val in kw.items():
            setattr(self, attr, val)
        
        self.init_params()
        self.dipole_parameters()
        self.calc_geo()
        self.init_geo()    
        self.z_dipole()

    def init_params(self):
        """ Initialisation of basic geometric parameters
        """         
        self.muon_mass = cons.physical_constants[
            "muon mass energy equivalent in MeV"][0]*1e6
        self.muon_lifetime = 2.1969811e-6
        self.inj_gamma = self.E_inj/self.muon_mass
        self.ext_gamma = self.E_ext/self.muon_mass
        self.inj_Brho = np.sqrt(self.inj_gamma**2-1)*self.muon_mass/cons.c  #Magnetic rigidity at inj
        self.ext_Brho = np.sqrt(self.ext_gamma**2-1)*self.muon_mass/cons.c  #Magnetic rigidity at ext
        self.rev_time = self.C/cons.c
        self.rev_freq = 1./self.rev_time        
        self.tot_arc_length = self.filling_ratio*self.C
        self.tot_insertion_length = self.C-self.tot_arc_length
        self.arc_length = self.tot_arc_length/self.nb_arc
        self.insertion_length = self.tot_insertion_length/self.nb_arc
        self.RF_length_tot = self.filling_ratio_RF*self.insertion_length*self.nb_RF_section
        self.nb_cell_arcs = self.nb_cell_arc*self.nb_arc  #Total number of cells in RCS
        self.cell_length = self.arc_length/self.nb_cell_arc  #Length of 1 cell
        self.cell_angle = 2*np.pi/self.nb_cell_arcs  #Angle of 1 cell
        
    def dipole_parameters(self):
        """ Associate to each type of dipole its expression of B(t) and create a dico dipole_families
        containing parameters that will be calculated in calc_geo function. 
        """
        list_key = sorted(list(self.dipoles.keys()))
        def funcB(i):
            """Function to attribute a magnetic field to each type of dipole from the given pattern. 
            Information from the type of magnetic field are given in the dico dipoles.
            Ex :"dipoles":{"B1": 10., "B2": (1.5, "linear")} 
            If a float is given ("B1"): B=cst=10T
            If a list is given ("B2")
                "linear":
                    len(val) == 2: (example "B2": (1.5, "linear")) 
                    triangle function from -1.5 to 1.5 T, to be taken from [-pi/2,pi/2] for linear ramp 
                    len(val) == 3 : (example "B2": (-1.5, 1, "linear")) 
                    triangle function from -1.5 to 1 T
                else: (example "BNC": [0.0, 1.8, -0.38197])
                    B(t) defined with harmonics
            If a string is given: use of analytical expression
            """           
            key = list_key[i]
            val = self.dipoles[key]
            if isinstance(val, (float, int)): #Case B=cst (typically for SC)
                return lambda x: val
            elif isinstance(val, (tuple, list, np.ndarray)): #NC Dipole B(t)
                if val[-1] == "linear":  
                    if len(val) == 2:  #B(t) (NC): Triangle, phase to be taken [-pi/2,pi/2]
                        Bmax = val[0]
                        return lambda x: (1-2*np.abs(np.mod((-np.pi/2 + np.pi*x)+np.pi/2, 2*np.pi)-np.pi)/np.pi)*Bmax 
                    elif len(val) == 3:
                        mean = 0.5*(val[0]+val[1])
                        diff = np.abs(0.5*(val[1]-val[0]))
                        return lambda x: (2.*np.abs(np.mod((-np.pi/2 + np.pi*x)-np.pi/2, 2*np.pi))/np.pi-1.)*diff+mean
                    else:
                        raise ValueError(f"The dipole family {key} is not valid !")
                else: #B(t) NC: with harmonics
                    tab = np.array(val)
                    return lambda x: np.sum([v*np.sin(i*(-np.pi/2 + np.pi*x)) for i,v in enumerate(tab)])
            elif isinstance(val, str):  #B(t) (NC), analytical expression in a string
                return eval(val) #Function of time
            else:
                raise ValueError(f"The dipole family {key} is not valid !")
        #Create dico of dipole types with associated magnetic field, will be completed in calc_geo
        self.dipole_families = dict()
        for i_key, key in enumerate(list_key):
            self.dipole_families[key] = dict() 
            self.dipole_families[key]["B"] = funcB(i_key)    
        self.n_dipole_families = len(self.dipole_families)
        #Check all dipoles in pattern are defined
        for key in self.pattern:
            if key not in self.dipole_families.keys():
                raise ValueError(f"{key} is not defined")
         
    def calc_geo(self):
        """Calculates sequences list_a and list_b (see A. CHANCE note) that allows calculation of geometric parameters.
        Calculated parameters: 
        dipole_families[key] attributes:
            B : Args : x(float) :time
                Returns: Magnetic field
            length_half_cell : Dipole length in a half cell
            tot_length : Total dipole length in RCS
            nb_half_cell : nb dipole in half-cell
            nb_tot :  total nb of dipoles in RCS
            length : length of 1 dipole
            BL : Args : x(float) :time
                 Returns: B(x)*length on 1 half-cell
        e0: initial angle of half-cell
        L0: Dipole length in a half cell (all dipoles included)
        dipole_length_tot : Total dipole length in RCS (all dipoles included)
        func_Brho : Args: x (float): time
                    Returns: Magnetic rigidity
        nc_best : optimised # of cells per arc, if nb_cell_arc > nc_best : nb_cell_arc = nc_best
        QP_dipole_spacing : spacing between dipole and quadrupole
        """        
        self.e0 = np.sin(self.cell_angle/4)
        #Isomagnetic case, only 1 family of dipole (NC)
        if self.n_dipole_families == 1:  
            key = list(self.dipole_families.keys())[0]
            dip = self.dipole_families[key]
            Bext = dip["B"](self.t_ext)
            self.L0 = self.ext_Brho/Bext*2*self.e0 #Dipole length in a half cell
            self.dipole_length_tot = 2*self.nb_cell_arcs*self.L0 #Total dipole length in RCS
            dip["length_half_cell"] = self.L0
            dip["tot_length"] = self.dipole_length_tot
            dip["nb_half_cell"] = len(self.pattern)
            dip["nb_tot"] = dip["nb_half_cell"]*2*self.nb_cell_arcs
            dip["length"] = self.dipole_length_tot/dip["nb_tot"]
            dip["BL"] = lambda x: dip["length_half_cell"]*dip["B"](x)
            self.list_a = np.ones(dip["nb_half_cell"])/dip["nb_half_cell"]
            self.list_b = lambda x: np.ones(len(self.pattern))
            self.func_Brho = lambda x: self.L0*dip["B"](x)/2./self.e0 #Magnetic rigidity
        #More than 1 dipole family    
        else: 
            A = np.ones((2, self.n_dipole_families))
            list_key = sorted(self.dipole_families.keys())
            list_dip = [self.dipole_families[key] for key in list_key]
            A[0] = [dip["B"](self.t_inj)/2./self.e0 for dip in list_dip] 
            A[1] = [dip["B"](self.t_ext)/2./self.e0 for dip in list_dip]
            B = np.ones(2)
            B[0] = self.inj_Brho
            B[1] = self.ext_Brho
            if np.abs(np.linalg.det(A[:,:2])) > 1e-6 and self.n_dipole_families == 2:
                invA = np.linalg.inv(A[:,:2])
                Lsol = np.dot(invA, B)
            else:
                list_phi = np.linspace(self.t_inj, self.t_ext, 101)[1:-1]
                C = np.ones((len(list_phi), self.n_dipole_families))
                for i, phi in enumerate(list_phi):
                    C[i] = [dip["B"](phi)/2./self.e0 for dip in list_dip]
                D = self.inj_Brho+(self.ext_Brho-self.inj_Brho)*(list_phi-self.t_inj)/(self.t_ext-self.t_inj)
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
            self.L0 = np.sum(Lsol) #Dipole length in a half cell
            self.dipole_length_tot = 2*self.nb_cell_arcs*self.L0 #Total dipole length in RCS
            pattern = list(self.pattern)
            for i, dip in enumerate(list_dip):
                dip["length_half_cell"] = Lsol[i]
                dip["tot_length"] = 2*self.nb_cell_arcs*Lsol[i]
                dip["nb_half_cell"] = pattern.count(list_key[i])
                dip["length"] = Lsol[i]/dip["nb_half_cell"]
                dip["nb_tot"] = dip["nb_half_cell"]*2*self.nb_cell_arcs
                dip["BL"] = lambda x: dip["length_half_cell"]*dip["B"](x)
            self.func_Brho = lambda x: np.sum([l*dip["B"](x) for l, dip in zip(Lsol, list_dip)])/2./self.e0 #Magnetic rigidity
            self.list_a = np.ones(len(self.pattern))
            for i, key in enumerate(self.pattern):
                dip = self.dipole_families[key]
                self.list_a[i] = dip["length_half_cell"]/self.L0/dip["nb_half_cell"]
            self.list_ba = lambda x: np.array([
                self.L0*self.dipole_families[key]["B"](x)/self.func_Brho(x)/2/self.e0 for key in (self.pattern)])
            self.list_b = lambda x:np.where(np.abs(self.list_ba(x))<1e-8, 1e-8, self.list_ba(x)) #To find a solution when Bnc=0
        
        if self.dipole_spacing is not None:
            self.QP_dipole_spacing = ( #Spacing between quad and dipole
                self.cell_length/2-self.L0-(len(self.pattern)-1)*self.dipole_spacing-self.LSSS)/2.
            self.nc_best = int((self.arc_length-self.dipole_length_tot/self.nb_arc)/(
                (len(self.pattern)-1)*self.dipole_spacing+self.LSSS)/2) #Optimised nb_cell_arc
        else:
            raise KeyError("dipole_spacing is not defined")
        
        # Number of cells per arc may have to be optimised, not to get negative length of QP_dipole_spacing 
        if self.nb_cell_arc > self.nc_best:
            warnings.warn(f'Number of cells per arc was optimised from {self.nb_cell_arc} to {self.nc_best}', UserWarning)
            self.nb_cell_arc = self.nc_best
            self.init_params() #Recalculate parameters after optimisation 
            self.dipole_parameters()
            self.calc_geo()
    
    def init_geo(self): 
        """ Basic geometric paramaters
        """              
        self.nd = len(self.list_a) # Nb of dipole per half_cell
        self.h = 2*self.e0/self.L0 # Reference bending [1/m]
        self.rho = 1/self.h #Reference radius [m]
        self.Ln = self.L0*self.list_a #Total length of dipoles [m]
        self.nsub=10  #Number of intermediary points in dipoles to compute trajectory
        
    def theta (self,x): 
        """ Bending angle of dipoles [rad]
        Args: x (float): time
        Returns: list 
        """
        eps=self.epsilon(x)
        return -eps[1:]+eps[:-1]

    def L_dd_path (self,x): 
        """ Path length in the drift between the dipoles [m]
        Args: x (float): time
        Returns: list
        """        
        return self.dipole_spacing/np.cos(self.epsilon(x)[0: len(self.pattern)])
    
    def L_dip (self,x): 
        """ Chord length of trajectory in dipoles [m]
        Args: x (float): time
        Returns: list
        """
        ye=self.y_end(x)
        yb=self.y_begin(x)
        return [np.sqrt((self.dipole_families[key]["length"])**2 + (ye[ii]-yb[ii])**2) for ii,key in enumerate(self.pattern)]

    def L_dip_path (self,x): 
        """ Path length in dipoles [m]
        Args: x (float): time
        Returns: list
        """
        ee = self.epsilon(x)
        e1 = ee[1:]
        e0 = ee[:-1]
        ang = e0-e1
        return(ang/self.hn(x))   

    def L_qp_dip_path (self,x):
        """ Path length in the drift between a quad and a dipole
        Args: x (float): time
        Returns: float
        """
        ee = self.epsilon(x)
        return(self.QP_dipole_spacing/np.cos(ee[0]))
        
    def dz_(self, x):
        """ Deviation of position in dipole. Function defined to compute z_dipole (see below). 
        Args: x (float): time
        Returns: list
        """
        ex = self.epsilon(x)
        bb = self.list_b(x)
        return self.rho * np.append(
            [0], 
            np.cumsum((np.cos(ex[1:]) - np.cos(ex[:-1])) / bb))

    def dld_(self, x): 
        """ Deviation of position in drift. Function defined to compute z_dipole (see below). 
        Args: x (float): time
        Returns: list
        """                
        ex = self.epsilon(x)
        return self.dipole_spacing * np.append(
            [0], 
            np.cumsum(np.tan(ex[1:-1])))
    
    def z_dipole(self): 
        """ Calculates coordinates of points in complex plane:
            z_begin : at entry of dipoles 
            z_end : at exit of dipoles 
        """        
        cuml = np.append([0], np.cumsum(self.Ln))
        cumld = np.arange(self.nd)*self.dipole_spacing
        dz = self.dz_
        dld = self.dld_
        self.z_begin = lambda x: [c1 + c2 + 1j*(z + d) for c1, c2, z, d in zip(cuml[:-1],cumld,dz(x)[:-1],dld(x))]
        self.z_end = lambda x: [c1 + c2 + 1j*(z+d) for c1, c2, z, d in zip(cuml[1:],cumld,dz(x)[1:],dld(x))] 
        
    def hn (self,x): 
        """Bending of dipoles [1/m]
        Args: x (float): time
        Returns: list
        """
        return self.list_b(x) * self.h
    
    def sn (self,x): 
        """ Intermediate sequence sn_i to compute sequence phi_i
        Args: x (float): time
        Returns: list 
        """
        return np.append([0], np.cumsum(self.list_a*self.list_b(x)))
    
    def phi (self,x): 
        """ Sequence of phi_i=sin(epsilon_i) (see function epsilon below)
        Args: x (float): time
        Returns: list
        """
        return self.e0*(1-2*self.sn(x))
    
    def epsilon (self,x): 
        """ Edge angles at the entrance of dipoles [rad]
        Args: x (float): time 
        Returns: list
        """
        return np.arcsin(self.phi(x))
    
    def zn(self,x):
        """ Coordinates of intermediary points inside the dipoles. 
        The number of points is defined by nsub.
        Args: x (float): time
        Returns: list
        """
        eps=self.epsilon(x)
        z_b=self.z_begin(x)
        bb=self.list_b(x)
        z = np.zeros((self.nsub + 1) * self.nd + 2, dtype=complex)
        z[1] = (0.5*self.LSSS+self.QP_dipole_spacing)/np.cos(eps[0])*np.exp(1j*eps[0])
        # z[1] = (self.QP_dipole_spacing)/np.cos(eps[0])*np.exp(1j*eps[0])
        dc =  np.linspace(0., 1., self.nsub+1)
        for n in np.arange(self.nd):
            e0 = eps[n]
            e1 = eps[n+1]
            e = e0 + dc*(e1-e0)
            cc = (np.cos(e)-np.cos(e0))/bb[n]
            i0 = 1 + n*(self.nsub+1)
            z0 = z[1]+z_b[n]
            z[i0:i0+self.nsub+1] = z0+self.Ln[n]*dc+1j*self.rho*cc
        z[-1] = z[-2] + (0.5*self.LSSS+self.QP_dipole_spacing)/np.cos(eps[-1])*np.exp(1j*eps[-1])
        # z[-1] = z[-2] + (self.QP_dipole_spacing)/np.cos(eps[-1])*np.exp(1j*eps[-1])
        return(z)

    def y_begin(self, x): 
        """ y coordinate at entry of each dipole [m]
        Args: x (float): time
        Returns: list
        """
        return np.imag(self.z_begin(x))

    def y_end(self, x):  
        """ y coordinate at exit of each dipole [m]
        Args: x (float): time
        Returns: list
        """
        return np.imag(self.z_end(x))

    def extrema(self, x):
        """Calculates the extrema of trajectories y_min, y_max and width 
        covered by the trajectory in each dipole.
        Args: x (float): time
        Returns: 
            y_min (list) : min y coordinate for the trajectory in each dipole
            y_max (list) : max y coordinate for the trajectory in each dipole
            width (list) : width = y_max-y_min of trajectory in each dipole
            max_apert_noshift (float) : np.max(y_max)-np.min(y_min) : max aperture if 
              no shift are made between the dipoles, "they are all aligned" 
            max_apert (float): np.max(width[1:-1]), max aperture if the dipoles are positioned to
            minimize overall width. 
        """                     
        y_min = np.zeros(self.nd)
        y_max = np.zeros(self.nd)
        width = np.zeros(self.nd)
        list_e = self.epsilon(x)
        list_z_beg = self.y_begin(x)
        list_z_end = self.y_end(x)
        lb=self.list_b(x)
        for n in np.arange(self.nd):  
            e0 = list_e[n]
            e1 = list_e[n+1]
            y0 = list_z_beg[n]
            y1 = list_z_end[n]
            bb = lb[n]
            if e0>0:
                if e1>0:
                    y_min[n] = y0
                    y_max[n] = y1
                else:
                    y_min[n] = np.min([y0, y1])
                    y_max[n] = y0+self.rho*(1-np.cos(e0))/bb
            elif e1>0:
                y_min[n] = y0+self.rho*(1-np.cos(e0))/bb
                y_max[n] = np.max([y0, y1])
            else:
                y_min[n] = y1
                y_max[n] = y0
        width = y_max - y_min
        max_apert_noshift = np.max(y_max)-np.min(y_min)
        max_apert = np.max(width[1:-1])
        return(y_min, y_max,width,max_apert_noshift,max_apert)

    def path_length(self,x): 
        """Path length in ONE HALF-CELL, which pattern is given as input. 
        Length insertion LSSS for quadrupoles is included.
        Args: x (float): time
        Returns: float
        """        
        ee = self.epsilon(x)
        e1 = ee[1:]
        e0 = ee[:-1]
        ang = e0-e1
        path_length = np.sum(ang/self.hn(x))+self.dipole_spacing*np.sum(1/np.cos(e1[:-1]))+(self.LSSS+2*self.QP_dipole_spacing)/np.cos(e0[0])
        return(path_length)
        
    def path_length_tot(self,x):   
        """ Total path length in the nb_arc of the RCS. 
        Path length in RF insertion is not included, it undergoes no variation during ramping. 
        Args: x (float): time
        Returns: float
        """
        return 2*self.nb_cell_arc*self.nb_arc*self.path_length(x)
    
    def set_path_length_min(self):
        """ Find the minimum path length, stored in self.path_length_min. 
        Returns: results of minimization
            self.set_path_length_min().fun : minimum path length calculated on arcs
            self.set_path_length_min().x : time of minimum
        """
        result = minimize_scalar(self.path_length_tot, method="bounded",
                                 bounds=(self.t_inj,self.t_ext))
        self.path_length_min = result.fun
        return (result)
        
def plot_traj(RCS, t_traj):
    """ Function that plots the trajectory for a given geometry and time.
    Args:
        RCS (Geometry object): result from Geoemtry class
        t_traj (list): time for each trajectory
    """    
    plt.figure()
    for i in t_traj:
        plt.plot(np.real(RCS.zn(i)), np.imag(RCS.zn(i)))
    zmin = np.min(np.imag(RCS.zn(1)))
    zmax = np.max(np.imag(RCS.zn(1)))
    dz = zmax - zmin
    zloc = (0.5*RCS.LSSS+RCS.QP_dipole_spacing)
    iz = 1
    for z1, z2 in zip(RCS.z_begin(1), RCS.z_end(1)):
        x1, x2 = np.real(z1), np.real(z2)
        plt.axvline(zloc + x1,ls="--",color="k")
        plt.axvline(zloc + x2,ls="--",color="k")
        plt.text(zloc + 0.5*(x1+x2), zmin + dz*1.2, f"D{iz:d}", horizontalalignment='center', verticalalignment='center')
        iz += 1
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    #plt.legend()
    plt.show()
        
if __name__ == "__main__":
    file_input='/mnt/c/muco/class_geometry/para_RCS_ME.txt'  
    RCS = Geometry(file_input)
    t_print=0.5
    y_min, y_max,width,max_apert_noshift,max_apert=RCS.extrema(t_print)
    print('Results for t=',t_print)
    print('Theta',RCS.theta(t_print))
    print('Width', width)
    print('Max apert no shift',max_apert_noshift)
    print('Max width from t_inj to t_ext',np.max(RCS.extrema(RCS.t_ext)[1])-np.min(RCS.extrema(RCS.t_inj)[0]))
    print('Path length',RCS.path_length(t_print))
    print('Path length tot',RCS.path_length_tot(t_print))
    
    # Plot n_traj between injection and extraction
    n_traj=8
    t_traj=np.linspace(RCS.t_inj,RCS.t_ext,n_traj)
    plot_traj(RCS,t_traj)
    
    #Plottings
    n_plot=15
    t_np=np.linspace(RCS.t_inj,RCS.t_ext,n_plot)
    #Bfield (NC)
    plt.figure()
    plt.plot(t_np, [RCS.dipole_families['BNC']["B"](t) for t in t_np])
    plt.xlabel('phase [rad]')
    plt.ylabel('B [T]')
    plt.show()
    #Total Path length
    arr_path_length = np.array([RCS.path_length_tot(t) for t in t_np])
    plt.figure()
    plt.plot( t_np, arr_path_length)
    plt.xlabel('phase [rad]')
    plt.ylabel('total path length [m]')
    plt.show()
    #Total Path length diff
    RCS.set_path_length_min()
    plt.figure()
    plt.plot( t_np, arr_path_length-RCS.path_length_min)
    plt.xlabel('phase [rad]')
    plt.ylabel('total path length difference [m]')
    plt.show()
    #DeltaX
    plt.figure()
    plt.plot(t_np,[RCS.extrema(t)[3] for t in t_np]) 
    plt.xlabel('phase [rad]')
    plt.ylabel('Max apert (no shift) [m]')
    plt.show()
    