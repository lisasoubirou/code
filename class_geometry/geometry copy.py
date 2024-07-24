# -*- coding: utf-8 -*-
"""
This class aims at computing geometric parameters of the RCS from basic parameters of the 
acceleration chain.

The following parameters can be chosen: 
    C : Circumference of the RCS
    E_inj : Injection energy
    E_ext : Extraction energy
    LSSS : Length of the straight section in between the cells, for QP/SXT
    dipole_spacing: Length between the dipole of a same half-cell
    nb_arc : Number of arcs in the RCS
    nb_RF_section :  Number of insertion in the RCS
    nb_cell_arc : Number of cells per arc => if non physical, will be re-adjusted 
    dipoles: Describe the dipole families and their magnetic properties
    pattern : Dipole pattern of a half-cell (example: ["BSC","BNC","BSC"])
    filling_ratio : filling ratio of the arcs (remaining is for insertion) 
    filling_ratio_RF : filling ratio of the RF in the Rf insertion
Those parameters can be modified, the geometry will be recomputed accordingly. 

@author: Lisa Soubirou
"""
import yaml
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import warnings
import scipy.constants as cons
from scipy.optimize import minimize, minimize_scalar

def load_file(file_input):
    """Load file for different format"""
    if file_input is None:
        kw_file={}
    else: 
        if isinstance(file_input, dict):
            kw_file = file_input
        elif Path(file_input).exists():
            try:
                with open(file_input, "r") as f:
                    kw_file = json.load(f)
                    print(f"Load json file {file_input}")
            except:
                try:
                    with open(file_input, "r") as f:
                        kw_file = yaml.safe_load(f)
                        print(f"Load yaml file {file_input}")
                except:
                    raise yaml.parser.ParserError(f"{file_input} is an invalid file")    
        else:
            raise FileExistsError(f"{file_input} does not exist")
    return(kw_file)

class Geometry:
    default_values = {  #Define some default parameters (from RCS2)
    "C": 5990,
    "E_inj": 	313.830e9,
    "E_ext": 	750e9,
    "LSSS": 2.5,
    "dipole_spacing": 0.3,  
    "nb_cell_arc": 8, 
    "nb_RF_section": 24,
    "nb_arc": 26,
    "dipoles":
        {"BSC": 10., 
         "BNC": (1.8, "linear")
         },
    "pattern": ["BSC", "BNC", "BSC"],
    "filling_ratio": 0.85,
    "filling_ratio_RF": 0.4,
    }

    def __init__(self, file_input=None, **kw_user): 
        kw_def = dict(Geometry.default_values)
        kw_file=load_file(file_input)
        kw_def.update(kw_file) 
        kw_def.update(kw_user)
        for attr, val in kw_def.items():
            if attr in dict(Geometry.default_values):
                setattr(self, '_'+attr, val)
            else:
                raise KeyError("{} is an invalid key value".format(attr))
        self.check_physics()
        self.muon_mass = cons.physical_constants[
            "muon mass energy equivalent in MeV"][0]*1e6
        self.muon_lifetime = 2.1969811e-6
        self.t_inj=0
        self.t_ext=1
        self.reinit_all()

    def check_physics(self):
        """Check if input parameters are physical"""
        if self._C <= 0:
            raise ValueError("Circumference must be > 0")
        if self._E_inj >= self.E_ext or self._E_inj <= 0:
             raise ValueError("Injection energy must be smaller than extraction energy or > 0")
        if self.E_ext <= self.E_inj or self.E_ext <= 0:
            raise ValueError("Extraction energy must be larger than injection energy or > 0")
        if self._nb_cell_arc <= 0:
             raise ValueError("Number of cells per arc must be at least 1")
        if self._nb_arc <= 0:
             raise ValueError("Number of arcs must be at least 1")
        if self._LSSS <= 0:
            raise ValueError("LSSS length must > 0")
        if self._dipole_spacing <= 0:
            raise ValueError("Dipole spacing must be > 0")
        if self._filling_ratio <= 0 or self._filling_ratio >= 1:
            raise ValueError("Filling ratio must be in the range [0,1]")
        if self._filling_ratio_RF <= 0 or self._filling_ratio_RF >= 1:
            raise ValueError("Filling ratio of RF must be in the range [0,1]")
        
    def reinit_all(self):
        self.dipole_parameters()
        self.calc_geo()
        self.z_dipole()

    def init_field_B(self,key):
            """Attribute a magnetic field to each type of dipole from the given pattern. 
            Magnetic properties are given in the dict dipoles. 
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
            val = self._dipoles[key]
            if isinstance(val, (float, int)): #Case B=cst (typically for SC)
                self.dipole_families[key]["B"] = lambda x: val
            elif isinstance(val, (tuple, list, np.ndarray)): #NC Dipole B(t)
                if val[-1] == "linear":  
                    if len(val) == 2:  #B(t) (NC): Triangle, symmetric from -B0/B0, phase to be taken between [-pi/2,pi/2]
                        Bmax = val[0]
                        self.dipole_families[key]["B"] = lambda x: (1-2*np.abs(np.mod((-np.pi/2 + np.pi*x)+np.pi/2, 2*np.pi)-np.pi)/np.pi)*Bmax 
                    elif len(val) == 3: #B(t) (NC): Triangle, from B1/B2, phase to be taken between [-pi/2,pi/2]
                        mean = 0.5*(val[0]+val[1])
                        diff = np.abs(0.5*(val[1]-val[0]))
                        self.dipole_families[key]["B"] = lambda x: (2.*np.abs(np.mod((-np.pi/2 + np.pi*x)-np.pi/2, 2*np.pi))/np.pi-1.)*diff+mean
                    else:
                        raise ValueError(f"The dipole family {key} is not valid !")
                else:  #B(t) NC: with harmonics
                    tab = np.array(val)
                    self.dipole_families[key]["B"] = lambda x: np.sum([v*np.sin(i*(-np.pi/2 + np.pi*x)) for i,v in enumerate(tab)])
            elif isinstance(val, str):  #B(t) (NC), analytical expression in a string
                self.dipole_families[key]["B"] = eval(val) 
            else:
                raise ValueError(f"The dipole family {key} is not valid !")

    def dipole_parameters(self):
        """ Associate an expression of B(t) to each type of dipole. 
        Create a dico dipole_families containing parameters that will be calculated in calc_geo function. 
        """
        list_key = sorted(list(self._dipoles.keys()))
        self.dipole_families = dict()
        for key in (list_key):
            self.dipole_families[key] = dict() 
            self.init_field_B(key)    
        self.n_dipole_families = len(self.dipole_families)
        #Check all dipoles in pattern are defined
        for key in self._pattern:
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
        """        
        self.e0 = np.sin(self.cell_angle/4)
        #Isomagnetic case, only 1 family of dipole (normal conducting RCS)
        if self.n_dipole_families == 1:  
            key = list(self.dipole_families.keys())[0]
            dip = self.dipole_families[key]
            Bext = dip["B"](self.t_ext)
            self.L0 = self.ext_Brho/Bext*2*self.e0 #Dipole length in a half cell
            self.dipole_length_tot = 2*self.nb_cell_rcs*self.L0 #Total dipole length in RCS
            dip["length_half_cell"] = self.L0
            dip["tot_length"] = self.dipole_length_tot
            dip["nb_half_cell"] = len(self._pattern)
            dip["nb_tot"] = dip["nb_half_cell"]*2*self.nb_cell_rcs
            dip["length"] = self.dipole_length_tot/dip["nb_tot"]
            dip["BL"] = lambda x: dip["length_half_cell"]*dip["B"](x)
            self.list_a = np.ones(dip["nb_half_cell"])/dip["nb_half_cell"]
            self.list_b = lambda x: np.ones(len(self._pattern))
            self.func_Brho = lambda x: self.L0*dip["B"](x)/2./self.e0 #Magnetic rigidity
        #More than 1 dipole family (hybrid with 2 families or more)    
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
            self.dipole_length_tot = 2*self.nb_cell_rcs*self.L0 #Total dipole length in RCS
            pattern = list(self._pattern)
            for i, dip in enumerate(list_dip):
                dip["length_half_cell"] = Lsol[i]
                dip["tot_length"] = 2*self.nb_cell_rcs*Lsol[i]
                dip["nb_half_cell"] = pattern.count(list_key[i])
                dip["length"] = Lsol[i]/dip["nb_half_cell"]
                dip["nb_tot"] = dip["nb_half_cell"]*2*self.nb_cell_rcs
                dip["BL"] = lambda x: dip["length_half_cell"]*dip["B"](x)
            self.func_Brho = lambda x: np.sum([l*dip["B"](x) for l, dip in zip(Lsol, list_dip)])/2./self.e0 #Magnetic rigidity
            self.list_a = np.ones(len(self._pattern))
            for i, key in enumerate(self._pattern):
                dip = self.dipole_families[key]
                self.list_a[i] = dip["length_half_cell"]/self.L0/dip["nb_half_cell"]
            self.list_ba = lambda x: np.array([
                self.L0*self.dipole_families[key]["B"](x)/self.func_Brho(x)/2/self.e0 for key in (self._pattern)])
            self.list_b = lambda x:np.where(np.abs(self.list_ba(x))<1e-8, 1e-8, self.list_ba(x)) #To find a solution when Bnc=0
        
        # self.QP_dipole_spacing = ( #Spacing between quad and dipole
        #     self.cell_length/2-self.L0-(len(self._pattern)-1)*self._dipole_spacing-self._LSSS)/2.
        self.nc_best = int((self.arc_length-self.dipole_length_tot/self._nb_arc)/(
            (len(self._pattern)+1)*self._dipole_spacing+self._LSSS)/2) #Optimised nb_cell_arc
        
        # Number of cells per arc may have to be optimised, or could get unphysical geometry
        if self._nb_cell_arc > self.nc_best:
            if self.nc_best <=0:
                raise ValueError("Optimum number of cells is negative: unphysical solution. Check input parameters.")
            warnings.warn(f'Number of cells per arc was optimised from {self._nb_cell_arc} to {self.nc_best}', UserWarning)
            self._nb_cell_arc = self.nc_best
            self.dipole_parameters()
            self.calc_geo()

    @property
    def C (self) -> float:
        """Circumference [m]"""
        return self._C
        
    @C.setter
    def C(self, value: float):
        if value <= 0:
            raise ValueError("Circumference must be > 0")
        self._C = value
        self.reinit_all()

    @property
    def E_inj(self) -> float:
        """Injection energy [eV]"""
        return self._E_inj
    
    @E_inj.setter
    def E_inj(self, value: float):
        if value >= self.E_ext or value <= 0:
             raise ValueError("Injection energy must be smaller than extraction energy or > 0")
        self._E_inj = value
        self.reinit_all()

    @property
    def E_ext(self) -> float:
        """Extraction energy [eV]"""
        return self._E_ext
    
    @E_ext.setter
    def E_ext(self, value: float):
        if value <= self.E_inj or value <= 0:
             raise ValueError("Extraction energy must be larger than injection energy or > 0")
        self._E_ext = value
        self.reinit_all()

    @property
    def nb_cell_arc(self) -> int:
        """Number of cells in 1 arc"""
        return self._nb_cell_arc
    
    @nb_cell_arc.setter
    def nb_cell_arc(self, value: int):
        if value <= 0:
             raise ValueError("Number of cells per arc must be at least 1")
        self._nb_cell_arc = value
        self.reinit_all()

    @property
    def nb_arc(self) -> int:
        """Number of arc in RCS"""
        return self._nb_arc
    
    @nb_arc.setter
    def nb_arc(self, value: int):
        if value <= 0:
             raise ValueError("Number of arcs must be at least 1")
        self._nb_arc = value
        self.reinit_all()

    @property
    def nb_RF_section(self) -> int:
        """Number of RF insertion in RCS"""
        return self._nb_RF_section
    
    @nb_RF_section.setter
    def nb_RF_section(self, value: int):
        if value <= 0:
             raise ValueError("Number of RF sections must be at least 1")
        self._nb_RF_section = value
        self.reinit_all()

    @property
    def LSSS(self) -> float:
        """Length of the straight section in between the cells, for qp/sxt"""
        return self._LSSS
    
    @LSSS.setter
    def LSSS(self, value: float):
        if value <= 0:
            raise ValueError("LSSS length must > 0")
        self._LSSS = value
        self.reinit_all()

    @property
    def dipole_spacing(self) -> float:
        """Length between the dipoles in a cell"""
        return self._dipole_spacing
    
    @dipole_spacing.setter
    def dipole_spacing(self, value: float):
        if value <= 0:
            raise ValueError("Dipole spacing must be > 0")
        self._dipole_spacing = value
        self.reinit_all()

    @property
    def filling_ratio(self) -> float:
        """Filling ratio of the arc length wrt C"""
        return self._filling_ratio
    
    @filling_ratio.setter
    def filling_ratio(self, value: float):
        if value <= 0 or value >= 1:
            raise ValueError("Filling ratio must be in the range [0,1]")
        self._filling_ratio = value
        self.reinit_all()

    @property
    def filling_ratio_RF(self) -> float:
        """Filling ratio of the RF in the RF section"""
        return self._filling_ratio_RF
    
    @filling_ratio_RF.setter
    def filling_ratio_RF(self, value: float):
        if value <= 0 or value >= 1:
            raise ValueError("Filling ratio of RF must be in the range [0,1]")
        self._filling_ratio_RF = value
        self.reinit_all()
    
    @property
    def dipoles(self) -> dict:
        """Describe the dipole families and their magnetic properties"""
        return self._dipoles
    
    @dipoles.setter
    def dipoles (self,value):
        self._dipoles = value
        self.reinit_all()
    
    @property
    def pattern(self) -> list:
        """Dipole pattern of a half-cell (example: ["BSC","BNC","BSC"])"""
        return self._pattern
    
    @pattern.setter
    def pattern(self, value: list):
        self._pattern = value
        self.reinit_all()

    @property
    def L_extra_rcs(self) -> float:
        """Extra length left for the rcs after computing the cell length"""
        return self.tot_arc_length-self.dipole_length_tot-2*self.nb_cell_rcs*(self.nd+1)*self.dipole_spacing-self.LSSS*2*self.nb_cell_rcs
    
    @property
    def L_extra_arc(self) -> float:
        """Extra length left for one arc after computing the cell length"""
        return self.L_extra_rcs/self.nb_arc

    @property
    def inj_gamma(self) -> float:
        """Injection gamma"""
        return self._E_inj/self.muon_mass
    
    @property
    def ext_gamma(self) -> float:
        """Extraction gamma"""
        return self._E_ext/self.muon_mass
    
    @property
    def inj_Brho(self) -> float:
        """Magnetic rigidity at injection [eV/c]"""
        return np.sqrt(self.inj_gamma**2-1)*self.muon_mass/cons.c
    
    @property
    def ext_Brho(self) -> float:
        """Magnetic rigidity at extraction [eV/c]"""
        return np.sqrt(self.ext_gamma**2-1)*self.muon_mass/cons.c
    
    @property
    def tot_arc_length(self) -> float:
        """Total length of arcs in RCS [m]"""
        return self._filling_ratio*self._C
    
    @property
    def tot_insertion_length(self) -> float:
        """Total length of insertion in RCS [m]"""
        return self._C-self.tot_arc_length
    
    @property
    def arc_length(self) -> float:
        """Length of 1 arc [m]"""
        return self.tot_arc_length/self._nb_arc
    
    @property
    def insertion_length(self) -> float:
        """Length of 1 insertion [m]"""
        return self.tot_insertion_length/self._nb_arc
    
    @property
    def RF_length_tot(self) -> float:
        """Total length dedicated to RF in RF insertion [m]"""
        return self._filling_ratio_RF*self.insertion_length*self._nb_RF_section
    
    @property
    def nb_cell_rcs(self) -> float:
        """Total number of cells in RCS"""
        return self._nb_cell_arc*self._nb_arc
    
    @property
    def cell_length(self) -> float:
        """Length of 1 cell [m]"""
        return self.arc_length/self._nb_cell_arc
    
    @property
    def cell_angle(self) -> float:
        """Angle of 1 cell"""
        return 2*np.pi/self.nb_cell_rcs
    
    @property
    def filling_ratio_dipole(self) -> float:
        """Filling ratio of dipole in the RCS: Ltot/C"""
        return self.dipole_length_tot/self._C

    @property
    def nd(self) -> int:
        """Number of dipoles per half_cell"""
        return len(self.list_a)
    
    @property
    def h(self) -> float:
        """Reference bending [1/m]"""
        return 2*self.e0/self.L0
    
    @property
    def rho(self) -> float:
        """Reference radius [m]"""
        return 1/self.h
    
    @property
    def Ln(self) -> float:
        """Total length of dipoles [m]"""
        return self.L0*self.list_a
    
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
        return self._dipole_spacing/np.cos(self.epsilon(x)[0: len(self._pattern)])
    
    def L_dip (self,x): 
        """ Chord length of trajectory in dipoles [m]
        Args: x (float): time
        Returns: list
        """
        ye=self.y_end(x)
        yb=self.y_begin(x)
        return [np.sqrt((self.dipole_families[key]["length"])**2 + (ye[ii]-yb[ii])**2) for ii,key in enumerate(self._pattern)]

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
        return(self._dipole_spacing/np.cos(ee[0]))
        
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
        return self._dipole_spacing * np.append(
            [0], 
            np.cumsum(np.tan(ex[1:-1])))
    
    def z_dipole(self): 
        """ Calculates coordinates of points in complex plane:
            z_begin : at entry of dipoles 
            z_end : at exit of dipoles 
        """        
        cuml = np.append([0], np.cumsum(self.Ln))
        cumld = np.arange(self.nd)*self._dipole_spacing
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
        self.nsub=10  #Number of intermediary points in dipoles to compute trajectory
        eps=self.epsilon(x)
        z_b=self.z_begin(x)
        bb=self.list_b(x)
        z = np.zeros((self.nsub + 1) * self.nd + 2, dtype=complex)
        z[1] = (0.5*self._LSSS+self._dipole_spacing)/np.cos(eps[0])*np.exp(1j*eps[0])
        # z[1] = (self._dipole_spacing)/np.cos(eps[0])*np.exp(1j*eps[0])
        dc =  np.linspace(0., 1., self.nsub+1)
        for n in np.arange(self.nd):
            e0 = eps[n]
            e1 = eps[n+1]
            e = e0 + dc*(e1-e0)
            cc = (np.cos(e)-np.cos(e0))/bb[n]
            i0 = 1 + n*(self.nsub+1)
            z0 = z[1]+z_b[n]
            z[i0:i0+self.nsub+1] = z0+self.Ln[n]*dc+1j*self.rho*cc
        z[-1] = z[-2] + (0.5*self._LSSS+self._dipole_spacing)/np.cos(eps[-1])*np.exp(1j*eps[-1])
        # z[-1] = z[-2] + (self._dipole_spacing)/np.cos(eps[-1])*np.exp(1j*eps[-1])
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
        path_length = np.sum(ang/self.hn(x))+self._dipole_spacing*np.sum(1/np.cos(e1[:-1]))+(self._LSSS+2*self._dipole_spacing)/np.cos(e0[0])
        return(path_length)
        
    def path_length_tot(self,x):   
        """ Total path length in the nb_arc of the RCS. 
        Path length in RF insertion is not included, as it undergoes no variation during ramping. 
        Args: x (float): time
        Returns: float
        """
        return 2*self._nb_cell_arc*self._nb_arc*self.path_length(x)
    
    @property
    def path_length_min(self):
        """Minimum path length for times between t_inj and t_ext"""
        result = minimize_scalar(self.path_length_tot, method="bounded",
                                 bounds=(self.t_inj,self.t_ext))
        return  result.fun
    
    @property
    def max_path_diff(self):
        """Maximum path length difference from t_inj to t_ext"""
        result = minimize_scalar(self.path_length_tot, method="bounded",
                                 bounds=(self.t_inj,self.t_ext))
        min_path=result.fun
        f= lambda x: -self.path_length_tot(x)
        result_max = minimize_scalar(f, method="bounded",
                                 bounds=(self.t_inj,self.t_ext))
        return -result_max.fun-min_path

    def plot_traj(self, t_traj=np.linspace(0,1,8)):
        """ Plots the trajectory for a given time between t_inj and t_ext.
        Args:
            t_traj (list): time for each trajectory
        """    
        plt.figure()
        for i in t_traj:
            plt.plot(np.real(self.zn(i)), np.imag(self.zn(i)))
        zmin = np.min(np.imag(self.zn(1)))
        zmax = np.max(np.imag(self.zn(1)))
        dz = zmax - zmin
        zloc = (0.5*self._LSSS+self._dipole_spacing)
        iz = 1
        for z1, z2 in zip(self.z_begin(1), self.z_end(1)):
            x1, x2 = np.real(z1), np.real(z2)
            plt.axvline(zloc + x1,ls="--",color="k")
            plt.axvline(zloc + x2,ls="--",color="k")
            plt.text(zloc + 0.5*(x1+x2), zmin + dz*1.2, f"D{iz:d}", horizontalalignment='center', verticalalignment='center')
            iz += 1
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.show()