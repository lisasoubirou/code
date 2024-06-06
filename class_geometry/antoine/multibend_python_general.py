# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:57:44 2022

@author: Achance
"""

from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

from matplotlib import rc

rc("figure", dpi=200)
rc("legend", columnspacing=1.2, handletextpad=0.4, 
   handlelength=2, labelspacing=0.4, borderaxespad=0.4)
rc("axes", labelsize=12)

class MultiBend:
    __default__ = {
        "Ld": 11.1*164*8*4,
        "nc": 10,
        "a": [0.25, 0.5, 0.25],
        "b": [1, 1, 1],
        "nsub": 10,
        "Ldd": 0.4,
        "Lqq": 0.15,
        "Lbq": 0.5,
        "LQP": 1.5,
        "LSX": 0.5,
        "LSSS": 2.5,
        "Larc": 11.1*164*8*4,
        "mux": np.pi/2,
        "pattern": "fodo_cell"}
    def __init__(self, **kw):
        self.generate(**kw)
            
    def generate(self, **kw):
        dic0 = dict(MultiBend.__default__)
        dic0.update(**kw)
        for key in MultiBend.__default__:
            setattr(self, key, dic0[key])
        self.a = np.array(self.a)
        assert self.a.ndim == 1, "The dimension of a is not equal to 1"        
        self.n = len(self.a)
        self.b = np.array(self.b, ndmin=2)
        assert self.b.ndim == 2, "The dimension of b is not equal to 2"
        assert self.a.ndim == 1, "The dimension of a is not equal to 1"        
        assert self.b.shape[1] == len(self.a), "The length of b is not equal to the one of a"
        assert np.abs(np.sum(self.a)-1) < 1e-8, "The sum of a is not equal to 1"
        for bb in self.b:
            assert np.abs(np.sum(self.a*bb)-1) < 1e-8, "The sum of a*b is not equal to 1"   
        self.b = np.where(np.abs(self.b) < 1e-6, 1e-6, self.b)
        self.nb = len(self.b)
        self.L0 = self.Ld/self.nc/2
        self.thetac = 2*np.pi/self.nc
        phi0 = np.sin(self.thetac/4)
        self.h = 2*phi0/self.L0
        self.rho = 1/self.h
        self.Ln = self.L0*self.a
        self.hn = self.b*self.h
        self.sn = np.array([np.append([0], np.cumsum(self.a*bb)) for bb in self.b])
        self.phi = phi0*(1-2*self.sn)
        self.epsilon = np.arcsin(self.phi)
        self.nd = len(self.a)
        self.make_traj()
        self.make_extrema()
        self.make_path_length()
        self.calc_momentum_compaction(self.mux)

    def make_traj(self, nsub=None):
        if nsub is None:
            nsub = self.nsub
        cuml = np.append([0], np.cumsum(self.Ln))
        cumld = np.arange(self.nd)*self.Ldd
        dz = self.rho*np.array(
            [np.append([0], np.cumsum((np.cos(e1[1:])-np.cos(e1[:-1]))/bb)) for bb, e1 in zip(
            self.b, self.epsilon)])
        dld = self.Ldd*np.array(
            [np.append([0], np.cumsum(np.tan(e1[1:-1]))) for e1 in self.epsilon])
        self.z_begin = cuml[:-1] + cumld + 1j*(dz[:,:-1]+dld)
        self.z_end = cuml[1:] + cumld + 1j*(dz[:,1:]+dld)
        self.zn = np.zeros((self.nb, (nsub+1)*self.nd+2), dtype=complex)
        self.zn[:,1] = (0.5*self.LSSS+self.Lbq)/np.cos(self.epsilon[:,0])*np.exp(1j*self.epsilon[:,0])
        for ii, epsilon in enumerate(self.epsilon):
            for n in np.arange(self.nd):
                e0 = epsilon[n]
                e1 = epsilon[n+1]
                dc =  np.linspace(0., 1., nsub+1)
                e = e0 + dc*(e1-e0)
                cc =  (np.cos(e)-np.cos(e0))/self.b[ii,n]
                # cc = np.linspace(0, np.cos(e1)-np.cos(e0), nsub+1)/self.b[ii,n]
                i0 = 1 + n*(nsub+1)
                z0 = self.zn[ii, 1]+self.z_begin[ii,n]
                self.zn[ii, i0:i0+nsub+1] = z0+self.Ln[n]*dc+self.rho*cc*1j
        self.zn[:,-1] = self.zn[:,-2] + (0.5*self.LSSS+self.Lbq)/np.cos(self.epsilon[:,-1])*np.exp(1j*self.epsilon[:,-1])

    def make_extrema(self):
        y_begin = np.imag(self.z_begin)
        y_end = np.imag(self.z_end)
        self.y_min = np.zeros((self.nb, self.nd))
        self.y_max = np.zeros((self.nb, self.nd))
        for ii, epsilon in enumerate(self.epsilon):
            for n in np.arange(self.nd):
                e0 = epsilon[n]
                e1 = epsilon[n+1]
                y0 = y_begin[ii, n]
                y1 = y_end[ii, n]
                bb = self.b[ii, n]
                if e0>0:
                    if e1>0:
                        self.y_min[ii, n] = y0
                        self.y_max[ii, n] = y1
                    else:
                        self.y_min[ii, n] = np.min([y0, y1])
                        self.y_max[ii, n] = y0+self.rho*(1-np.cos(e0))/bb
                elif e1>0:
                    self.y_min[ii, n] = y0+self.rho*(1-np.cos(e0))/bb
                    self.y_max[ii, n] = np.max([y0, y1])
                else:
                    self.y_min[ii, n] = y1
                    self.y_max[ii, n] = y0
        self.width = np.max(self.y_max, axis=0)-np.min(self.y_min, axis=0)
        self.max_apert_noshift = np.max(self.y_max)-np.min(self.y_min)
        self.max_apert = np.max(self.width[1:-1])

    def make_path_length(self):
        e1 = self.epsilon[:,1:]
        e0 = self.epsilon[:,:-1]
        ang = e0-e1
        self.path_length = np.sum(ang/self.hn, axis=1)+self.Ldd*np.sum(
            1/np.cos(e1),axis=1)+2*(self.LSSS+self.Lbq)/np.cos(e0[:,0])
        self.path_length_tot = 2*self.nc*self.path_length
        self.path_length_diff = np.max(self.path_length)-np.min(self.path_length)
        self.path_length_diff_tot = 2*self.nc*self.path_length_diff

    def calc_momentum_compaction(self, mux):
        a = self.a[1]
        b = self.b[1, -1]
        self.alpha = (np.pi/self.nc)**2*(
            1/np.sin(mux/2)**2-1/4+
            self.Ld/6/self.Larc+
            (a*b+self.n)/(6*self.Larc*self.n*(1+self.n))*(
                a*(1-b)*self.Ld + 4*self.nc*self.Ldd*(
                    self.n*(2+self.n)-a*b*(1+2*self.n))
                )
            )        
        
    def plot(self, xlim=(None, None), ylim=(None,None), figout=None):
        zn = self.zn.T
        plt.plot(np.real(zn), np.imag(zn))
        #plt.axvline(0,ls="--",color="k")
        zmin = np.min(np.imag(zn))
        zmax = np.max(np.imag(zn))
        dz = zmax - zmin
        zloc = (0.5*self.LSSS+self.Lbq)
        iz = 1
        for z1, z2 in zip(self.z_begin[0], self.z_end[0]):
            x1, x2 = np.real(z1), np.real(z2)
            plt.axvline(zloc + x1,ls="--",color="k")
            plt.axvline(zloc + x2,ls="--",color="k")
            plt.text(zloc + 0.5*(x1+x2), zmin + dz*1.1, f"D{iz:d}", 
                     horizontalalignment='center', verticalalignment='center')
            iz += 1
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.xlim(xlim)
        plt.ylim(ylim)
        if figout is not None:
            plt.savefig(figout)
        plt.show()
        plt.close()
        
    def make_madx_file(self, pattern=None):
        if pattern is None:
            pattern = self.pattern
        pattern = Path(pattern)
        aa = np.ravel(self.a)
        bb = np.ravel(self.b)
        nnc = np.ravel(self.nc)
        ll1 = np.ravel(self.L1)
        ll2 = np.ravel(self.L2)
        tthc = np.ravel(self.thetac)
        ee = np.transpose([np.ravel(e) for e in self.epsilon])
        rrho = np.transpose([np.ravel(r) for r in self.rhon])
        lcell = 2*np.ravel(self.path_length)+2*self.LSSS+4*self.Lbq
        for a, b, nc, l1, l2, lc, eps, rho, thc in zip(aa, bb, nnc, ll1, ll2, lcell, ee, rrho, tthc):
            new_name = pattern.stem + f"_{self.n:d}_{nc:d}_{a:2.3f}_{b:2.3f}.madx"
            with open(pattern.with_name(new_name), "w") as fout:
                print(f"TITLE,'FODO cell {nc} cells a={a:2.3f} b={b:2.3f}';", file=fout)
                print("OPTION,-ECHO,info,warn;", file=fout)
                print("OPTION, RBARC=FALSE;", file=fout)
                print("", file=fout)
                print("!---------------------------------------------------------------!", file=fout)
                print("BEAM, PARTICLE=ELECTRON, ENERGY=20, EX=1.5E-06, EY=3.E-06;", file=fout)

                print("!**********************************************************************", file=fout)
                print("!***************************** parameters *****************************", file=fout)
                print("!**********************************************************************", file=fout)
                print(f"int const n = {self.n:d}; ! number of family 2 in a half-cell", file=fout)
                print(f"int const nc = {nc:d}; ! total number of FODO cells", file=fout)
                print(f"real const a = {a}; ! ratio between the total length of family 2 and total dipole length", file=fout)
                print(f"real const b = {b}; ! ratio between the curvature radius of family 2 and reference radius", file=fout)
                
                print("", file=fout)
                print("!**********************************************************************", file=fout)
                print("!****************************** geometry ******************************", file=fout)
                print("!**********************************************************************", file=fout)
                
                print(f"real const L_arc = {nc*lc}; ! total length of the arcs [m]", file=fout)
                # real const length_coll = 97749.3853565; ! total collider length
                print(f"real const L_cell = {lc}; ! total length of the FODO cell", file=fout)
                print(f"real const ang_cell = {thc}; ! total angle of the FODO cell", file=fout)
                print(f"real const L_MB1 = {l1}; ! length of dipole of family 1 [m]", file=fout)
                print(f"real const L_MB2 = {l2}; ! length of dipole of family 2 [m]", file=fout)
                print(f"real const L_MQ = {self.LQP};  ! length of the main quadrupole [m]", file=fout)
                print(f"real const L_MS = {self.LSX};  ! length of the main sextupole [m]", file=fout)
                print(f"real const L_SSS = {self.LSSS};  ! length of the section housing qpoles and correctors [m]", file=fout)
                print(f"real const dl_mb_mb = {self.Ldd}; ! distance between 2 dipoles [m]", file=fout)
                print(f"real const dl_qq = {self.Lqq}; ! distance between 2 elements in the quadrupole section [m]", file=fout)
                print(f"real const dl_bq = {self.Lbq}; ! distance between dipole and the qp section [m]", file=fout)
                
                print("", file=fout)
                print("!**********************************************************************", file=fout)
                print("!****************************** elements ******************************", file=fout)
                print("!**********************************************************************", file=fout)
                for n, r, e1 ,e2 in zip(np.arange(2*self.n+1), rho, eps[:-1], eps[1:]):
                    ang = (e1-e2)
                    l = r*ang
                    if self.nsub == 1:
                        print(f"mb{n+1}: sbend, l={l}, angle={ang}, e1={e1}, e2={-e2};", file=fout)
                    elif self.nsub == 1:
                        print(f"mb{n+1}b: sbend, l={l}/{self.nsub}, angle={ang}/{self.nsub}, e1={e1}, e2=0;", file=fout)
                        print(f"mb{n+1}e: sbend, l={l}/{self.nsub}, angle={ang}/{self.nsub}, e1=0, e2={-e2};", file=fout)
                    else:
                        print(f"mb{n+1}b: sbend, l={l}/{self.nsub}, angle={ang}/{self.nsub}, e1={e1}, e2=0;", file=fout)
                        print(f"mb{n+1}m: sbend, l={l}/{self.nsub}, angle={ang}/{self.nsub}, e1=0, e2=0;", file=fout)
                        print(f"mb{n+1}e: sbend, l={l}/{self.nsub}, angle={ang}/{self.nsub}, e1=0, e2={-e2};", file=fout)
                print("mq: quadrupole, l=L_MQ;", file=fout)
                print("ms: sextupole, l=L_MS;", file=fout)
                
                print("k1q1 = sqrt(2)/L_cell/L_MQ/4;", file=fout)
                print("k1q2 = -sqrt(2)/L_cell/L_MQ/4;", file=fout)

                print("", file=fout)
                print("!**********************************************************************", file=fout)
                print("!****************************** sequence ******************************", file=fout)
                print("!**********************************************************************", file=fout)
                print("fodo_cell: SEQUENCE, L = l_cell, REFER = EXIT;", file=fout)
                print("ms.1: ms, at=l_ms, k2:=k2sx1;", file=fout)
                print("mq.1: mq, at=l_ms+l_mq+dl_qq, k1:=k1q1;", file=fout)
                z0 = 0.5*self.LSSS+self.Lbq
                for n, r, e1 ,e2 in zip(np.arange(2*self.n+1), rho, eps[:-1], eps[1:]):
                    ang = (e1-e2)
                    l = r*ang
                    if self.nsub == 1:
                        z0 += l
                        print(f"mb{n+1}, at={z0};", file=fout)
                    else:
                        for k in np.arange(self.nsub):
                            z0 += l/self.nsub
                            if k == 0:
                                print(f"mb{n+1}b, at={z0};", file=fout)
                            elif k == self.nsub-1:
                                print(f"mb{n+1}e, at={z0};", file=fout)
                            else:                        
                                print(f"mb{n+1}m, at={z0};", file=fout)                        
                    z0 += self.Ldd/np.cos(e2)
                print("ms.2: ms, at=l_ms+0.5*L_cell, k2:=k2sx2;", file=fout)
                print("mq.2: mq, at=l_ms+l_mq+dl_qq+0.5*L_cell, k1:=k1q2;", file=fout)
                z0 = 0.5*self.LSSS+self.Lbq+0.5*lc
                for n, r, e1 ,e2 in zip(np.arange(2*self.n+1), rho, eps[:-1], eps[1:]):
                    ang = (e1-e2)
                    l = r*ang
                    if self.nsub == 1:
                        z0 += l
                        print(f"mb{n+1}, at={z0};", file=fout)
                    else:
                        for k in np.arange(self.nsub):
                            z0 += l/self.nsub
                            if k == 0:
                                print(f"mb{n+1}b, at={z0};", file=fout)
                            elif k == self.nsub-1:
                                print(f"mb{n+1}e, at={z0};", file=fout)
                            else:                        
                                print(f"mb{n+1}m, at={z0};", file=fout)                        
                    z0 += self.Ldd/np.cos(e2)
                print("ENDSEQUENCE;", file=fout)

                print("full_ring: SEQUENCE, L = l_arc, REFER = EXIT;", file=fout)
                for k in np.arange(nc):
                    print(f"fodo_cell, at= l_cell*{k+1};", file=fout)
                print("ENDSEQUENCE;", file=fout)
                
                print("use, sequence = fodo_cell;", file=fout)
                print(f"survey, theta0={thc/4}, file=survey_cell_{self.n:d}_{nc:d}_{a:2.3f}_{b:2.3f}.svy;", file=fout)
                
                print("use, sequence = fodo_cell;", file=fout)
                print("match, sequence=fodo_cell;", file=fout)
                print("global,sequence=fodo_cell,q1=0.25,q2=0.25;", file=fout)
                print("vary,name=K1Q1,step=1.0e-6;", file=fout)
                print("vary,name=K1Q2,step=1.0e-6;", file=fout)
                print("jacobian,calls=200,tolerance=1.e-21;", file=fout)
                print("lmdif,calls=10000;", file=fout)
                print("jacobian,calls=100,tolerance=1e-15,bisec=1;", file=fout)
                print("endmatch;", file=fout)

                print("use, sequence = fodo_cell;", file=fout)
                print("match, sequence=fodo_cell;", file=fout)
                print("global,sequence=fodo_cell,dq1=0.,dq2=0.;", file=fout)
                print("vary,name=K2SX1,step=1.0e-6;", file=fout)
                print("vary,name=K2SX2,step=1.0e-6;", file=fout)
                print("jacobian,calls=200,tolerance=1.e-21;", file=fout)
                print("lmdif,calls=10000;", file=fout)
                print("jacobian,calls=100,tolerance=1e-15,bisec=1;", file=fout)
                print("endmatch;", file=fout)

                print("use, sequence = full_ring;", file=fout)
                print(f"survey, file=survey_{self.n:d}_{nc:d}_{a:2.3f}_{b:2.3f}.svy;", file=fout)

                # stop;
        

if __name__== "__main__":        
    import __main__
    import scipy.constants as cons
    cur_dir = Path(__main__.__file__).parent.resolve()
    Bnc = 1.8
    Bsc = 10
    br_inj = 313935*1e6/cons.c
    br_ext = 750106*1e6/cons.c
    Ldd = 0.4
    tot_nc = np.pi*(br_ext-br_inj)/Bnc
    tot_sc = np.pi*(br_ext+br_inj)/Bsc
    
    tot_nc_harm2 = tot_nc/8
    Ld = tot_nc + tot_sc + tot_nc_harm2
    Larc = Ld + 150*(4*0.4*2+4*2)
    rho0 = Ld/2/np.pi
    h0 = 2*np.pi/Ld
    
    def make_tab(nd, nb=1, sc_first=False):
        a_nc1 = tot_nc/Ld
        a_sc = tot_sc/Ld
        a_nc2 = tot_nc_harm2/Ld
        dphi = np.linspace(-np.pi/2, np.pi/2, nb)
        Bnc1 = Bnc*np.sin(dphi)
        Bnc2 = -Bnc*np.sin(2*dphi)/8        
        Bbar = Bnc1*a_nc1+Bsc*a_sc+Bnc2*a_nc2
        b_nc1 = Bnc1/Bbar        
        b_sc = Bsc/Bbar
        b_nc2 = Bnc2/Bbar
        if sc_first:
            a_nc2 /= 2*nd-1
            a_sc /= 2*nd
            a_nc1 /= 2*nd
            tab_a = [a_sc, a_nc1, a_nc2]*nd
            tab_b = [b_sc, b_nc1, b_nc2]*nd
        else:
            a_nc1 /= 2*nd
            a_sc /= 2*nd-1
            a_nc2 /= 2*nd
            tab_a = [a_nc1, a_nc2, a_sc]*nd
            tab_b = [b_nc1, b_nc2, b_sc]*nd
        tab_a += tab_a[::-1][1:]
        tab_b += tab_b[::-1][1:]
        return (np.array(tab_a), np.array(tab_b).T)
 
    nb = 6
    tab_a, tab_b = make_tab(2, nb, sc_first=True)
    multi_sc = MultiBend(a = tab_a, b= tab_b, nc=250, Ld=Ld, Larc=Larc)#, LSSS=0, Lbq=0)
    multi_sc.plot()#ylim=(0,0.1))    

    tab_nc = np.arange(60, 301, 4, dtype=int)
    # multi_sc.plot(figout="sc_first_traj_150.png")#ylim=(0,0.1))    

    tab_a, tab_b = make_tab(1, nb, sc_first=True)
    multi1 = [
        MultiBend(a=tab_a,  b=tab_b, nc=nc, Ld=Ld, Larc=Larc)
        for nc in tab_nc]
    tab_a, tab_b = make_tab(2, nb, sc_first=True)
    multi2 = [
        MultiBend(a=tab_a,  b=tab_b, nc=nc, Ld=Ld, Larc=Larc)
        for nc in tab_nc]
    tab_a, tab_b = make_tab(3, nb, sc_first=True)
    multi3 = [
        MultiBend(a=tab_a,  b=tab_b, nc=nc, Ld=Ld, Larc=Larc)
        for nc in tab_nc]
    plt.semilogy(tab_nc, [np.ravel(m.path_length_diff_tot) for m in multi1], "b-", label="3 dipoles")
    plt.semilogy(tab_nc, [np.ravel(m.path_length_diff_tot) for m in multi2], "r-", label="5 dipoles")
    plt.semilogy(tab_nc, [np.ravel(m.path_length_diff_tot) for m in multi3], "g-", label="7 dipoles")
    # plt.semilogy(tab_nc, [np.ravel(m.path_length_approx_diff_tot) for m in multi1], "b--", label="approx 3 dipoles")
    # plt.semilogy(tab_nc, [np.ravel(m.path_length_approx_diff_tot) for m in multi2], "r--", label="approx 5 dipoles")
    # plt.semilogy(tab_nc, [np.ravel(m.path_length_approx_diff_tot) for m in multi3], "g--", label="approx 7 dipoles")
    plt.legend()
    plt.xlabel("Number of FODO cells")
    plt.ylabel("Total path length difference [m]")
    plt.ylim(1e-2,1)
    # plt.savefig("sc_first_path_length.png")
    plt.show()
    plt.close()
    plt.semilogy(tab_nc, [np.ravel(m.max_apert) for m in multi1], "b-", label="3 dipoles")
    plt.semilogy(tab_nc, [np.ravel(m.max_apert) for m in multi2], "r-", label="5 dipoles")
    plt.semilogy(tab_nc, [np.ravel(m.max_apert) for m in multi3], "g-", label="7 dipoles")
    plt.legend()
    plt.xlabel("Number of FODO cells")
    plt.ylabel("Trajectory difference [m]")
    plt.ylim(1e-2,1)
    # plt.savefig("sc_first_max_offset.png")
    plt.show()
    plt.close()
    
    tab_a, tab_b = make_tab(2, nb, sc_first=False)
    multi_nc = MultiBend(a = tab_a, b= tab_b, nc=150, Ld=Ld, Larc=Larc)
    multi_nc.plot()#ylim=(0,0.1))    

    tab_a, tab_b = make_tab(1, nb, sc_first=False)
    multi1 = [
        MultiBend(a=tab_a,  b=tab_b, nc=nc, Ld=Ld, Larc=Larc)
        for nc in tab_nc]
    tab_a, tab_b = make_tab(1, nb, sc_first=False)
    multi2 = [
        MultiBend(a=tab_a,  b=tab_b, nc=nc, Ld=Ld, Larc=Larc)
        for nc in tab_nc]
    tab_a, tab_b = make_tab(1, nb, sc_first=False)
    multi3 = [
        MultiBend(a=tab_a,  b=tab_b, nc=nc, Ld=Ld, Larc=Larc)
        for nc in tab_nc]
    plt.semilogy(tab_nc, [np.ravel(m.path_length_diff_tot) for m in multi1], "b-", label="3 dipoles")
    plt.semilogy(tab_nc, [np.ravel(m.path_length_diff_tot) for m in multi2], "r-", label="5 dipoles")
    plt.semilogy(tab_nc, [np.ravel(m.path_length_diff_tot) for m in multi3], "g-", label="7 dipoles")
    # plt.semilogy(tab_nc, [np.ravel(m.path_length_approx_diff_tot) for m in multi1], "b--", label="approx 3 dipoles")
    # plt.semilogy(tab_nc, [np.ravel(m.path_length_approx_diff_tot) for m in multi2], "r--", label="approx 5 dipoles")
    # plt.semilogy(tab_nc, [np.ravel(m.path_length_approx_diff_tot) for m in multi3], "g--", label="approx 7 dipoles")
    plt.legend()
    plt.xlabel("Number of FODO cells")
    plt.ylabel("Total path length difference [m]")
    plt.ylim(1e-2,1)
    # plt.savefig("nc_first_path_length.png")
    plt.show()
    plt.close()
    plt.semilogy(tab_nc, [np.ravel(m.max_apert) for m in multi1], "b-", label="3 dipoles")
    plt.semilogy(tab_nc, [np.ravel(m.max_apert) for m in multi2], "r-", label="5 dipoles")
    plt.semilogy(tab_nc, [np.ravel(m.max_apert) for m in multi3], "g-", label="7 dipoles")
    plt.legend()
    plt.xlabel("Number of FODO cells")
    plt.ylabel("Trajectory difference [m]")
    plt.ylim(1e-2,1)
    # plt.savefig("nc_first_max_offset.png")
    plt.show()
    plt.close()

