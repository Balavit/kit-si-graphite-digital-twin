import os, sys
import numpy as np, pandas as pd
from scipy.optimize import least_squares
def model(p,w):
    R0,L,R1,t1,a1,R2,t2,a2,s=p
    jw=1j*w
    Z=R0+jw*L+R1/(1+(jw*t1)**a1)+R2/(1+(jw*t2)**a2)+s*(1-1j)/np.sqrt(w)
    return Z
def fit(f,zre,zim,p0=None):
    w=2*np.pi*f; z=zre+1j*zim
    # units mOhm, H->µH scale: L in mOhm*s
    if p0 is None:
        R0=zre[np.argmin(np.abs(zim))]; p0=[R0,2e-4,3.0,1.5e-3,0.8,1.0,0.1,0.8,0.5]
    lb=[0,0,0,2e-4,0.4,0,2e-2,0.4,0]; ub=[100,1e-2,50,1e-2,1,100,3,1,30]
    p0=np.clip(p0,np.array(lb)*1.001+1e-9,np.array(ub)*0.999)
    def res(p):
        zm=model(p,w); sc=np.abs(z)+0.5
        return np.r_[(zm.real-z.real)/sc,(zm.imag-z.imag)/sc]
    r=least_squares(res,p0,bounds=(lb,ub),max_nfev=4000)
    rmse=np.sqrt(np.mean(r.fun**2))
    return r.x,rmse
def drt(f,zre,zim,lam=1e-2,ntau=60):
    # Tikhonov DRT on the polarization part (after removing R0 and inductance via fit is overkill); simple: subtract min real
    w=2*np.pi*f; tau=np.logspace(np.log10(1/w.max())-1,np.log10(1/w.min())+1,ntau)
    A=1/(1+1j*np.outer(w,tau)); 
    Ar=np.c_[np.ones(len(w)),A.real]; Ai=np.c_[np.zeros(len(w)),A.imag]
    # fit both parts, nonneg via nnls on stacked with regularization
    from scipy.optimize import nnls
    D=np.diff(np.eye(ntau+1),axis=0)[:, :]; 
    M=np.r_[Ar,Ai,lam*D]; y=np.r_[zre,zim,np.zeros(D.shape[0])]
    g,_=nnls(M,y); return tau,g[1:],g[0]
if __name__=="__main__":
    z=pd.read_csv(os.environ['KIT_RAW']+'/eis/cell_eisv2_P012_1_S12_C11.csv',sep=';'); z=z[(z.is_rt==1)&(z.valid==1)&z.z_re_comp_mOhm.notna()]
    for ts in sorted(z.timestamp_s.unique())[::12]:
        for soc in (10,50,90):
            s=z[(z.timestamp_s==ts)&(z.soc_nom==soc)].sort_values('freq_Hz',ascending=False)
            if len(s)<20: continue
            f=s.freq_Hz.values; zr=s.z_re_comp_mOhm.values; zi=-s.z_im_comp_mOhm.values  # file stores -Im(Z)
            # careful: fit uses zim as given (negative for capacitive); model imag negative
            p,rm=fit(f,zr,zi)
            tau,g,_=drt(f,zr-zr.min(),zi)
            pk=tau[np.r_[False,(g[1:-1]>g[:-2])&(g[1:-1]>g[2:])&(g[1:-1]>0.05*g.max()),False]]
            print(f"t={ (ts-z.timestamp_s.min())/86400:5.0f}d soc={soc:2d} R0={p[0]:.2f} R1={p[2]:.2f} t1={p[3]*1e3:.2f}ms a1={p[4]:.2f} R2={p[5]:.2f} t2={p[6]*1e3:.0f}ms a2={p[7]:.2f} W={p[8]:.2f} rmse={rm:.3f} | DRT peaks(ms): {np.round(pk*1e3,2)}")
