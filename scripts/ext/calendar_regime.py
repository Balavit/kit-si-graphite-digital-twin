import os, sys
import pandas as pd, numpy as np, json
from scipy.optimize import curve_fit
from scipy import stats
T=pd.read_csv('checkups.csv'); cal=T[T.age_type==1].sort_values(['cell','days'])
pw=lambda t,L0,k,n: L0+k*t**n; sq=lambda t,L0,k: L0+k*np.sqrt(t); lin=lambda t,L0,k: L0+k*t
def fits(t,L):
    out={}
    p,_=curve_fit(pw,t,L,p0=[0,1e-3,0.6],bounds=([-0.1,0,0.1],[0.1,1,2]),maxfev=50000); out['n']=p[2]; out['k']=p[1]; out['L0']=p[0]
    out['r2_pow']=1-((L-pw(t,*p))**2).sum()/((L-L.mean())**2).sum()
    ps,_=curve_fit(sq,t,L); out['k_sqrt']=ps[1]; out['r2_sqrt']=1-((L-sq(t,*ps))**2).sum()/((L-L.mean())**2).sum()
    pl,_=curve_fit(lin,t,L); out['k_lin']=pl[1]; out['r2_lin']=1-((L-lin(t,*pl))**2).sum()/((L-L.mean())**2).sum()
    # no-offset log-log fit as used before (reference = first point after day 1)
    Lr=1-(1-L)/(1-L[0]); m=Lr>0.002; out['n_loglog_noOffset']=np.polyfit(np.log(t[m]),np.log(Lr[m]),1)[0] if m.sum()>5 else np.nan
    # F-test: does the free exponent improve on sqrt significantly?
    rss_s=((L-sq(t,*ps))**2).sum(); rss_p=((L-pw(t,*p))**2).sum(); n=len(t)
    F=((rss_s-rss_p)/1)/(rss_p/(n-3)); out['p_vs_sqrt']=1-stats.f.cdf(F,1,n-3)
    return out
rows=[]
for c,g in cal.groupby('cell'):
    g=g[g.days>1]; t=g.days.values; L=(1-g.cap/g.cap.iloc[0]).values
    if len(t)<10: continue
    r=fits(t,L); r.update(cell=c,T=g.age_temp.iloc[0],SoC=g.age_soc.iloc[0],loss_end=L[-1],days=t[-1]); rows.append(r)
C=pd.DataFrame(rows); C.to_csv('calendar_cells.csv',index=False)
rows=[]
for (temp,soc),g in cal.groupby(['age_temp','age_soc']):
    piv=g.pivot_table(index=g.days.round(0),columns='cell',values='cap'); m=piv.mean(axis=1); t=m.index.values; L=1-m.values/m.values[0]; mk=t>1
    r=fits(t[mk],L[mk]); r.update(T=temp,SoC=soc,n_cells=piv.shape[1],loss_end=L[-1]); rows.append(r)
K=pd.DataFrame(rows); K.to_csv('calendar_conditions.csv',index=False)
K['regime']=np.where(K.n>0.75,'linear','sub-sqrt'); 
print(K[['T','SoC','n','L0','r2_pow','r2_sqrt','r2_lin','n_loglog_noOffset','p_vs_sqrt','loss_end','regime']].round(3).to_string())
mild=K[K.regime=='sub-sqrt']; sev=K[K.regime=='linear']
out=dict(n_mild=dict(mean=float(mild.n.mean()),sd=float(mild.n.std()),n=len(mild)),n_severe=dict(mean=float(sev.n.mean()),sd=float(sev.n.std()),conditions=sev[['T','SoC']].values.tolist()),
 n_loglog_noOffset_mean=float(K.n_loglog_noOffset.mean()),n_loglog_noOffset_sd=float(K.n_loglog_noOffset.std()),
 L0_mean=float(mild.L0.mean()),r2_sqrt_mild=dict(min=float(mild.r2_sqrt.min()),median=float(mild.r2_sqrt.median())),r2_sqrt_severe=float(sev.r2_sqrt.mean()),r2_lin_severe=float(sev.r2_lin.mean()),
 loss_end_mild=float(mild.loss_end.mean()),loss_end_severe=float(sev.loss_end.mean()),
 cells_n_mean=float(C.n.mean()),cells_n_sd=float(C.n.std()),cells_frac_n_below_0p6=float((C.n<0.6).mean()),cells_frac_sig_vs_sqrt=float((C.p_vs_sqrt<0.05).mean()))
# Arrhenius of sqrt-t rate constant with offset, mild regime only, per SoC
arr={}
for soc,g in C[C.n<0.75].groupby('SoC'):
    gg=g.groupby('T').k_sqrt.mean(); 
    if len(gg)<3: continue
    sl,ic,r,p,se=stats.linregress(1/(gg.index.values+273.15),np.log(gg.values)); arr[int(soc)]=dict(Ea_kJ=float(-sl*8.314/1000),r2=float(r**2),nT=len(gg))
out['Ea_by_SoC_sqrt_offset']=arr
json.dump(out,open('calendar_regime.json','w'),indent=1); print(json.dumps(out,indent=1))
