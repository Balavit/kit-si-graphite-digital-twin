import os, sys
import pandas as pd, numpy as np, itertools, json
from sklearn.linear_model import LinearRegression
rng=np.random.default_rng(0)
T=pd.read_csv('checkups.csv').sort_values(['cell','days']); T['soh']=T.cap/3.0
KNEE=0.75  # pre-knee data only (loss < 25%)
def stlsq(X,y,thr=0.05,iters=20,w=None):
    # normalized STLSQ; column 0 is the intercept and is never thresholded
    sc=X.std(0); sc[0]=1; sc[sc==0]=1; Xn=X/sc; coef=np.linalg.lstsq(Xn,y,rcond=None)[0]; active=np.ones(X.shape[1],bool)
    for _ in range(iters):
        ref=np.abs(coef[1:][active[1:]]).max() if active[1:].any() else 0
        small=np.abs(coef)<thr*ref; small[0]=False
        active=active&~small
        coef=np.zeros(X.shape[1]); coef[active]=np.linalg.lstsq(Xn[:,active],y,rcond=None)[0]
    return coef/sc, active
def bic(X,y,active,n):
    k=active.sum(); 
    if k==0: return np.inf
    b=np.linalg.lstsq(X[:,active],y,rcond=None)[0]; rss=((y-X[:,active]@b)**2).sum(); return n*np.log(rss/n)+k*np.log(n)
def select(X,y,names,cells=None):
    cand=[]
    for thr in [0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.7,0.9]:
        c,a=stlsq(X,y,thr)
        uc=np.unique(cells); fl=np.array_split(np.random.default_rng(1).permutation(uc),5); err=[]
        for f in fl:
            tr=~np.isin(cells,f); b=np.linalg.lstsq(X[tr][:,a],y[tr],rcond=None)[0]; err.append(((y[~tr]-X[~tr][:,a]@b)**2).mean())
        cand.append((float(np.mean(err)),float(np.std(err)/np.sqrt(len(err))),thr,c,a))
    vmin=min(x[0] for x in cand); se=[x[1] for x in cand if x[0]==vmin][0]
    ok=[x for x in cand if x[0]<=vmin+se]            # one-standard-error rule: sparsest model within 1 SE of the best
    best=min(ok,key=lambda x:(x[4].sum(),x[0]))
    return best[0],best[2],best[3],best[4]
out={}
# ---------- calendar: log-space library
cal=T[T.age_type==1]; rows=[]
for c,g in cal.groupby('cell'):
    g=g[g.days>1]; q0=g.cap.iloc[0]; L=1-g.cap/q0; t=g.days.values
    for i in range(1,len(g)):
        if L.iloc[i]<=L.iloc[i-1] or L.iloc[i]<0.003 or g.soh.iloc[i]<KNEE: continue
        dt=t[i]-t[i-1]; rate=(L.iloc[i]-L.iloc[i-1])/dt; tm=np.sqrt(t[i]*t[i-1])
        rows.append(dict(cell=c,lnrate=np.log(rate),lnt=np.log(tm),soc=g.age_soc.iloc[0]/100,invT=1/(8.314e-3*(g.age_temp.iloc[0]+273.15)),T=g.age_temp.iloc[0]))
D=pd.DataFrame(rows); D['invT']=D.invT-D.invT.mean()  # centre 1/RT so main effects are interpretable
lib={'1':np.ones(len(D)),'ln t':D.lnt,'SoC':D.soc,'1/RT':D.invT,'SoC·1/RT':D.soc*D.invT,'SoC²':D.soc**2,'ln t·SoC':D.lnt*D.soc,'ln t·1/RT':D.lnt*D.invT,'(1/RT)²':D.invT**2}
names=list(lib); X=np.column_stack([lib[k] for k in names]); y=D.lnrate.values
v,thr,c,a=select(X,y,names,D.cell.values)
imposed=['1','ln t','SoC','1/RT']; ai=np.array([n in imposed for n in names])
def r2(active):
    b=np.linalg.lstsq(X[:,active],y,rcond=None)[0]; return 1-((y-X[:,active]@b)**2).sum()/((y-y.mean())**2).sum()
# leave-cells-out CV
cells=D.cell.unique(); folds=np.array_split(rng.permutation(cells),5)
def cv(active):
    err=[]
    for f in folds:
        tr=~D.cell.isin(f).values; b=np.linalg.lstsq(X[tr][:,active],y[tr],rcond=None)[0]; err.append(((y[~tr]-X[~tr][:,active]@b)**2).mean())
    return float(np.mean(err))
# bootstrap stability
boot=np.zeros(len(names)); B=60
for _ in range(B):
    s=rng.choice(cells,len(cells)); idx=np.concatenate([np.where(D.cell.values==cc)[0] for cc in s])
    _,_,_,ab=select(X[idx],y[idx],names,D.cell.values[idx]); boot+=ab
out['calendar']=dict(n_increments=len(D),n_cells=len(cells),discovered={n:float(c[i]) for i,n in enumerate(names) if a[i]},
    r2_discovered=float(r2(a)),r2_imposed=float(r2(ai)),cv_mse_discovered=cv(a),cv_mse_imposed=cv(ai),bic_discovered=float(v),bic_imposed=float(bic(X,y,ai,len(y))),
    bootstrap_selection_freq={n:float(boot[i]/B) for i,n in enumerate(names)})
# interpret: exponent n = 1 + coef(ln t); Ea = -coef(1/RT); SoC-dependent Ea: Ea(SoC)= -(c_invT + c_soc_invT*SoC)
# ---------- cycling: linear-space library on loss per Ah, both Arrhenius signs
cyc=T[T.age_type.isin([2])]; rows=[]
for c,g in cyc.groupby('cell'):
    g=g[g.days>1]; q0=g.cap.iloc[0]; L=1-g.cap/q0; Q=g.q_chg.cumsum().values if 'q_chg' in g else None
    # throughput: use total charge throughput column if available
    Qt=g.q_chg.values
    for i in range(1,len(g)):
        dQ=Qt[i]-Qt[i-1]
        if dQ<=1 or L.iloc[i]<=L.iloc[i-1] or g.soh.iloc[i]<KNEE: continue
        dt=g.days.values[i]-g.days.values[i-1]
        rows.append(dict(cell=c,rate=(L.iloc[i]-L.iloc[i-1])/dQ,ratet=(L.iloc[i]-L.iloc[i-1])/dt,Tk=g.age_temp.iloc[0]+273.15,C=g.age_chg_rate.iloc[0],dV=g.V_max_cyc_V.iloc[0]-g.V_min_cyc_V.iloc[0],Vmax=g.V_max_cyc_V.iloc[0],Vmin=g.V_min_cyc_V.iloc[0],t=np.sqrt(g.days.values[i]*g.days.values[i-1]),dQdt=dQ/dt,T=g.age_temp.iloc[0]))
E=pd.DataFrame(rows); R=8.314e-3; Tref=298.15
arrm=np.exp(-20/R*(1/E.Tk-1/Tref)); arrp=np.exp(+20/R*(1/E.Tk-1/Tref))   # 20 kJ/mol reference shapes, both signs
base={'1':np.ones(len(E)),'C':E.C,'ΔV':E.dV,'A⁻(SEI)':arrm,'A⁺(plating)':arrp,'C·A⁺':E.C*arrp,'C·A⁻':E.C*arrm,'ΔV·A⁻':E.dV*arrm,'ΔV·A⁺':E.dV*arrp,'C·ΔV':E.C*E.dV,'C²':E.C**2,'1/t·dt/dQ':(1/E.t)/E.dQdt,'Vmax':E.Vmax}
names2=list(base); X2=np.column_stack([base[k] for k in names2]); y2=E.rate.values*1e3  # loss per kAh -> per Ah *1e3
v2,thr2,c2,a2=select(X2,y2,names2,E.cell.values)
imposed2=['1','A⁻(SEI)','C·A⁻','ΔV·A⁻','A⁺(plating)','C·A⁺']; ai2=np.array([n in imposed2 for n in names2])
cells2=E.cell.unique(); folds=np.array_split(rng.permutation(cells2),5)
def r2b(X,y,active):
    b=np.linalg.lstsq(X[:,active],y,rcond=None)[0]; return float(1-((y-X[:,active]@b)**2).sum()/((y-y.mean())**2).sum())
def cv2(active):
    err=[]
    for f in folds:
        tr=~E.cell.isin(f).values; b=np.linalg.lstsq(X2[tr][:,active],y2[tr],rcond=None)[0]; err.append(((y2[~tr]-X2[~tr][:,active]@b)**2).mean())
    return float(np.mean(err))
boot2=np.zeros(len(names2))
for _ in range(B):
    s=rng.choice(cells2,len(cells2)); idx=np.concatenate([np.where(E.cell.values==cc)[0] for cc in s])
    _,_,_,ab=select(X2[idx],y2[idx],names2,E.cell.values[idx]); boot2+=ab
out['cycling']=dict(n_increments=len(E),n_cells=len(cells2),discovered={n:float(c2[i]) for i,n in enumerate(names2) if a2[i]},
    r2_discovered=r2b(X2,y2,a2),r2_imposed=r2b(X2,y2,ai2),cv_mse_discovered=cv2(a2),cv_mse_imposed=cv2(ai2),bic_discovered=float(v2),bic_imposed=float(bic(X2,y2,ai2,len(y2))),
    bootstrap_selection_freq={n:float(boot2[i]/B) for i,n in enumerate(names2)},units='loss fraction per kAh throughput')
json.dump(out,open('symreg.json','w'),indent=1); D.to_csv('symreg_calendar_data.csv',index=False); E.to_csv('symreg_cycling_data.csv',index=False)
print(json.dumps(out,indent=1))
