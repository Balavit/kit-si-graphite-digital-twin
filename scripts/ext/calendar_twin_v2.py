import os, sys
import pandas as pd, numpy as np, json
from scipy.optimize import least_squares
T=pd.read_csv('checkups.csv').sort_values(['cell','days']); cal=T[T.age_type==1]
cells=cal.drop_duplicates('cell').set_index('cell'); R=8.314e-3
rng=np.random.default_rng(0); test=[]
for cond,g in cells.groupby('cond'): test+=list(rng.choice(g.index,1,replace=False))
def feats(c): r=cells.loc[c]; return 1/(R*(r.age_temp+273.15))-1/(R*298.15), r.age_soc/100
def m1(th,d,iT,soc):  # one-exponent law (as in the twin): k t^n
    lnk,E,s,n=th; return np.exp(lnk+s*soc-E*iT)*d**n
def m2(th,d,iT,soc):  # regime law: transient + sqrt(t) + linear term switched on by SoC and T
    lnk,E,s,L0,tau,lnkl,El,sl,soc0,w=th
    ksq=np.exp(lnk+s*soc-E*iT); klin=np.exp(lnkl+sl*soc-El*iT)/(1+np.exp(-(soc-soc0)/w))
    return L0*(1-np.exp(-d/tau))+ksq*np.sqrt(d)+klin*d
def fit(model,th0,lo,hi,D):
    def res(th):
        r=[]
        for c,g in D.groupby('cell'):
            iT,soc=feats(c); L=(1-g.cap/g.cap.iloc[0]).values; r.append(L-model(th,g.days.values,iT,soc))
        return np.concatenate(r)/0.01
    return least_squares(res,th0,bounds=(lo,hi),loss='soft_l1').x
tr=cal[~cal.cell.isin(test)]; te=cal[cal.cell.isin(test)]
th1=fit(m1,[np.log(0.004),10,0.8,0.5],[-10,0,-2,0.2],[0,80,3,1.0],tr)
th2=fit(m2,[np.log(0.003),5,0.5,0.01,10,np.log(1e-4),30,2,0.9,0.05],[-12,0,-2,0,1,-14,0,-2,0.5,0.01],[0,80,3,0.05,60,-2,120,6,1.2,0.3],tr)
out={'th1':dict(zip(['lnk','E','s','n'],map(float,th1))),'th2':dict(zip(['lnk_sqrt','E_sqrt','s_sqrt','L0','tau_d','lnk_lin','E_lin','s_lin','soc0','w'],map(float,th2)))}
# prospective: population forecast (no adaptation) from history H days: error on all later check-ups, plus with a 1-parameter offset adaptation (fit a scalar multiplier on history)
def evalH(H):
    e1=[];e2=[];e1a=[];e2a=[]
    for c,g in te.groupby('cell'):
        iT,soc=feats(c); L=(1-g.cap/g.cap.iloc[0]).values; d=g.days.values; h=d<=H; f=d>H
        if f.sum()<3 or h.sum()<3: continue
        p1=m1(th1,d,iT,soc); p2=m2(th2,d,iT,soc)
        e1+=list(L[f]-p1[f]); e2+=list(L[f]-p2[f])
        a1=np.sum(L[h]*p1[h])/max(np.sum(p1[h]**2),1e-9); a2=np.sum(L[h]*p2[h])/max(np.sum(p2[h]**2),1e-9)
        e1a+=list(L[f]-a1*p1[f]); e2a+=list(L[f]-a2*p2[f])
    rm=lambda e: float(np.sqrt(np.mean(np.square(e))))
    return dict(one_exponent=rm(e1),regime=rm(e2),one_exponent_adapted=rm(e1a),regime_adapted=rm(e2a),n=len(e1))
out['forecast_rmse_heldout_calendar']={H:evalH(H) for H in [100,200,300]}
# in-sample fit quality by condition for the regime model: which conditions get a linear term
rows=[]
for cond,g in cells.groupby('cond'):
    iT,soc=feats(g.index[0]); lnk,E,s,L0,tau,lnkl,El,sl,soc0,w=th2
    klin=np.exp(lnkl+sl*soc-El*iT)/(1+np.exp(-(soc-soc0)/w)); ksq=np.exp(lnk+s*soc-E*iT)
    rows.append(dict(T=g.age_temp.iloc[0],SoC=g.age_soc.iloc[0],k_sqrt=ksq,k_lin=klin,linear_share_600d=klin*600/(klin*600+ksq*np.sqrt(600)+L0)))
out['regime_terms']=pd.DataFrame(rows).round(5).to_dict('records')
json.dump(out,open('calendar_twin_v2.json','w'),indent=1); print(json.dumps(out,indent=1))
