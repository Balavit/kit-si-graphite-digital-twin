import os, sys
import pandas as pd, numpy as np, json, itertools
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(0)
U=pd.read_csv('checkups.csv').sort_values(['cell','days']); U['soh']=U.cap/3
H=pd.read_csv('H2_knee.csv').set_index('cell')
E=pd.read_csv('eis_decomp.csv'); RB=pd.read_csv('reversible_blocks.csv'); CY=pd.read_csv('cycles.csv')
# per-cell, per-checkup EIS bands at SoC 50 and 90 (nearest in time)
def band(cell,soc,t,k):
    g=E[(E.cell==cell)&(E.soc==soc)]
    if len(g)==0: return np.nan
    i=np.argmin(np.abs(g.timestamp_s.values-t)); return g[k].values[i] if abs(g.timestamp_s.values[i]-t)<3*86400 else np.nan
rows=[]
for cell,g in U[U.age_type.isin([2,3])].groupby('cell'):
    if cell not in H.index: continue
    g=g.reset_index(drop=True); knee=H.loc[cell,'knee']; dk=H.loc[cell,'day_knee'] if knee else np.inf
    cyc=CY[CY.cell==cell].sort_values('timestamp_s'); rb=RB[RB.cell==cell]
    for i in range(3,len(g)):
        d=g.days[i]
        if d>=dk: break
        w=g.iloc[i-2:i+1]
        f=dict(cell=cell,cond=g.cond[0],days=d,T=g.age_temp[0],C=g.age_chg_rate[0],dV=g.V_max_cyc_V[0]-g.V_min_cyc_V[0],atype=g.age_type[0],
               soh=g.soh[i], dsoh_dt=np.polyfit(w.days,w.soh,1)[0]*21, curv=(g.soh[i]-2*g.soh[i-1]+g.soh[i-2]),
               R0_rel=g.R0_s50[i]/g.R0_s50[0], Rmid_rel=g.Rmid_s50[i]/g.Rmid_s50[0], Rlow_rel=g.Rlow_s50[i]/g.Rlow_s50[0], Rmid10_rel=g.Rmid_s10[i]/g.Rmid_s10[0],
               Rp10ms_rel=g.Rp10ms_s50[i]/g.Rp10ms_s50[0],
               HF_rel=band(cell,50,g.t[i],'R_sei')-band(cell,50,g.t[0],'R_sei'), MF_rel=band(cell,50,g.t[i],'R_ct')-band(cell,50,g.t[0],'R_ct'),
               MF90_rel=band(cell,90,g.t[i],'R_ct')-band(cell,90,g.t[0],'R_ct'))
        blk=cyc[(cyc.timestamp_s>g.t[i-1])&(cyc.timestamp_s<g.t[i])&(cyc.cyc_charged==0)]
        f['CE_med']=blk.coulomb_efficiency.median() if len(blk)>3 else np.nan
        r=rb[(rb.block==i-1)]; f['rev_frac']=r.rev_frac.values[0] if len(r) else np.nan
        f['knee_within_90']=int(knee and (dk-d)<=90); f['knee_within_60']=int(knee and (dk-d)<=60); f['days_to_knee']=dk-d if knee else np.nan
        rows.append(f)
D=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan); D.to_csv('knee_features.csv',index=False)
FEAT=['soh','dsoh_dt','curv','R0_rel','Rmid_rel','Rlow_rel','Rmid10_rel','Rp10ms_rel','HF_rel','MF_rel','MF90_rel','CE_med','rev_frac','T','C','dV']
X=D[FEAT].fillna(D[FEAT].median()).values; 
out={}
def gcv(y,feats):
    Xf=D[feats].fillna(D[feats].median()).values; p=np.zeros(len(y)); groups=D.cond.values
    for gname in np.unique(groups):
        te=groups==gname
        if y[~te].sum()==0: p[te]=0; continue
        m=GradientBoostingClassifier(n_estimators=150,max_depth=2,learning_rate=0.05,subsample=0.8,random_state=0).fit(Xf[~te],y[~te]); p[te]=m.predict_proba(Xf[te])[:,1]
    return p
for hz in [60,90]:
    y=D[f'knee_within_{hz}'].values; p=gcv(y,FEAT); auc=roc_auc_score(y,p)
    # baseline: capacity-only features
    pb=gcv(y,['soh','dsoh_dt','curv']); aucb=roc_auc_score(y,pb)
    pr=gcv(y,['soh','dsoh_dt','curv','R0_rel','Rmid_rel','Rlow_rel','Rmid10_rel','Rp10ms_rel','HF_rel','MF_rel','MF90_rel']); aucr=roc_auc_score(y,pr)
    D[f'p{hz}']=p
    # lead time: first check-up where p>0.5, per knee cell
    lead=[]
    for cell,g in D[D.days_to_knee.notna()].groupby('cell'):
        g=g.sort_values('days'); fire=g[g[f'p{hz}']>0.5]
        lead.append(dict(cell=cell,T=g['T'].iloc[0],lead_days=fire.days_to_knee.iloc[0] if len(fire) else 0, false_alarm_before=0))
    L=pd.DataFrame(lead)
    # false alarms on non-knee cells: fraction of non-knee cells with any p>0.5
    nk=D[D.days_to_knee.isna()].groupby('cell')[f'p{hz}'].max(); 
    out[f'h{hz}']=dict(auc_all=float(auc),auc_capacity_only=float(aucb),auc_capacity_plus_resistance=float(aucr),n_rows=int(len(y)),n_pos=int(y.sum()),
        lead_days_median=float(L.lead_days.median()),lead_days_by_T=L.groupby('T').lead_days.median().to_dict(),frac_knee_cells_warned=float((L.lead_days>0).mean()),
        false_alarm_cells=float((nk>0.5).mean()),n_nonknee_cells=int(len(nk)))
# ---- exact Shapley (interventional, background = 200 rows) for the 90-day model trained on all data
y=D['knee_within_90'].values; m=GradientBoostingClassifier(n_estimators=150,max_depth=2,learning_rate=0.05,subsample=0.8,random_state=0).fit(X,y)
bg=X[rng.choice(len(X),200,replace=False)]
pos=np.where(y==1)[0]; samp=rng.choice(pos,min(60,len(pos)),replace=False)
nf=len(FEAT); from math import factorial
wts={k:factorial(k)*factorial(nf-k-1)/factorial(nf) for k in range(nf)}
def v(xrow,S):
    Z=bg.copy(); Z[:,list(S)]=xrow[list(S)]; return m.predict_proba(Z)[:,1].mean()
# exact over 2^16 coalitions is 65k model calls per row x 60 rows: too many; use exact Shapley on top-8 features by permutation importance, rest kept at background
from sklearn.inspection import permutation_importance
pi=permutation_importance(m,X,y,n_repeats=10,random_state=0,scoring='roc_auc').importances_mean; top=list(np.argsort(pi)[::-1][:8])
phi=np.zeros((len(samp),nf))
for r,i in enumerate(samp):
    x=X[i]; cache={}
    def val(S):
        key=tuple(sorted(S))
        if key not in cache: cache[key]=v(x,S)
        return cache[key]
    for j in top:
        others=[k for k in top if k!=j]; s=0
        for k in range(len(others)+1):
            for S in itertools.combinations(others,k):
                w=factorial(k)*factorial(len(top)-k-1)/factorial(len(top)); s+=w*(val(set(S)|{j})-val(set(S)))
        phi[r,j]=s
D_s=D.iloc[samp]
shap_mean={FEAT[j]:float(np.abs(phi[:,j]).mean()) for j in top}
shap_cold={FEAT[j]:float(phi[D_s['T'].values<=10,j].mean()) for j in top}; shap_warm={FEAT[j]:float(phi[D_s['T'].values>=25,j].mean()) for j in top}
out['shapley']=dict(mean_abs=shap_mean,mean_cold=shap_cold,mean_warm=shap_warm,permutation_importance={FEAT[j]:float(pi[j]) for j in range(nf)})
# univariate precursor: which feature at 90 days before knee differs most from matched non-knee cells (same T)
pre=D[(D.days_to_knee>60)&(D.days_to_knee<=120)]; non=D[D.days_to_knee.isna()]
uni={}
for f in FEAT[:13]:
    a=pre[f].dropna(); b=non[non['T'].isin(pre['T'].unique())][f].dropna()
    from scipy import stats
    uni[f]=dict(knee_median=float(a.median()),nonknee_median=float(b.median()),p=float(stats.mannwhitneyu(a,b).pvalue))
out['univariate_60_120d_before']=uni
json.dump(out,open('knee_warn.json','w'),indent=1); np.save('knee_shap.npy',phi); D.to_csv('knee_features.csv',index=False)
print(json.dumps(out,indent=1))
