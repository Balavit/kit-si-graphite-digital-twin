import numpy as np, pandas as pd, json, time
src=open("twin2.py").read().split("# ---------------- instance twin")[0]
exec(src)   # fits TH on train cells (≈45 s)
H2=pd.read_csv("H2_knee.csv"); H2tr=H2[~H2.cell.isin(test_cells)]
# ---- knee stage fitted on knee cells of the TRAIN split only
kn=H2tr[H2tr.knee]; p_knee={t: float(((H2tr["T"]==t)&H2tr.knee).sum()/max((H2tr["T"]==t).sum(),1)) for t in (0,10,25,40)}
Lstar=float(1-kn.soh_knee.mean()); G=float(np.median(kn.slope_ratio.clip(1,8)))
print("knee stage from train knee cells: n=%d  L*=%.3f  acceleration g=%.2f  p_knee by T="%(len(kn),Lstar,G),{k:round(v,2) for k,v in p_knee.items()})
def pop_inc(f, d, q, d0, q0):
    Ls,Lp,_=model(TH,[0,0,0],d,q,f); Ls0,Lp0,_=model(TH,[0,0,0],d0,q0,f); return (Ls+Lp)-(Ls0+Lp0)
def forecast(c, g, H, nmc=300, rng=None):
    """Population twin + level correction + probabilistic knee mixture; returns per-future-checkup mean, q05, q95, p_knee_post."""
    f=feats(c); q0=g.cap.iloc[0]; p=g[g.days<=H]; fu=g[g.days>H]; LH=1-p.cap.values[-1]/q0; dH,qH=p.days.values[-1],p.q_chg.values[-1]
    Tc=int(cells.loc[c].age_temp); prior=p_knee.get(Tc,0.2) if cells.loc[c].age_type!=1 else 0.0
    # posterior on 'kneed already': compare observed recent increment with model inc with/without g, if L_H > L*
    pk=prior
    if LH>Lstar and len(p)>=3:
        w=p[p.days>=H-90] if (p.days>=H-90).sum()>=2 else p.iloc[-2:]
        inc_obs=(1-w.cap.values[-1]/q0)-(1-w.cap.values[0]/q0); inc_mod=pop_inc(f,w.days.values[-1],w.q_chg.values[-1],w.days.values[0],w.q_chg.values[0])
        s=0.0113; l0=np.exp(-0.5*((inc_obs-inc_mod)/s)**2); l1=np.exp(-0.5*((inc_obs-G*inc_mod)/s)**2); pk=prior*l1/(prior*l1+(1-prior)*l0+1e-12)
    inc=pop_inc(f,fu.days.values,fu.q_chg.values,dH,qH)
    # trajectories: no-knee: L=LH+inc ; knee: once L crosses L*, remaining increments ×g
    def traj(knee, scale):
        L=LH+inc*scale
        if knee:
            L=np.empty_like(inc); prev=LH; prev_inc=0.0
            for i,x in enumerate(inc):
                di=(x-prev_inc)*scale; prev_inc=x
                prev=prev+di*(G if prev>=Lstar else 1.0); L[i]=prev
        return L
    rng=rng or np.random.default_rng(0)
    S=[]
    for _ in range(nmc):
        k=rng.uniform()<pk; sc=np.exp(rng.normal(0,0.12)); S.append(traj(k,sc))
    S=np.array(S); mean=S.mean(0); lo,hi=np.percentile(S,[5,95],axis=0)
    return pd.DataFrame(dict(days=fu.days.values,true=1-fu.cap.values/q0,mean=mean,q05=lo,q95=hi,noknee=traj(False,1.0),p_knee=pk)), pk
rows=[]
for H in (100,200,300):
    for c,g in test.groupby("cell"):
        if g.days.max()<H+40 or (g.days<=H).sum()<3: continue
        F,pk=forecast(c,g,H)
        rows.append(dict(H=H,cell=c,atype=cells.loc[c].age_type,T=cells.loc[c].age_temp,rmse_mix=np.sqrt(((F["mean"]-F.true)**2).mean()),rmse_nok=np.sqrt(((F.noknee-F.true)**2).mean()),cov=((F.true>=F.q05)&(F.true<=F.q95)).mean(),p_knee=pk,kneed=bool(H2.set_index("cell").knee.get(c,False)),end_err_mix=abs(F["mean"].iloc[-1]-F.true.iloc[-1]),end_err_nok=abs(F.noknee.iloc[-1]-F.true.iloc[-1])))
E=pd.DataFrame(rows); E.to_csv("knee_eval.csv",index=False)
print(E.groupby("H")[["rmse_mix","rmse_nok","end_err_mix","end_err_nok","cov"]].mean().round(4).to_string())
print("cycling cells only:"); print(E[E.atype==2].groupby("H")[["rmse_mix","rmse_nok","cov"]].mean().round(4).to_string())
print("knee cells only:"); print(E[E.kneed].groupby("H")[["rmse_mix","rmse_nok","end_err_mix","end_err_nok","cov","p_knee"]].mean().round(4).to_string())
# ---------------- RL-style offline policy evaluation: choose charging condition per temperature
# action = (charge rate, voltage window); objective = expected Ah throughput before 70 % capacity within 600 days, risk-adjusted (CVaR 20 %)
EOL=0.30
def life_mc(T_c, crate, vwin, prof, nmc=400, rng=np.random.default_rng(1)):
    f=dict(Tk=T_c+273.15,soc=0.8,crate=crate,vwin=vwin,prof=prof); days=np.arange(0,601,5.0)
    # throughput per day at this condition from the actual fleet (cycles/day) ~ proportional to charge rate; estimate from train cells of that condition
    sel=train[(train.age_type==2)&(train.age_temp==T_c)&(np.isclose(train.age_chg_rate,crate))&(np.isclose(train.V_max_cyc_V-train.V_min_cyc_V,vwin,atol=0.02))]
    if len(sel)<3: return None
    qpd=float((sel.groupby("cell").apply(lambda g: g.q_chg.max()/g.days.max())).median()); q=days*qpd
    Ls,Lp,_=model(TH,[0,0,0],days,q,f); base=Ls+Lp; pk=p_knee.get(int(T_c),0.2); out=[]
    for _ in range(nmc):
        sc=np.exp(rng.normal(0,0.12)); k=rng.uniform()<pk; L=np.zeros_like(base); prev=0; pinc=0
        for i,x in enumerate(base):
            di=(x-pinc)*sc; pinc=x; prev=prev+di*(G if (k and prev>=Lstar) else 1.0); L[i]=prev
        idx=np.argmax(L>=EOL) if (L>=EOL).any() else len(L)-1; out.append(q[idx])
    out=np.array(out); return dict(T=T_c,crate=crate,vwin=round(vwin,2),mean_Ah=out.mean(),cvar20=np.sort(out)[:nmc//5].mean(),qpd=qpd,n_cells=sel.cell.nunique(),plating_share=float(Lp[-1]/(Ls[-1]+Lp[-1])))
acts=[]
for T_c in (0,10,25,40):
    for cr in (0.33,1.0,1.67):
        for vw in (1.7,0.84,0.95):
            r=life_mc(T_c,cr,vw,0.0)
            if r: acts.append(r)
A=pd.DataFrame(acts)
# empirical check: observed throughput at 70 % capacity (or at end if not reached) for test cells of each action
def obs_life(g):
    q0=g.cap.iloc[0]; L=1-g.cap/q0; hit=np.where(L>=EOL)[0]
    return g.q_chg.iloc[hit[0]] if len(hit) else np.nan
obs=test[test.age_type==2].groupby("cell").apply(lambda g: pd.Series(dict(T=g.age_temp.iloc[0],crate=g.age_chg_rate.iloc[0],vwin=round(g.V_max_cyc_V.iloc[0]-g.V_min_cyc_V.iloc[0],2),Ah_obs=obs_life(g),reached=np.isfinite(obs_life(g)),Ah_end=g.q_chg.max(),soh_end=g.soh.iloc[-1]))).reset_index()
A=A.merge(obs.groupby(["T","crate","vwin"]).agg(Ah_obs=("Ah_obs","mean"),reached=("reached","mean"),Ah_end=("Ah_end","mean")).reset_index(),on=["T","crate","vwin"],how="left")
A.to_csv("policy_eval.csv",index=False)
print(A.round(2).to_string())
from scipy.stats import spearmanr
m=A.Ah_obs.notna(); print("Spearman(predicted mean Ah to EOL, observed) over actions with observed EOL: rho=%.2f n=%d"%(spearmanr(A.mean_Ah[m],A.Ah_obs[m]).statistic,m.sum()))
for T_c in (0,10,25,40):
    a=A[A["T"]==T_c]; 
    if len(a): print("T=%d: twin picks (CVaR) crate=%.2f vwin=%.2f ; observed best among reached: %s"%(T_c,a.loc[a.cvar20.idxmax(),"crate"],a.loc[a.cvar20.idxmax(),"vwin"], a.loc[a.Ah_obs.idxmax(),["crate","vwin","Ah_obs"]].round(2).to_dict() if a.Ah_obs.notna().any() else "none reached EOL"))
json.dump(dict(Lstar=Lstar,G=G,p_knee=p_knee),open("knee.json","w"))
