import os
"""Step 5: calibration of Monte Carlo bands (conformal, fitted on train cells) + explainability analyses
(term attribution, counterfactual rollouts, sensitivity of lifetime to hidden-state routes)."""
import numpy as np, pandas as pd, json
src=open("knee_rl.py").read().split("rows=[]\nfor H in (100,200,300):")[0]
exec(src)   # fits TH, knee stage; defines model, feats, forecast, train/test, p_knee, Lstar, G
OUT={}
# ---------- 1. conformal calibration of rollout bands on TRAIN cells, applied to TEST cells
def band_scores(D, H):
    sc=[]
    for c,g in D.groupby("cell"):
        if g.days.max()<H+40 or (g.days<=H).sum()<3: continue
        F,_=forecast(c,g,H,nmc=200); half=(F.q95-F.q05)/2+1e-4; sc.append(np.abs(F.true-F["mean"])/half)
    return np.concatenate(sc)
calib={}
for H in (100,200,300):
    s=band_scores(train,H); k=float(np.quantile(s,0.9)); calib[H]=k
    rows=[]
    for c,g in test.groupby("cell"):
        if g.days.max()<H+40 or (g.days<=H).sum()<3: continue
        F,_=forecast(c,g,H,nmc=200); half=(F.q95-F.q05)/2*k; rows.append(dict(covg=((F.true>=F["mean"]-half)&(F.true<=F["mean"]+half)).mean(), width=float((2*half).mean()), atype=cells.loc[c].age_type))
    r=pd.DataFrame(rows); OUT[f"calib_H{H}"]=dict(factor=k, coverage_raw=None, coverage_conformal=float(r.covg.mean()), mean_width=float(r.width.mean()), by_type=r.groupby("atype").covg.mean().round(3).to_dict())
    print("H",H,"conformal factor %.2f  test coverage %.3f  width %.3f"%(k,r.covg.mean(),r.width.mean()),flush=True)
# ---------- 2. term attribution of capacity loss per condition at 300 days (exact: the model is a sum of routes)
def routes(f, d, q):
    lnkc,Ec,s_soc,lnkl,El,lnkq,Eq,s_c,s_w,lnkp,Ep,s_cp,Ls_,lng,s_prof,a,b,s_gT=TH; iT=1/(R*f["Tk"])-1/(R*298.15)
    cal=cal_route(TH,f,d); cyc=np.exp(lnkq+Eq*(-iT)+s_c*f["crate"]+s_w*f["vwin"]+s_prof*f["prof"])*q/1000; pl=np.exp(lnkp+Ep*iT+s_cp*f["crate"])*q/1000
    return cal,cyc,pl
att=[]
for cond,g in T[T.age_type==2].groupby("cond"):
    c=g.cell.iloc[0]; f=feats(c); gg=g[g.cell==c]; d=min(300,gg.days.max()); q=np.interp(d,gg.days,gg.q_chg)
    cal,cyc,pl=routes(f,d,q); tot=cal+cyc+pl
    att.append(dict(cond=cond,T=cells.loc[c].age_temp,crate=cells.loc[c].age_chg_rate,vwin=round(cells.loc[c].V_max_cyc_V-cells.loc[c].V_min_cyc_V,2),calendar=cal/tot,cycling_SEI=cyc/tot,plating=pl/tot,total=tot))
ATT=pd.DataFrame(att); ATT.to_csv("attribution.csv",index=False); OUT["attribution_by_T"]=ATT.groupby("T")[["calendar","cycling_SEI","plating"]].mean().round(3).to_dict()
# ---------- 3. counterfactual rollouts: same cell, one operating variable changed (population model, 400-day horizon)
def life_days(f, qpd, horizon=600):
    d=np.arange(0,horizon+1,5.0); cal,cyc,pl=routes(f,d,d*qpd); L=cal+cyc+pl; hit=np.where(L>=0.30)[0]; return float(d[hit[0]]) if len(hit) else float("inf"), L
CF=[]
for T_c in (0,10,25,40):
    for cr in (0.33,1.0,1.67):
        for vw,lbl in ((1.7,"full"),(0.84,"narrow")):
            f=dict(Tk=T_c+273.15,soc=0.8,crate=cr,vwin=vw,prof=0.0); sel=train[(train.age_type==2)&(train.age_temp==T_c)&np.isclose(train.age_chg_rate,cr)&np.isclose(train.V_max_cyc_V-train.V_min_cyc_V,vw,atol=0.02)]
            qpd=float((sel.groupby("cell").apply(lambda g: g.q_chg.max()/g.days.max())).median()) if len(sel) else np.nan
            if not np.isfinite(qpd): continue
            base,_=life_days(f,qpd)
            cf={}
            for nm,f2 in (("+10 K",dict(f,Tk=f["Tk"]+10)),("-10 K",dict(f,Tk=f["Tk"]-10)),("rate 0.33 C",dict(f,crate=0.33)),("narrow window" if lbl=="full" else "full window",dict(f,vwin=0.84 if lbl=="full" else 1.7))):
                l,_=life_days(f2,qpd); cf[nm]=l/base if np.isfinite(base) and np.isfinite(l) else np.nan
            CF.append(dict(T=T_c,crate=cr,window=lbl,life_days=base,**cf))
CF=pd.DataFrame(CF); CF.to_csv("counterfactuals.csv",index=False); print(CF.round(2).to_string())
# ---------- 4. sensitivity of lifetime to each route (Sobol-like first-order indices, log-normal perturbation of each route rate, sd 0.3) across conditions
S=[]
rng=np.random.default_rng(3)
for _,r in CF.iterrows():
    f=dict(Tk=r["T"]+273.15,soc=0.8,crate=r.crate,vwin=(1.7 if r.window=="full" else 0.84),prof=0.0); sel=train[(train.age_type==2)&(train.age_temp==r["T"])&np.isclose(train.age_chg_rate,r.crate)&np.isclose(train.V_max_cyc_V-train.V_min_cyc_V,f["vwin"],atol=0.02)]
    qpd=float((sel.groupby("cell").apply(lambda g: g.q_chg.max()/g.days.max())).median()); d=np.arange(0,601,5.0); cal,cyc,pl=routes(f,d,d*qpd)
    N=400; Z=rng.normal(0,0.3,(N,3)); lives=[]
    for z in Z:
        L=cal*np.exp(z[0])+cyc*np.exp(z[1])+pl*np.exp(z[2]); hit=np.where(L>=0.30)[0]; lives.append(d[hit[0]] if len(hit) else 650)
    lives=np.array(lives); var=lives.var()+1e-9
    idx=[np.var([lives[np.argsort(Z[:,j])[k*40:(k+1)*40]].mean() for k in range(10)])/var for j in range(3)]   # binned first-order index
    S.append(dict(T=r["T"],crate=r.crate,window=r.window,S_calendar=idx[0],S_cycling_SEI=idx[1],S_plating=idx[2]))
S=pd.DataFrame(S); S.to_csv("sensitivity.csv",index=False); OUT["sensitivity_by_T"]=S.groupby("T")[["S_calendar","S_cycling_SEI","S_plating"]].mean().round(3).to_dict()
print(S.groupby(["T","window"])[["S_calendar","S_cycling_SEI","S_plating"]].mean().round(2).to_string())
json.dump(OUT,open("xai.json","w"),indent=1,default=float)
# ---------- figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":11.5,"axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":12,"axes.labelsize":11.5,"legend.fontsize":10.5,"xtick.labelsize":10.5,"ytick.labelsize":10.5})
fig,axs=plt.subplots(2,2,figsize=(10.4,9.4)); ax=axs.ravel()
a=ATT.groupby("T")[["calendar","cycling_SEI","plating"]].mean(); a.index=[f"{int(t)}" for t in a.index]; a.plot.bar(ax=ax[0],stacked=True,rot=0,color=["#9ecae1","#fdae6b","#c1272d"],width=0.6); ax[0].set_xlabel("Cycling temperature (°C)"); ax[0].set_ylabel("Share of capacity loss at 300 days"); ax[0].set_title("a  Exact route attribution at 300 days"); ax[0].legend(["Calendar","Cycling, SEI-type","Cycling, plating-type"],fontsize=10.5,frameon=False,loc="lower left")
piv=CF[CF.window=="full"].pivot(index="T",columns="crate",values="narrow window"); piv.plot.bar(ax=ax[1],rot=0,width=0.7); ax[1].axhline(1,color="k",lw=0.6); ax[1].set_xlabel("Temperature (°C)"); ax[1].set_ylabel("Lifetime ratio, narrow / full window"); ax[1].set_title("b  Counterfactual: narrow window instead of full"); ax[1].legend(title="charge rate (C)",fontsize=10.5,frameon=False)
piv2=CF.pivot_table(index="T",columns="window",values="+10 K"); piv2.plot.bar(ax=ax[2],rot=0,width=0.6,color=["#555","#bbb"]); ax[2].axhline(1,color="k",lw=0.6); ax[2].set_xlabel("Temperature (°C)"); ax[2].set_ylabel("Lifetime ratio, +10 K"); ax[2].set_title("c  Counterfactual: 10 K warmer"); ax[2].legend(["full window","narrow window"],fontsize=10.5,frameon=False)
s=S.groupby("T")[["S_calendar","S_cycling_SEI","S_plating"]].mean(); s.plot.bar(ax=ax[3],rot=0,width=0.7,color=["#9ecae1","#fdae6b","#c1272d"]); ax[3].set_xlabel("Temperature (°C)"); ax[3].set_ylabel("First-order sensitivity of lifetime"); ax[3].set_title("d  Which route controls lifetime (variance share)"); ax[3].legend(["Calendar","SEI-type","Plating-type"],fontsize=10.5,frameon=False)
fig.tight_layout(); fig.savefig(os.environ.get("KIT_OUT",".")+"/fig_xai_materials.png",dpi=200); print("fig saved")
