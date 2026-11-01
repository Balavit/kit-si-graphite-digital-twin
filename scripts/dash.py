import os
import numpy as np, pandas as pd, json
src=open("knee_rl.py").read().split("rows=[]\nfor H in (100,200,300):")[0]
exec(src)   # TH, knee stage, forecast(), train/test, cells, feats, model
K=json.load(open("xai.json")); kfac=K["calib_H200"]["factor"]
def trace(c, g, H=200):
    """Population twin with level correction: filtered total loss, routes, R_mid prediction vs observation, knee posterior; then forecast."""
    f=feats(c); q0=g.cap.iloc[0]; r0=g.Rmid_s50.dropna().iloc[0]
    rows=[]
    for _,r in g.iterrows():
        cal,cyc,pl=[np.atleast_1d(v)[0] for v in (lambda a,b,c_: (a,b,c_))(*[x for x in (lambda: (model(TH,[0,0,0],r.days,r.q_chg,f)[0],))()][0:1]*0+list(model(TH,[0,0,0],r.days,r.q_chg,f)[:2])+[0])] if False else (None,None,None)
        Ls,Lp,Rr=model(TH,[0,0,0],r.days,r.q_chg,f)
        calr=cal_route(TH,f,r.days)
        rows.append(dict(days=r.days,obs_loss=1-r.cap/q0,mod_loss=Ls+Lp,L_cal=calr,L_cyc=Ls-calr,L_pl=Lp,obs_R=r.Rmid_s50/r0 if np.isfinite(r.Rmid_s50) else np.nan,mod_R=Rr,obs_Rp=r.Rp1s_s50/g.Rp1s_s50.dropna().iloc[0] if np.isfinite(r.Rp1s_s50) else np.nan))
    D=pd.DataFrame(rows); D["level_corr"]=D.obs_loss-D.mod_loss   # the level correction the filter applies at each check-up
    F,pk=forecast(c,g,H,nmc=300); F["half"]=(F.q95-F.q05)/2*kfac
    return D,F,pk
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":14,"axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":14,"axes.labelsize":14,"legend.fontsize":12.5,"xtick.labelsize":12.5,"ytick.labelsize":12.5})
# pick four held-out cells: calendar 40°C/100%, cycling 25°C full window, cycling 0°C narrow (knee), profile
pick=[]
for want in [dict(age_type=1,age_temp=40,age_soc=100),dict(age_type=2,age_temp=25,age_chg_rate=1.0),dict(age_type=2,age_temp=0,age_chg_rate=0.33),dict(age_type=3,age_temp=25)]:
    cand=[c for c in test_cells if all(np.isclose(cells.loc[c][k],v) for k,v in want.items()) and T[T.cell==c].days.max()>400]
    if cand: pick.append(cand[0])
fig,axs=plt.subplots(len(pick),3,figsize=(15.5,4.1*len(pick)))
for i,c in enumerate(pick):
    g=T[T.cell==c].sort_values("days"); D,F,pk=trace(c,g); row=cells.loc[c]
    lbl={1:f"calendar {int(row.age_temp)} °C, {int(row.age_soc)} % SoC",2:f"cycling {int(row.age_temp)} °C, {row.age_chg_rate:.2f} C, {row.V_min_cyc_V:.2f}–{row.V_max_cyc_V:.2f} V",3:f"drive profile {int(row.age_temp)} °C"}[int(row.age_type)]
    ax=axs[i,0]; ax.stackplot(D.days,D.L_cal,D.L_cyc,D.L_pl,colors=["#9ecae1","#fdae6b","#c1272d"],labels=["calendar","SEI-type","plating-type"],alpha=0.9); ax.plot(D.days,D.obs_loss,"k.",ms=3,label="observed loss")
    ax.set_ylabel("Capacity loss"); ax.set_title(f"{'abcdefghijkl'[3*i]}  {c[:6]}: {lbl} — inferred routes"); 
    if i==0: ax.legend(fontsize=12.5,frameon=False,loc="upper left")
    ax=axs[i,1]; ax.plot(D.days,D.obs_R,"o",ms=3,color="#444",label="EIS R$_{mid}$ (check-up)"); ax.plot(D.days,D.mod_R,"-",color="#fdae6b",lw=1.3,label="twin: from SEI-type routes"); ax.plot(D.days,D.obs_Rp,"x",ms=3,color="#999",label="1 s pulse R (independent)")
    ax.set_ylabel("Resistance / initial"); ax.set_title(f"{'abcdefghijkl'[3*i+1]}  second channel: resistance"); 
    if i==0: ax.legend(fontsize=12.5,frameon=False,loc="upper left")
    ax=axs[i,2]; ax.plot(g.days,g.soh,"k.",ms=3,label="check-ups"); ax.fill_between(F.days,1-F["mean"]-F.half,1-F["mean"]+F.half,color="#c1272d",alpha=0.15,lw=0,label="90 % band (conformal)"); ax.plot(F.days,1-F["mean"],color="#c1272d",lw=1.2,label="forecast from day 200"); ax.axvline(200,color="#888",ls="--",lw=0.7)
    ax.set_ylim(0.3,1.0); ax.set_ylabel("Capacity / nominal"); ax.set_title(f"{'abcdefghijkl'[3*i+2]}  forecast; posterior knee probability {pk:.2f}")
    if i==0: ax.legend(fontsize=12.5,frameon=False,loc="lower left")
    if i==len(pick)-1:
        for a in axs[i]: a.set_xlabel("Days")
fig.tight_layout(); fig.savefig(os.environ.get("KIT_OUT",".")+"/fig6_cell_dashboard.png",dpi=200); print("picked",pick)
