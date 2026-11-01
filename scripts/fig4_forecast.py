import os
import pandas as pd, numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":11.5,"axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":12,"axes.labelsize":11.5,"legend.fontsize":10.5,"xtick.labelsize":10.5,"ytick.labelsize":10.5})
R=json.load(open("twin2.json"))["res"]; E=pd.read_csv("twin2_examples.csv"); T=pd.read_csv("checkups.csv"); T["soh"]=T.cap/3.0
if "cell" not in E.columns: E=E.rename(columns={E.columns[0]:"cell"})
fig,axs=plt.subplots(3,2,figsize=(10.2,13.6)); ax=axs.ravel()
Hs=[100,200,300]; keys=[("linear","Linear extrapolation"),("checkups_only_no_instance","Physics twin, population parameters"),("twin","Physics twin, instance-adapted")]
x=np.arange(3)
for j,(k,nm) in enumerate(keys): ax[0].bar(x+(j-1)*0.26,[R[str(h)][k] for h in Hs],0.26,label=nm)
ax[0].set_xticks(x); ax[0].set_xticklabels([f"{h} days" for h in Hs]); ax[0].set_ylabel("RMSE of future capacity (fraction of nominal)"); ax[0].set_xlabel("History assimilated before forecasting"); ax[0].legend(frameon=False); ax[0].set_title("a  Prospective forecast, held-out cells")
bt=pd.DataFrame({h:R[str(h)]["by_type"] for h in Hs}).T; bt.columns=["Calendar","Cyclic","Drive profile"]; bt.plot.bar(ax=ax[1],rot=0,width=0.7); ax[1].set_xlabel("History (days)"); ax[1].set_ylabel("RMSE (twin)"); ax[1].set_title("b  Error by aging mode"); ax[1].legend(frameon=False)
lp=pd.DataFrame({h:R[str(h)]["Lpl_frac_by_T"] for h in Hs}); lp.index=lp.index.astype(float)
for h in Hs: ax[2].plot(lp.index,lp[h]*100,"o-",ms=5,lw=1.5,label=f"after {h} days")
ax[2].set_xlabel("Cycling temperature (°C)"); ax[2].set_ylabel("Loss attributed to plating route (%)"); ax[2].legend(frameon=False); ax[2].set_title("c  Plating share by temperature")
for i,(c,g) in enumerate(E.groupby("cell")):
    if i>=3: break
    a=ax[3+i]; full=T[T.cell==c].sort_values("days"); a.plot(full.days,full.soh,"k.",ms=4,label="Check-ups")
    a.fill_between(g.days,g.q05,g.q95,color="#c1272d",alpha=0.2,lw=0,label="Monte Carlo 90% band (raw)"); a.plot(g.days,g.pred_soh,color="#c1272d",lw=1.6,label="Twin forecast from day 200"); a.axvline(200,color="#888",ls="--",lw=0.9)
    row=T[T.cell==c].iloc[0]; a.set_title(f"{'def'[i]}  Cell {c[:6]}: {int(row.age_temp)} °C, {row.age_chg_rate:.2f} C charge"); a.set_xlabel("Days"); a.set_ylabel("Capacity / 3.0 Ah"); a.set_ylim(0.4,1.0)
    if i==0: a.legend(frameon=False,loc="lower left")
fig.tight_layout(); fig.savefig(os.environ.get("KIT_OUT",".")+"/fig_twin_real_cells.png",dpi=200); print("fig4 ok")
