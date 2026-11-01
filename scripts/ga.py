import os
import pandas as pd, numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":10.5,"axes.labelsize":10,"legend.fontsize":9,"xtick.labelsize":9,"ytick.labelsize":9})
T=pd.read_csv("checkups.csv"); T["soh"]=T.cap/3.0; hyp=json.load(open("hyp.json")); A=pd.read_csv("policy_eval.csv"); ATT=pd.read_csv("attribution.csv"); R=json.load(open("twin2.json"))["res"]; E=pd.read_csv("twin2_examples.csv"); X=json.load(open("xai.json"))
cm={0:"#2166ac",10:"#67a9cf",25:"#ef8a62",40:"#b2182b"}; ink="#1a1a1a"; acc="#c1272d"
# ------------------------------------------------ graphical abstract
fig=plt.figure(figsize=(15.5,7.6),dpi=220); fig.patch.set_facecolor("white")
def box(x,y,w,h,title,color="#f2f2f2",ec="#888",lw=1):
    fig.patches.append(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.004,rounding_size=0.012",fc=color,ec=ec,lw=lw,transform=fig.transFigure,zorder=-10))
    fig.text(x+w/2,y+h-0.028,title,ha="center",va="top",fontsize=11.0,fontweight="bold",color=ink)
def arrow(x0,y0,x1,y1):
    fig.patches.append(FancyArrowPatch((x0,y0),(x1,y1),transform=fig.transFigure,arrowstyle="-|>",mutation_scale=14,lw=1.4,color="#444",zorder=5))
fig.text(0.5,0.965,"A digital twin reads the hidden degradation routes of 228 Si–graphite cells from in-service check-ups, explains them, and learns to operate around them",ha="center",va="top",fontsize=12.0,fontweight="bold",color=ink)
# 1 fleet
box(0.02,0.08,0.21,0.83,"1  Real fleet (KIT dataset)")
ax=fig.add_axes([0.05,0.52,0.16,0.26]); 
for c,g in T[T.age_type==2].groupby("cell"):
    g=g.sort_values("days"); ax.plot(g.days,g.soh,color=cm[int(g.age_temp.iloc[0])],lw=0.4,alpha=0.6)
ax.set_ylim(0.4,1); ax.set_xlabel("Days",fontsize=10.5); ax.set_ylabel("Capacity / nominal",fontsize=10.5); ax.set_title("144 cycling cells, 0–40 °C",fontsize=11.0,pad=3)
fig.text(0.125,0.44,"228 cells · 44 conditions · 600 days\n4,006 check-ups: capacity, EIS, pulse\n569,675 operando cycle records\nNo post-mortem — the routes are hidden",ha="center",va="top",fontsize=10.5,color=ink,linespacing=1.5)
# 2 twin
box(0.26,0.08,0.24,0.83,"2  Twin: three hidden routes")
fig.text(0.38,0.80,"L = L$_{cal}$ + L$_{cyc}$ + L$_{pl}$",ha="center",fontsize=11.0,color=ink)
fig.text(0.38,0.745,"calendar SEI · cycling SEI · lithium plating\nobserved through capacity and R$_{mid}$",ha="center",va="top",fontsize=10.5,color=ink,linespacing=1.5)
ax=fig.add_axes([0.29,0.36,0.195,0.27]); g=E[E.cell==E.cell.iloc[0]]; full=T[T.cell==E.cell.iloc[0]].sort_values("days")
ax.plot(full.days,full.soh,"k.",ms=2.5,label="check-ups"); k=X["calib_H200"]["factor"]; half=(g.q95-g.q05)/2*k
ax.fill_between(g.days,g["pred_soh"]-half,g["pred_soh"]+half,color=acc,alpha=0.15,lw=0,label="90% band (conformal)"); ax.plot(g.days,g.pred_soh,color=acc,lw=1.2,label="Monte Carlo forecast"); ax.axvline(200,color="#888",ls="--",lw=0.7)
ax.set_ylim(0.4,1.0); ax.set_xlabel("Days",fontsize=10.5); ax.set_title("Synchronize, then fast-forward",fontsize=11.0,pad=3); ax.legend(fontsize=7.5,frameon=False,loc="lower left")
fig.text(0.38,0.25,"EKF synchronization at every check-up\nMonte Carlo rollouts: rate × knee outcome\nForecast error ⅓ of extrapolation at 100 days",ha="center",va="top",fontsize=10.5,color=ink,linespacing=1.5)
# 3 explain
box(0.53,0.08,0.22,0.83,"3  Explain at route level")
ax=fig.add_axes([0.565,0.54,0.17,0.25]); a=ATT.groupby("T")[["calendar","cycling_SEI","plating"]].mean()
b=np.zeros(4); 
for col,cc,lbl in (("calendar","#9ecae1","calendar"),("cycling_SEI","#fdae6b","SEI-type"),("plating",acc,"plating-type")):
    ax.bar(range(4),a[col],bottom=b,color=cc,label=lbl,width=0.65); b+=a[col].values
ax.set_xticks(range(4)); ax.set_xticklabels(["0","10","25","40"]); ax.set_xlabel("Cycling temperature (°C)",fontsize=10.5); ax.set_ylabel("Share of loss",fontsize=10.5); ax.set_title("Exact attribution",fontsize=11.0); ax.set_ylim(0,1.25); ax.legend(fontsize=7.5,frameon=False,loc="upper center",ncol=3)
fig.text(0.64,0.43,"Counterfactuals: narrow window ×1.8–3.8 life\nSensitivity: plating controls life when cold,\nSEI-type when warm",ha="center",va="top",fontsize=10.5,color=ink,linespacing=1.5)
fig.text(0.64,0.29,"Findings\n• calendar loss: √t below ~40 °C / 90 % SoC, linear above\n• knee at 76 ± 4 % capacity, not at a resistance\n• cold cycling: loss without resistance growth\n• sister cells diverge in use, not at the factory",ha="center",va="top",fontsize=10.5,color=ink,linespacing=1.45)
# 4 act
box(0.78,0.08,0.20,0.83,"4  Act: operating map")
ax=fig.add_axes([0.805,0.50,0.15,0.29]); piv=A.pivot_table(index="T",columns=["crate","vwin"],values="cvar20")
cols=[(cr,vw) for cr in (0.33,1.0,1.67) for vw in (1.7,0.84,0.95)]; M=np.array([[piv.loc[t].get((cr,vw),np.nan) if t in piv.index else np.nan for (cr,vw) in cols] for t in (0,10,25,40)])
im=ax.imshow(M/1000,cmap="YlGn",aspect="auto"); ax.set_yticks(range(4)); ax.set_yticklabels(["0 °C","10 °C","25 °C","40 °C"]); ax.set_xticks(range(9)); ax.set_xticklabels([f"{cr:g}C\n{'full' if vw==1.7 else ('narrow' if vw==0.84 else 'mid')}" for cr,vw in cols],fontsize=7.0,rotation=90)
for i,t in enumerate((0,10,25,40)):
    j=int(np.nanargmax(M[i])); ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,fill=False,ec=acc,lw=1.6))
ax.set_title("Risk-adjusted throughput to EOL (kAh)",fontsize=10.5); cb=fig.colorbar(im,ax=ax,fraction=0.05,pad=0.02); cb.ax.tick_params(labelsize=5.5)
fig.text(0.88,0.36,"400 rollouts per action, CVaR$_{20}$ objective\nRecommendation = observed best at 3 of 4 T\nrank correlation 0.75 vs held-out cells\n\nNarrow the window everywhere;\nslow the charge only when cold",ha="center",va="top",fontsize=10.5,color=ink,linespacing=1.45)
for x0,x1 in ((0.23,0.26),(0.50,0.53),(0.75,0.78)): arrow(x0,0.5,x1,0.5)
[a.set_zorder(10) for a in fig.axes]
fig.savefig(os.environ.get("KIT_OUT",".")+"/graphical_abstract.png",bbox_inches="tight",facecolor="white"); print("GA saved")
# ------------------------------------------------ Fig 5: operating map + observed, and per-cell dashboard
fig=plt.figure(figsize=(10.2,10.5)); gs=fig.add_gridspec(2,2,height_ratios=[1.15,1],hspace=0.55,wspace=0.3); axs=[fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]
plt.rcParams.update({"font.size":11.5,"axes.titlesize":12,"axes.labelsize":11.5,"legend.fontsize":10.5,"xtick.labelsize":10.5,"ytick.labelsize":10.5})
ax=axs[0]; im=ax.imshow(M/1000,cmap="YlGn",aspect="auto"); ax.set_yticks(range(4)); ax.set_yticklabels(["0","10","25","40"]); ax.set_ylabel("Temperature (°C)"); ax.set_xticks(range(9)); ax.set_xticklabels([f"{cr:g} C, {'2.5–4.2' if vw==1.7 else ('3.25–4.09' if vw==0.84 else '3.25–4.2')} V" for cr,vw in cols],rotation=35,ha="right",fontsize=10.5)
for i,t in enumerate((0,10,25,40)):
    j=int(np.nanargmax(M[i])); ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,fill=False,ec=acc,lw=1.8))
    for jj,(cr,vw) in enumerate(cols):
        r=A[(A["T"]==t)&np.isclose(A.crate,cr)&np.isclose(A.vwin,vw)]
        if len(r) and np.isfinite(r.Ah_obs.iloc[0]): ax.text(jj,i,f"{r.Ah_obs.iloc[0]/1000:.1f}",ha="center",va="center",fontsize=11,color="k")
plt.colorbar(im,ax=ax,fraction=0.046,label="Predicted CVaR$_{20}$ throughput (kAh)"); ax.set_title("a  Operating map: predicted CVaR₂₀ throughput; numbers = observed kAh to end of life; red = twin's pick")
from scipy.stats import spearmanr
ax=axs[1]; m=A.Ah_obs.notna(); rho=spearmanr(A.mean_Ah[m],A.Ah_obs[m]).statistic; ax.scatter(A.mean_Ah[m]/1000,A.Ah_obs[m]/1000,c=[cm[int(t)] for t in A["T"][m]],s=40,edgecolor="k",lw=0.4); lim=max(A.mean_Ah.max(),A.Ah_obs.max())/1000*1.05; ax.plot([0,lim],[0,lim],"k--",lw=0.7); ax.set_xlabel("Predicted mean throughput to EOL (kAh)"); ax.set_ylabel("Observed in held-out cells (kAh)"); ax.set_title(f"b  Predicted against observed, {int(m.sum())} actions (ρ = {rho:.2f})")
ax=axs[2]; lp=A.pivot_table(index="T",columns="vwin",values="plating_share"); lp.columns=["3.25–4.09 V","3.25–4.2 V","2.5–4.2 V"]; lp.plot(ax=ax,marker="o",ms=6,lw=1.4); ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Plating-type share of loss"); ax.set_title("c  Plating share of loss by window and temperature"); ax.legend(frameon=False,title="window")
from matplotlib.lines import Line2D
axs[1].legend(handles=[Line2D([],[],marker="o",ls="",color=cm[t],markeredgecolor="k",label=f"{t} °C") for t in (0,10,25,40)],frameon=False,loc="upper left")
fig.savefig(os.environ.get("KIT_OUT",".")+"/fig5_operating_map.png",dpi=200,bbox_inches="tight"); print("fig5 saved")
