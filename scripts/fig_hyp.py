import os
import pandas as pd, numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":11.5,"axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":12,"axes.labelsize":11.5,"legend.fontsize":10.5,"xtick.labelsize":10.5,"ytick.labelsize":10.5})
T = pd.read_csv("checkups.csv"); T["soh"]=T.cap/3.0; T=T.sort_values(["cell","days"])
H1=pd.read_csv("H1_calendar.csv"); H2=pd.read_csv("H2_knee.csv"); H3=pd.read_csv("H3_cold.csv"); H4=pd.read_csv("H4_var.csv"); hyp=json.load(open("hyp.json"))
cm={0:"#2166ac",10:"#67a9cf",25:"#ef8a62",40:"#b2182b"}
fig,axs=plt.subplots(3,2,figsize=(10.2,13.6)); ax=axs.ravel()
# a: regime map of the calendar exponent (fit with free offset), per condition
K=pd.read_csv("calendar_conditions.csv"); Ts=[0,10,25,40]; Ss=[10,50,90,100]
M=np.array([[K[(K["T"]==t)&(K.SoC==sc)].n.values[0] for sc in Ss] for t in Ts])
im=ax[0].imshow(M,cmap="coolwarm",vmin=0.3,vmax=1.2,origin="lower",aspect="auto")
for i,t in enumerate(Ts):
    for j,sc in enumerate(Ss): ax[0].text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",fontsize=9.5,color="k",fontweight="bold" if M[i,j]>0.75 else "normal")
ax[0].set_xticks(range(4)); ax[0].set_xticklabels([f"{s}%" for s in Ss]); ax[0].set_yticks(range(4)); ax[0].set_yticklabels([f"{t} °C" for t in Ts])
ax[0].set_xlabel("Storage state of charge"); ax[0].set_ylabel("Storage temperature"); ax[0].spines["top"].set_visible(True); ax[0].spines["right"].set_visible(True)
plt.colorbar(im,ax=ax[0],fraction=0.046,pad=0.03,label="time exponent n (loss = L₀ + k tⁿ)")
ax[0].set_title("a  Two calendar regimes: √t (n ≈ 0.5) and linear (n ≈ 1)")
# b: example condition-mean trajectories with sqrt and linear fits
from scipy.optimize import curve_fit
cal=T[T.age_type==1]
for (tt,sc),col in [((25,50),cm[25]),((40,50),"#e08a6a"),((40,100),cm[40])]:
    g=cal[(cal.age_temp==tt)&(cal.age_soc==sc)]; piv=g.pivot_table(index=g.days.round(0),columns="cell",values="cap"); m=piv.mean(axis=1); d=m.index.values; L=1-m.values/m.values[0]; mk=d>1
    ps,_=curve_fit(lambda t,a,k:a+k*np.sqrt(t),d[mk],L[mk]); pl,_=curve_fit(lambda t,a,k:a+k*t,d[mk],L[mk]); tt_=np.linspace(7,600,200)
    ax[1].plot(d,L*100,"o",ms=3.5,color=col,label=f"{tt} °C, {sc}% SoC")
    ax[1].plot(tt_,(ps[0]+ps[1]*np.sqrt(tt_))*100,"-",lw=1,color=col); ax[1].plot(tt_,(pl[0]+pl[1]*tt_)*100,"--",lw=1,color=col)
ax[1].plot([],[],"k-",lw=1,label="L₀ + k√t"); ax[1].plot([],[],"k--",lw=1,label="L₀ + k t")
ax[1].set_xlabel("Days"); ax[1].set_ylabel("Calendar capacity loss (%)"); ax[1].legend(fontsize=10,frameon=False); ax[1].set_title("b  √t below the boundary, linear above it")
# c: knee: SoH at knee vs Rmid rel at knee
kn=H2[H2.knee]
ax[2].scatter(kn.rmid_knee_rel,kn.soh_knee,c=[cm[int(t)] for t in kn["T"]],s=28,edgecolor="k",lw=0.4)
ax[2].axhline(kn.soh_knee.mean(),color="k",ls="--",lw=0.8); ax[2].set_xlabel("Mid-frequency resistance at knee / initial"); ax[2].set_ylabel("Capacity at knee / 3.0 Ah"); ax[2].set_ylim(0.6,0.9); ax[2].set_title(f"c  Knee: capacity CV {hyp['H2']['soh_at_knee']['cv']*100:.0f}%, resistance CV {hyp['H2']['rmid_rel_at_knee']['cv']*100:.0f}%")
for t in (0,10,25): ax[2].scatter([],[],color=cm[t],label=f"{t} °C: {hyp['H2']['knee_by_T'][str(float(t))]*100:.0f}% of cells knee")
ax[2].scatter([],[],color=cm[40],label="40 °C: 0% knee"); ax[2].legend(fontsize=10,frameon=False,loc="lower right")
# d: dR/dQ by T (plating signature)
g=H3.groupby("T").dR_dQ
ax[3].boxplot([g.get_group(t).values for t in (0,10,25,40)],tick_labels=["0","10","25","40"],patch_artist=True,boxprops=dict(facecolor="#ddd"),medianprops=dict(color="k"))
ax[3].set_xlabel("Cycling temperature (°C)"); ax[3].set_ylabel("Resistance growth per unit capacity loss"); ax[3].set_title(f"d  Cold cycling: capacity loss without resistance growth (p = {hyp['H3']['kruskal_dRdQ_T']:.0e})",fontsize=11)
# e: CE by T and C-rate
ce=pd.DataFrame(hyp["H3"]["CE_median_by_T_chg"]); ce.index=ce.index.astype(float)
for cr in ce.columns: ax[4].plot(ce.index,ce[cr],"o-",ms=4,lw=1,label=f"{float(cr):.2f} C charge")
ax[4].set_xlabel("Cycling temperature (°C)"); ax[4].set_ylabel("Median coulombic efficiency (%)"); ax[4].legend(fontsize=10,frameon=False); ax[4].set_title("e  Coulombic efficiency falls when cold and fast")
# f: variability decomposition
w=H4.soh300-H4.cond.map(H4.groupby("cond").soh300.mean()); l60=H4.loss60-H4.cond.map(H4.groupby("cond").loss60.mean())
ax[5].scatter(l60*100,w*100,s=12,color="#444",alpha=0.7); ax[5].set_xlabel("Loss in first 60 days, relative to condition mean (%)"); ax[5].set_ylabel("Capacity at 300 days, relative to condition mean (%)")
ax[5].set_title(f"f  Within-condition fate is set by the early rate (R² = {hyp['H4']['within_r2_loss60_only']:.2f})")
fig.tight_layout(); fig.savefig(os.environ.get("KIT_OUT",".")+"/fig_findings_H1-H4.png",dpi=200)
