import os
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":11.5,"axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":12,"axes.labelsize":11.5,"legend.fontsize":10.5,"xtick.labelsize":10.5,"ytick.labelsize":10.5})
d=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','simulation_benchmark','results','main_accuracy.csv'))
ORDER=["EKF","UKF","EnKF","PF","JUKF","GBM","LSTM","GRU","TCN","Transformer","PI-GRU","PRISM-TRACE"]; ORDER=[m for m in ORDER if m in set(d.method)]
LAB={"JUKF":"Joint UKF","PI-GRU":"PI-GRU"}
COL={"EKF":"#9ecae1","UKF":"#6baed6","EnKF":"#4292c6","PF":"#2171b5","JUKF":"#084594","GBM":"#c7c7c7","LSTM":"#a1d99b","GRU":"#74c476","TCN":"#41ab5d","Transformer":"#238b45","PI-GRU":"#e6a141","PRISM-TRACE":"#c1272d"}
g=d.groupby(["method","fleet"]).nrmse.agg(["mean","std"])
fig=plt.figure(figsize=(10.2,11.5)); gs=fig.add_gridspec(2,2,height_ratios=[1,1.15],hspace=0.35,wspace=0.5)
ax=fig.add_subplot(gs[0,:]); x=np.arange(len(ORDER)); w=0.38
for k,(fl,h) in enumerate([("ID",""),("Stress","//")]):
    m=[g.loc[(o,fl),"mean"] for o in ORDER]; s=[g.loc[(o,fl),"std"] if not np.isnan(g.loc[(o,fl),"std"]) else 0 for o in ORDER]
    ax.bar(x+(k-0.5)*w,m,w,color=[COL[o] for o in ORDER],hatch=h,edgecolor="k" if h else "none",lw=0.5,yerr=s,capsize=2,label=f"{'in-distribution' if fl=='ID' else 'stress'} test fleet")
ax.set_xticks(x); ax.set_xticklabels([LAB.get(o,o) for o in ORDER],rotation=30,ha="right"); ax.set_ylabel("Mean nRMSE over six hidden states"); ax.legend(frameon=False,loc="upper left"); ax.set_title("a  Hidden-state accuracy of twelve estimators on NM-HiddenBench")
states=["T","c","q","s","n","d"]; names=["T","c","q","σ","n","δ"]
for k,fl in enumerate(["ID","Stress"]):
    ax=fig.add_subplot(gs[1,k]); M=np.array([[d[(d.method==o)&(d.fleet==fl)][f"nrmse_{s}"].mean() for s in states] for o in ORDER])
    im=ax.imshow(M,cmap="YlOrRd",vmin=0,vmax=0.6,aspect="auto"); ax.set_xticks(range(6)); ax.set_xticklabels(names); ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([LAB.get(o,o) for o in ORDER])
    best=M.argmin(0)
    for i in range(len(ORDER)):
        for j in range(6): ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",fontsize=9.5,fontweight="bold" if best[j]==i else "normal",color="white" if M[i,j]>0.42 else "k")
    ax.set_title(("b  Per-state nRMSE, in-distribution test" if fl=="ID" else "c  Per-state nRMSE, stress test")+" (bold: best)"); ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True); ax.set_xlabel("Hidden state")
fig.savefig('/home/claude/figs_real/figS1_simulation_benchmark.png',dpi=200,bbox_inches="tight"); print("S1 ok")
