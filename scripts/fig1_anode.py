import os
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle
plt.rcParams.update({"font.size":11.5,"axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":12,"axes.labelsize":11.5,"legend.fontsize":10.5,"xtick.labelsize":10.5,"ytick.labelsize":10.5})
T=pd.read_csv("checkups.csv"); T["soh"]=T.cap/3.0
fig=plt.figure(figsize=(11.5,11.5)); gs=fig.add_gridspec(2,3,height_ratios=[1.2,1],hspace=0.18,wspace=0.32)
# ---- a: schematic of the anode and the three nanoscale routes
ax=fig.add_subplot(gs[0,:]); ax.set_xlim(0,140); ax.set_ylim(-12,50); ax.axis("off")
ink="#1a1a1a"; sei="#fdae6b"; seid="#b35806"; pl="#c1272d"; gr="#9e9e9e"; si="#6a51a3"; cu="#b87333"; FS=11.5
def anode(x0,aged):
    ax.add_patch(Rectangle((x0,2),36,38,fc="#f7f7f7",ec="#999",lw=0.8)); ax.add_patch(Rectangle((x0,2),36,3.2,fc=cu,ec="none"))
    ax.text(x0+18,3.6,"Cu current collector",ha="center",va="center",fontsize=FS-1,color="white")
    G=[(7,14,5.5),(20,12,6.5),(29,26,5.5),(12,28,6),(23,32,4.3)]; S=[(14,21),(27,18),(6,24),(20,24),(32,34),(5,34)]
    for (x,y,r) in G:
        if aged: ax.add_patch(Circle((x0+x,y),r+1.4,fc=sei,ec="none",alpha=0.9))
        ax.add_patch(Circle((x0+x,y),r,fc=gr,ec="#555",lw=0.6))
    for (x,y) in S:
        if aged: ax.add_patch(Circle((x0+x,y),2.7,fc=sei,ec="none",alpha=0.9)); ax.plot([x0+x-0.8,x0+x+0.6],[y-0.6,y+0.9],color="white",lw=0.6)
        ax.add_patch(Circle((x0+x,y),1.5,fc=si,ec="none"))
    if aged: ax.add_patch(Polygon([(x0+0.5,35),(x0+5,37.5),(x0+10,35.5),(x0+15,38),(x0+20,36),(x0+25,38.5),(x0+30,36.5),(x0+35.5,38),(x0+35.5,33.5),(x0+0.5,32.5)],closed=True,fc=pl,ec="none",alpha=0.85))
anode(2,False); anode(48,True)
ax.text(20,43,"Pristine anode (nm–µm scale)",ha="center",va="bottom",fontsize=FS+0.5,fontweight="bold")
ax.text(66,43,"Aged anode: three routes",ha="center",va="bottom",fontsize=FS+0.5,fontweight="bold")
ax.text(22,12,"graphite\n(~10 µm)",ha="center",va="center",fontsize=FS-1)
ax.annotate("SiOₓ particles (~100 nm)",xy=(16,21),xytext=(4,-4),fontsize=FS-0.5,color=si,ha="left",arrowprops=dict(arrowstyle="-",color=si,lw=0.7),annotation_clip=False)
ax.text(66,35.5,"plated Li (µm deposits)",ha="center",va="center",fontsize=FS-1,color="white",fontweight="bold")
ax.annotate("Li plating (cold or fast charge) → L$_{pl}$: consumes Li, little ΔR",xy=(60,37),xytext=(48,47.5),fontsize=FS-0.5,color=pl,ha="left",va="bottom",arrowprops=dict(arrowstyle="-",color=pl,lw=0.7),annotation_clip=False)
ax.annotate("SEI growth (nm film) → L$_{cal}$, L$_{cyc}$: consumes Li, raises R$_{mid}$",xy=(55,9),xytext=(48,-4),fontsize=FS-0.5,color=seid,ha="left",va="top",arrowprops=dict(arrowstyle="-",color=seid,lw=0.7),annotation_clip=False)
ax.annotate("SiOₓ cracking exposes fresh surface → new SEI",xy=(68,24),xytext=(48,-10),fontsize=FS-0.5,color=si,ha="left",va="top",arrowprops=dict(arrowstyle="-",color=si,lw=0.7),annotation_clip=False)
# measured channels box
ax.add_patch(FancyBboxPatch((94,2),44,38,boxstyle="round,pad=0.3,rounding_size=1.5",fc="white",ec="#555",lw=0.9))
ax.text(116,38,"What the cell reports in service",ha="center",va="top",fontsize=FS+0.5,fontweight="bold")
rows=[("Capacity at check-up","observes L$_{cal}$ + L$_{cyc}$ + L$_{pl}$",ink),("R$_{mid}$ from EIS (1 Hz)","SEI-type routes only",seid),("1 s pulse resistance","ohmic + charge transfer",gr),("Coulombic efficiency","plating signature",pl),("Operating-T cycle capacity","continuous, reversible part",ink)]
for i,(a,b,c) in enumerate(rows):
    y=32.5-i*6.6; ax.text(96.5,y,a,fontsize=FS-0.5,va="center",fontweight="bold"); ax.text(96.5,y-2.9,b,fontsize=FS-1,va="center",color=c)
for y0 in (30,21,12): ax.add_patch(FancyArrowPatch((84.5,y0),(93,y0),arrowstyle="-|>",mutation_scale=10,lw=0.8,color="#555"))
ax.text(0,47.5,"a",fontsize=15,fontweight="bold",va="bottom")
# ---- b, c, d: fleet
cm={0:"#2166ac",10:"#67a9cf",25:"#ef8a62",40:"#b2182b"}
for (nm,at,ax,lbl) in [("Calendar",1,fig.add_subplot(gs[1,0]),"b"),("Cyclic",2,fig.add_subplot(gs[1,1]),"c")]:
    for c,g in T[T.age_type==at].groupby("cell"):
        g=g.sort_values("days"); ax.plot(g.days,g.soh,color=cm[int(g.age_temp.iloc[0])],lw=0.5,alpha=0.7)
    ax.set_title(f"{lbl}  {nm} aging, {T[T.age_type==at].cell.nunique()} cells"); ax.set_xlabel("Days"); ax.set_ylabel("Capacity / 3.0 Ah"); ax.set_ylim(0.3,1.0)
ax=fig.add_subplot(gs[1,2]); s=T.dropna(subset=["Rmid_s50"])
ax.scatter(1-s.soh,s.Rmid_s50,s=3,c=[cm[int(t)] for t in s.age_temp],alpha=0.5); ax.set_xlabel("Capacity loss (fraction)"); ax.set_ylabel("R$_{mid}$ at 50 % SoC, 25 °C (mΩ)"); ax.set_title("d  Resistance against loss, by temperature"); ax.set_ylim(0,30)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],marker="o",ls="",color=cm[t],label=f"{t} °C") for t in (0,10,25,40)],frameon=False,loc="upper left")
fig.savefig(os.environ.get("KIT_OUT",".")+"/fig1_anode_fleet.png",dpi=200,bbox_inches="tight"); print("ok")
