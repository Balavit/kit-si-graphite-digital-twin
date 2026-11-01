import os
"""3D digital-twin figure: (a) the 18650 cell and its twin as a layered state model; (b) the twin's inferred route
surfaces over (temperature, time); (c) per-cell 3D trajectories in (SEI-type loss, plating-type loss, R_mid)."""
import numpy as np, pandas as pd, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plt.rcParams.update({"font.size":13,"axes.titlesize":13.5,"axes.labelsize":13,"legend.fontsize":12,"xtick.labelsize":11.5,"ytick.labelsize":11.5})
src=open("knee_rl.py").read().split("rows=[]\nfor H in (100,200,300):")[0]
exec(src)   # TH, model, feats, cells, train/test, routes via xai_calib? -> define routes here
def routes(f,d,q):
    lnkc,Ec,s_soc,lnkl,El,lnkq,Eq,s_c,s_w,lnkp,Ep,s_cp,Ls_,lng,s_prof,a,b,s_gT=TH; iT=1/(R*f["Tk"])-1/(R*298.15)
    cal=cal_route(TH,f,d); cyc=np.exp(lnkq+Eq*(-iT)+s_c*f["crate"]+s_w*f["vwin"]+s_prof*f["prof"])*q/1000; pl=np.exp(lnkp+Ep*iT+s_cp*f["crate"])*q/1000
    return cal,cyc,pl
fig=plt.figure(figsize=(13,12)); gs=fig.add_gridspec(2,2,width_ratios=[1.25,1],height_ratios=[1,1.1],wspace=0.2,hspace=0.22)
# ---------- (a) the cell and its twin: cylinder + exploded layers
ax=fig.add_subplot(gs[0,0],projection="3d"); ax.set_box_aspect((2.2,0.8,1.3))
th=np.linspace(0,2*np.pi,60); z=np.linspace(0,1,2); TH_,Z=np.meshgrid(th,z)
ax.plot_surface(np.cos(TH_)*0.45,np.sin(TH_)*0.45,Z*1.0,color="#b0b0b0",alpha=0.25,linewidth=0)
ax.plot(np.cos(th)*0.45,np.sin(th)*0.45,1.0,color="#777",lw=0.8); ax.plot(np.cos(th)*0.45,np.sin(th)*0.45,0,color="#777",lw=0.8)
# layered "twin" stack beside it: three route layers + two observation planes
def slab(x0,z0,w,h,d,color,label,alpha=0.85):
    X=[x0,x0+w]; Y=[-d/2,d/2]; Zs=[z0,z0+h]
    faces=[[(X[0],Y[0],Zs[0]),(X[1],Y[0],Zs[0]),(X[1],Y[1],Zs[0]),(X[0],Y[1],Zs[0])],[(X[0],Y[0],Zs[1]),(X[1],Y[0],Zs[1]),(X[1],Y[1],Zs[1]),(X[0],Y[1],Zs[1])],
           [(X[0],Y[0],Zs[0]),(X[1],Y[0],Zs[0]),(X[1],Y[0],Zs[1]),(X[0],Y[0],Zs[1])],[(X[0],Y[1],Zs[0]),(X[1],Y[1],Zs[0]),(X[1],Y[1],Zs[1]),(X[0],Y[1],Zs[1])],
           [(X[0],Y[0],Zs[0]),(X[0],Y[1],Zs[0]),(X[0],Y[1],Zs[1]),(X[0],Y[0],Zs[1])],[(X[1],Y[0],Zs[0]),(X[1],Y[1],Zs[0]),(X[1],Y[1],Zs[1]),(X[1],Y[0],Zs[1])]]
    ax.add_collection3d(Poly3DCollection(faces,facecolors=color,edgecolors="#444",linewidths=0.4,alpha=alpha)); ax.text(x0+w+0.3,0.6,z0+h/2,label,fontsize=10.5,va="center",ha="left")
slab(1.1,0.05,0.7,0.16,0.7,"#9ecae1","L$_{cal}$ calendar SEI")
slab(1.1,0.27,0.7,0.16,0.7,"#fdae6b","L$_{cyc}$ cycling SEI")
slab(1.1,0.49,0.7,0.16,0.7,"#c1272d","L$_{pl}$ Li plating")
slab(1.1,0.74,0.7,0.05,0.7,"#555","observed: capacity, R$_{mid}$",alpha=0.6)
slab(1.1,0.86,0.7,0.05,0.7,"#2b8cbe","twin: EKF + MC rollouts",alpha=0.6)
for zz in (0.13,0.35,0.57): ax.plot([0.47,1.08],[0,0],[zz,zz],color="#888",lw=0.8,ls="--")
ax.text(0,0,1.12,"18650 NMC / graphite–SiO$_x$ cell",ha="center",fontsize=15)
ax.set_xlim(-0.6,3.1); ax.set_ylim(-0.6,0.6); ax.set_zlim(0,1.15); ax.set_axis_off(); ax.view_init(elev=14,azim=-88)
ax.set_title("a  The cell and its twin: three hidden route layers,\ntwo observed channels, one estimator",fontsize=13.5)
# ---------- (b) route surfaces over temperature and time, cycling at 1 C narrow window, fleet-median throughput
ax=fig.add_subplot(gs[0,1],projection="3d")
Tg=np.linspace(0,40,25); dg=np.linspace(0,600,40); TT,DD=np.meshgrid(Tg,dg)
qpd=float(train[(train.age_type==2)].groupby("cell").apply(lambda g: g.q_chg.max()/g.days.max()).median())
CAL=np.zeros_like(TT); CYC=np.zeros_like(TT); PL=np.zeros_like(TT)
for i in range(TT.shape[0]):
    for j in range(TT.shape[1]):
        f=dict(Tk=TT[i,j]+273.15,soc=0.8,crate=1.0,vwin=0.84,prof=0.0); c1,c2,c3=routes(f,DD[i,j],DD[i,j]*qpd); CAL[i,j],CYC[i,j],PL[i,j]=c1,c2,c3
ax.plot_surface(TT,DD,CAL*100,color="#9ecae1",alpha=0.75,linewidth=0,label="calendar SEI")
ax.plot_surface(TT,DD,(CAL+CYC)*100,color="#fdae6b",alpha=0.75,linewidth=0)
ax.plot_surface(TT,DD,(CAL+CYC+PL)*100,color="#c1272d",alpha=0.6,linewidth=0)
ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Days"); ax.set_zlabel("Cumulative loss (%)"); ax.view_init(elev=22,azim=-130)
ax.set_title("b  Twin's loss surfaces (1 C, 3.25–4.09 V), stacked\nby route",fontsize=13.5)
from matplotlib.patches import Patch; ax.legend(handles=[Patch(color="#9ecae1",label="calendar SEI"),Patch(color="#fdae6b",label="+ cycling SEI"),Patch(color="#c1272d",label="+ plating")],loc="upper left",frameon=False)
# ---------- (c) per-cell trajectories in route space: SEI-type loss vs plating-type loss vs measured R_mid
ax=fig.add_subplot(gs[1,:],projection="3d"); cm={0:"#2166ac",10:"#67a9cf",25:"#ef8a62",40:"#b2182b"}
for c,g in T[T.age_type==2].groupby("cell"):
    g=g.dropna(subset=["Rmid_s50"]).sort_values("days")
    if len(g)<6: continue
    f=feats(c); cal,cyc,pl=routes(f,g.days.values,g.q_chg.values); r=g.Rmid_s50.values/g.Rmid_s50.values[0]
    ax.plot((cal+cyc)*100,pl*100,r,color=cm[int(cells.loc[c].age_temp)],lw=0.9,alpha=0.75)
ax.set_xlabel("SEI-type loss (%)"); ax.set_ylabel("Plating-type loss (%)"); ax.set_zlabel("Measured R$_{mid}$ / initial"); ax.view_init(elev=20,azim=-50)
ax.set_title("c  144 cycling cells in route space: cold cells move along\nthe plating axis without resistance growth",fontsize=13.5)
for t in (0,10,25,40): ax.plot([],[],[],color=cm[t],lw=2,label=f"{t} °C")
ax.legend(frameon=False,loc="upper left")
fig.savefig(os.environ.get("KIT_OUT",".")+"/fig7_twin_3d.png",dpi=200,bbox_inches="tight"); print("3d ok")
