import os
import pandas as pd, numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":11.5,"axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":12,"axes.labelsize":11.5,"legend.fontsize":10.5,"xtick.labelsize":10.5,"ytick.labelsize":10.5})
cm={0:"#2166ac",10:"#67a9cf",25:"#ef8a62",40:"#b2182b"}
E=pd.read_csv('eis_decomp.csv'); S=pd.read_csv('eis_signature.csv'); RB=pd.read_csv('reversible_blocks.csv'); CY=pd.read_csv('/home/claude/bat/cycles.csv'); U=pd.read_csv('/home/claude/bat/checkups.csv')
KW=json.load(open('knee_warn.json')); KF=pd.read_csv('knee_features.csv'); SR=json.load(open('symreg.json')); V2=json.load(open('calendar_twin_v2.json'))
# ================= Figure 8: impedance fingerprint and reversible loss
fig,axs=plt.subplots(3,2,figsize=(10.2,13.6)); ax=axs.ravel()
# a: one spectrum fresh vs aged with DRT bands (cell at 25 C 50% SoC)
from ecm import drt
z=pd.read_csv('/home/claude/data/raw/eis/cell_eisv2_P012_1_S12_C11.csv',sep=';'); z=z[(z.is_rt==1)&(z.valid==1)&(z.soc_nom==50)&z.z_re_comp_mOhm.notna()]
ts=sorted(z.timestamp_s.unique())
for t,col,lab in [(ts[0],"#444","fresh"),(ts[-1],cm[25],f"{(ts[-1]-ts[0])/86400:.0f} d")]:
    s=z[z.timestamp_s==t].sort_values('freq_Hz',ascending=False); ax[0].plot(s.z_re_comp_mOhm,s.z_im_comp_mOhm,"o-",ms=3,lw=1,color=col,label=lab)
ax[0].set_xlabel("Re Z (mΩ)"); ax[0].set_ylabel("−Im Z (mΩ)"); ax[0].set_ylim(-3,7); ax[0].legend(frameon=False); ax[0].set_title("a  Check-up spectrum, 0.05 Hz–10 kHz, 50% SoC, 25 °C")
ins=ax[0].inset_axes([0.45,0.55,0.5,0.4])
for t,col in [(ts[0],"#444"),(ts[-1],cm[25])]:
    s=z[z.timestamp_s==t].sort_values('freq_Hz',ascending=False); tau,g,_=drt(s.freq_Hz.values,s.z_re_comp_mOhm.values-s.z_re_comp_mOhm.min(),-s.z_im_comp_mOhm.values)
    m_=(tau*1e3>0.05)&(tau*1e3<2000); ins.semilogx(tau[m_]*1e3,g[m_],color=col,lw=1)
ins.axvspan(0.2,5,color="#ccc",alpha=0.4); ins.axvspan(5,500,color="#fdd",alpha=0.5); ins.set_xlabel("τ (ms)",fontsize=9.5); ins.set_ylabel("DRT (mΩ)",fontsize=9.5); ins.tick_params(labelsize=9); ins.set_ylim(0,3.5); ins.text(0.6,3.0,"HF",fontsize=9); ins.text(30,3.0,"MF",fontsize=9)
# b: fresh bands vs SoC
f=E.sort_values('days').groupby(['cell','soc']).first().reset_index(); gb=f.groupby('soc')[['R0','R_sei','R_ct']].median()
ax[1].plot(gb.index,gb.R_sei,"s-",color="#555",label="HF band 0.2–5 ms (film + contact)"); ax[1].plot(gb.index,gb.R_ct,"o-",color="#b2182b",label="MF band 5–500 ms (interfacial kinetics)")
ax[1].set_xlabel("State of charge at EIS (%)"); ax[1].set_ylabel("Band resistance, fresh cells (mΩ)"); ax[1].legend(frameon=False,fontsize=10.5); ax[1].set_title("b  HF band is SoC-independent, MF band is not")
# c: growth per unit loss, cold vs warm cycling and calendar (SoC 50)
c50=S[(S.soc==50)]
groups=[("Calendar\n0–40 °C",c50[c50.age_type==1]),("Cycling\n0–10 °C",c50[(c50.age_type.isin([2,3]))&(c50['T']<=10)]),("Cycling\n25–40 °C",c50[(c50.age_type.isin([2,3]))&(c50['T']>=25)])]
x=np.arange(3); w=0.26
for k,(key,col,lab) in enumerate([("dR0_dL","#999","ohmic R₀"),("dR_sei_dL","#555","HF band"),("dR_ct_dL","#b2182b","MF band")]):
    med=[g[key].median() for _,g in groups]; q1=[g[key].quantile(.25) for _,g in groups]; q3=[g[key].quantile(.75) for _,g in groups]
    ax[2].bar(x+(k-1)*w,med,w,color=col,label=lab,yerr=[np.array(med)-np.array(q1),np.array(q3)-np.array(med)],capsize=2,error_kw=dict(lw=0.8))
ax[2].axhline(0,color="k",lw=0.6); ax[2].set_xticks(x); ax[2].set_xticklabels([g[0] for g in groups]); ax[2].set_ylabel("ΔR per unit capacity loss (mΩ), 50% SoC"); ax[2].legend(frameon=False,fontsize=10.5)
ax[2].set_title("c  Cold grows the HF band, warm the MF band")
# d: example sawtooth of per-cycle discharge charge with check-ups (narrow window 25 C 0.33C) 
cand=U[(U.age_type==2)&(U.age_temp==25)&(U.age_chg_rate==0.33)&(U.V_min_cyc_V>3)&(U.V_max_cyc_V<4.1)].cell.unique(); cell=cand[0]
k=CY[(CY.cell==cell)&(CY.cyc_charged==0)].sort_values('days'); u=U[U.cell==cell]
ax[3].plot(k.days,k.delta_q_Ah.abs(),".",ms=2,color="#444",label="discharge charge per cycle, 3.25–4.09 V window")
for d in u.days: ax[3].axvline(d,color=cm[25],lw=0.5,alpha=0.6)
ax[3].plot([],[],color=cm[25],lw=0.8,label="check-up (full cycle at 25 °C)"); ax[3].set_xlim(0,200); ax[3].set_xlabel("Days"); ax[3].set_ylabel("Ah per cycle"); ax[3].legend(frameon=False,fontsize=10.5,loc="lower left"); ax[3].set_title("d  Within-block loss partly recovers at each check-up")
# e: reversible fraction by window and temperature
ok=RB[RB['drop']>0.01]; ok['win']=np.where(ok.Vmin<3,"2.5–4.2 V",np.where(ok.Vmax<4.1,"3.25–4.09 V","3.25–4.2 V"))
pos=0; ticks=[]; labs=[]
for wn in ["2.5–4.2 V","3.25–4.09 V","3.25–4.2 V"]:
    for t in (0,10,25,40):
        v=ok[(ok.win==wn)&(ok.age_temp==t)].rev_frac.clip(-0.5,1.0)
        if len(v)<5: continue
        bp=ax[4].boxplot([v.values],positions=[pos],widths=0.7,patch_artist=True,showfliers=False,boxprops=dict(facecolor=cm[t],alpha=0.8),medianprops=dict(color="k")); pos+=1
    ticks.append(pos-2.5); labs.append(wn); pos+=1
ax[4].axhline(0,color="k",lw=0.6); ax[4].set_xticks(ticks); ax[4].set_xticklabels(labs); ax[4].set_ylabel("Recovered fraction of within-block loss"); ax[4].set_title("e  Recovered share: 0 (full window), 40–70% (partial)")
from matplotlib.patches import Patch
ax[4].legend(handles=[Patch(color=cm[t],label=f"{t} °C") for t in (0,10,25,40)],frameon=False,fontsize=10.5,ncol=2,loc="lower right")
# f: knee: AUC comparison + precursor box
labels=["capacity\nfeatures","+ resistance\nfeatures","+ CE, recovery,\nconditions"]; a60=[KW['h60'][k] for k in ['auc_capacity_only','auc_capacity_plus_resistance','auc_all']]; a90=[KW['h90'][k] for k in ['auc_capacity_only','auc_capacity_plus_resistance','auc_all']]
x=np.arange(3); ax[5].bar(x-0.18,a60,0.36,color="#888",label="knee within 60 d"); ax[5].bar(x+0.18,a90,0.36,color="#444",label="knee within 90 d")
ax[5].axhline(0.5,color="k",ls="--",lw=0.7); ax[5].set_xticks(x); ax[5].set_xticklabels(labels); ax[5].set_ylim(0.4,1.0); ax[5].set_ylabel("Leave-condition-out AUC"); ax[5].legend(frameon=False,fontsize=10.5,loc="upper right")
ax[5].set_title("f  Knee warning: impedance adds nothing")
fig.tight_layout(); fig.savefig('/home/claude/figs_real/fig8_impedance_reversible_knee.png',dpi=200)
# ================= Figure 9: discovered laws
fig,axs=plt.subplots(3,1,figsize=(8.6,12.5)); ax=axs
# a: bootstrap selection frequency calendar
bf=SR['calendar']['bootstrap_selection_freq']; names=list(bf); vals=[bf[n] for n in names]; sel=SR['calendar']['discovered']
cols=["#b2182b" if n in sel else "#bbb" for n in names]
ax[0].barh(range(len(names)),vals,color=cols); ax[0].set_yticks(range(len(names))); ax[0].set_yticklabels(names); ax[0].invert_yaxis(); ax[0].set_xlabel("Selection frequency over 60 cell-bootstraps"); ax[0].set_title("a  Calendar law: terms chosen by sparse regression (ln rate)")
ax[0].axvline(0.8,color="k",ls="--",lw=0.7)
# b: cycling library
bf=SR['cycling']['bootstrap_selection_freq']; names=list(bf); vals=[bf[n] for n in names]; sel=SR['cycling']['discovered']; cols=["#b2182b" if n in sel else "#bbb" for n in names]
ax[1].barh(range(len(names)),vals,color=cols); ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels(names); ax[1].invert_yaxis(); ax[1].set_xlabel("Selection frequency over 60 cell-bootstraps"); ax[1].set_title("b  Cycling law: a time-driven term and cold-activated rate terms")
ax[1].axvline(0.8,color="k",ls="--",lw=0.7)
# c: forecast error one-exponent vs regime law
H=[100,200,300]; r=V2['forecast_rmse_heldout_calendar']
ax[2].plot(H,[r[str(h)]['one_exponent']*100 for h in H],"s-",color="#888",label="one-exponent law k tⁿ (as in the twin)"); ax[2].plot(H,[r[str(h)]['regime']*100 for h in H],"o-",color="#b2182b",label="regime law: k√t + k_lin(T, SoC) t")
ax[2].set_xlabel("History used (days)"); ax[2].set_ylabel("RMSE of forecast loss, held-out calendar cells (%)"); ax[2].set_ylim(0,1.6); ax[2].legend(frameon=False,fontsize=10.5); ax[2].set_title("c  The discovered law forecasts better out of sample")
fig.tight_layout(); fig.savefig('/home/claude/figs_real/fig9_discovered_laws.png',dpi=200); print("done")
