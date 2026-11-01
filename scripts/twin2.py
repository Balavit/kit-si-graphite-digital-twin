import pandas as pd, numpy as np, json, time, os
from scipy.optimize import least_squares
T = pd.read_csv("checkups.csv"); T["soh"]=T.cap/3.0; T=T.sort_values(["cell","days"])
C = pd.read_csv("cycles.csv").sort_values(["cell","days"]); C = C[C.cyc_charged==0]; C = C[C.groupby("cell").cumcount()%4==0]
R=8.314e-3; cells=T.drop_duplicates("cell").set_index("cell")
def feats(c):
    r=cells.loc[c]; prof = r.age_type==3
    return dict(Tk=r.age_temp+273.15, soc=r.age_soc/100 if r.age_type==1 else 0.8, crate=(r.age_chg_rate if r.age_type==2 else (1.0 if prof else 0.0)), vwin=(r.V_max_cyc_V-r.V_min_cyc_V) if r.age_type!=1 else 0.0, prof=float(prof))
NAMES=["lnkc","Ec","s_soc","lnkl","El","lnkq","Eq","s_c","s_w","lnkp","Ep","s_cp","Lstar","lng","s_prof","a","b","s_gT"]
SOC0,SOCW=0.77,0.15   # switch of the linear calendar regime (Section 3.5)
def model(th, phi, d, q, f):
    lnkc,Ec,s_soc,lnkl,El,lnkq,Eq,s_c,s_w,lnkp,Ep,s_cp,Lstar,lng,s_prof,a,b,s_gT = th
    iT = 1/(R*f["Tk"]) - 1/(R*298.15)
    kc = np.exp(lnkc+phi[0]+s_soc*f["soc"]-Ec*iT)                                   # sqrt(t) regime (diffusion-limited film)
    kl = np.exp(lnkl+phi[0]-El*iT)/(1+np.exp(-(f["soc"]-SOC0)/SOCW))                 # linear regime, thermally activated, switched on at high SoC
    kq = np.exp(lnkq+phi[1]+Eq*(-iT)+s_c*f["crate"]+s_w*f["vwin"]+s_prof*f["prof"])
    kp = np.exp(lnkp+phi[2]+Ep*iT+s_cp*f["crate"])           # plating: faster when cold (Ep>0 -> grows with 1/T)
    Lsei = kc*np.sqrt(d) + kl*d + kq*q/1000; Lpl = kp*q/1000
    L1 = Lsei+Lpl
    acc = 1 + np.exp(lng+s_gT*iT)*(1/(1+np.exp(-(L1-Lstar)/0.02)))    # second stage after the knee threshold
    Lpl = Lpl*acc; Lsei = Lsei*(1+0.3*(acc-1))
    return Lsei, Lpl, 1 + a*Lsei + b*Lsei**2
SEED=int(os.environ.get("SEED","0")); rng=np.random.default_rng(SEED); test_cells=[]
for cond,g in cells.groupby("cond"): test_cells += list(rng.choice(g.index, max(1,len(g)//3), replace=False))
train=T[~T.cell.isin(test_cells)]; test=T[T.cell.isin(test_cells)]
def cal_route(th, f, d):
    """calendar route alone (sqrt regime + linear regime), population parameters"""
    lnkc,Ec,s_soc,lnkl,El=th[:5]; iT=1/(R*f["Tk"])-1/(R*298.15)
    return np.exp(lnkc+s_soc*f["soc"]-Ec*iT)*np.sqrt(d)+np.exp(lnkl-El*iT)/(1+np.exp(-(f["soc"]-SOC0)/SOCW))*d
def resid(th, D):
    r=[]
    for c,g in D.groupby("cell"):
        f=feats(c); q0=g.cap.iloc[0]; loss=(1-g.cap/q0).values
        Ls,Lp,Rr=model(th,[0,0,0],g.days.values,g.q_chg.values,f); r.append((loss-(Ls+Lp))/0.02)
        m=g.Rmid_s50.notna().values
        if m.sum()>2:
            rr=(g.Rmid_s50/g.Rmid_s50.dropna().iloc[0]).values[m]; r.append((rr-Rr[m])/0.3)
    return np.concatenate(r)
th0=[np.log(0.003),2,0.5, np.log(1e-4),60, np.log(0.02),25,0.3,0.8, np.log(0.01),30,0.5, 0.24,np.log(2),0.0, 2.0,10.0, 20.0]
lo=[-10,0,-2,-14,0,-10,0,-2,-2,-10,0,-2,0.20,-2,-3,0,0,0]; hi=[0,80,3,-2,150,0,80,3,3,0,45,3,0.30,4,3,20,60,80]
t0=time.time(); fit=least_squares(resid,th0,bounds=(lo,hi),args=(train,),loss="soft_l1",f_scale=1.0); TH=fit.x
print("fit %.0fs"%(time.time()-t0)); print({k: round(float(v),3) for k,v in zip(NAMES,TH)})
print("ROM test RMSE SoH: %.4f"%np.sqrt(np.mean(np.concatenate([(((1-g.cap/g.cap.iloc[0]).values - sum(model(TH,[0,0,0],g.days.values,g.q_chg.values,feats(c))[:2]))**2) for c,g in test.groupby("cell")]))))
# ---------------- instance twin (EKF): z=[Lsei, Lpl, phi_sei, phi_pl, b_ot]
def run(c, g, H, use_phi=True, use_cyc=True, use_R=True, mc=0):
    f=feats(c); q0=g.cap.iloc[0]; r0=g.Rmid_s50.dropna().iloc[0] if g.Rmid_s50.notna().any() else np.nan
    obs=[(d,q,1-cap/q0,"RT",rm/r0 if (np.isfinite(rm) and np.isfinite(r0)) else np.nan) for d,q,cap,rm in zip(g.days,g.q_chg,g.cap,g.Rmid_s50)]
    if use_cyc:
        gc=C[C.cell==c]; obs += [(d,q,1-cap/q0,"OT",np.nan) for d,q,cap in zip(gc.days,gc.total_q_chg_sum_Ah,gc.cap_aged_est_Ah)]
    obs.sort(key=lambda o:o[0])
    import os; psd = float(os.environ.get('PSD','0.4')) if use_phi else 1e-6
    m=np.zeros(6); P=np.diag([1e-5,1e-5,psd**2,psd**2,psd**2,0.05**2]); Q=np.diag([1e-4,1e-4,0,0,0,1e-3])**2
    last=(0.0,0.0)
    def step(m,P,d,q):
        def g_(mm): Ls,Lp,_=model(TH,mm[2:5],d,q,f); Ls0,Lp0,_=model(TH,mm[2:5],last[0],last[1],f); return np.array([Ls-Ls0,Lp-Lp0])
        inc=g_(m); J=np.eye(6); eps=1e-4
        for j in (2,3,4): mm=m.copy(); mm[j]+=eps; J[0:2,j]=(g_(mm)-inc)/eps
        m=m.copy(); m[0:2]+=inc; P=J@P@J.T+Q*max(d-last[0],0); return m,P
    for d,q,y,kind,rr in obs:
        if d>H: break
        m,P=step(m,P,d,q); last=(d,q)
        hs=[]; 
        if kind=="RT": hs.append((np.array([1,1,0,0,0,0.]),y,0.008**2))
        else: hs.append((np.array([1,1,0,0,0,1.]),y,0.012**2))
        if kind=="RT" and use_R and np.isfinite(rr):
            a,b=TH[15],TH[16]; hs.append((np.array([a+2*b*m[0],0,0,0,0,0.]), rr-(1+a*m[0]+b*m[0]**2)+ (a+2*b*m[0])*m[0], 0.15**2))
        for Hh,yy,Rn in hs:
            S=Hh@P@Hh+Rn; K=P@Hh/S; m=m+K*(yy-Hh@m); P=(np.eye(6)-np.outer(K,Hh))@P
    fut=g[g.days>H]; out=[]
    if mc:   # Monte Carlo fast-forward from the posterior
        Z=np.random.default_rng(1).multivariate_normal(m,P,mc)
    for d,q,cap,rm in zip(fut.days,fut.q_chg,fut.cap,fut.Rmid_s50):
        mp,_=step(m,P,d,q); Ls,Lp,Rr=mp[0],mp[1],1+TH[15]*mp[0]+TH[16]*mp[0]**2
        rec=dict(days=d,true_soh=cap/3.0,pred_soh=(1-Ls-Lp)*q0/3.0,true_R=rm/r0 if np.isfinite(r0) else np.nan,pred_R=Rr,Lsei=Ls,Lpl=Lp)
        if mc:
            ss=[]; 
            for z in Z:
                zp,_=step(z,P,d,q); ss.append((1-zp[0]-zp[1])*q0/3.0)
            rec["q05"],rec["q95"]=np.percentile(ss,[5,95])
        out.append(rec)
    return pd.DataFrame(out), m
def evaluate(H, **kw):
    rows=[]
    for c,g in test.groupby("cell"):
        if g.days.max()<H+40 or (g.days<=H).sum()<3: continue
        pr,m=run(c,g,H,**kw)
        if len(pr): rows.append(dict(cell=c,atype=cells.loc[c].age_type,T=cells.loc[c].age_temp,rmse=np.sqrt(((pr.pred_soh-pr.true_soh)**2).mean()),end=abs(pr.pred_soh.iloc[-1]-pr.true_soh.iloc[-1]),rmseR=np.sqrt(((pr.pred_R-pr.true_R)**2).mean()) if pr.true_R.notna().any() else np.nan,phi=m[2:5].tolist(),Lpl_frac=float(pr.Lpl.iloc[-1]/(pr.Lsei.iloc[-1]+pr.Lpl.iloc[-1]+1e-9))))
    return pd.DataFrame(rows)
def linear(H):
    rows=[]
    for c,g in test.groupby("cell"):
        if g.days.max()<H+40 or (g.days<=H).sum()<3: continue
        p=g[g.days<=H]; fu=g[g.days>H]; a=np.polyfit(p.days,p.soh,1); pr=np.polyval(a,fu.days); rows.append(dict(cell=c,rmse=np.sqrt(((pr-fu.soh)**2).mean()),end=abs(pr[-1]-fu.soh.iloc[-1])))
    return pd.DataFrame(rows)
res={}
import os
if os.environ.get("SWEEP"):
    for H in (100,200,300):
        full=evaluate(H); ni=evaluate(H,use_phi=False)
        d=full.merge(ni,on="cell",suffixes=("","_ni")); d["dphi"]=np.abs(np.array(d.phi.tolist())).mean(1)
        print("PSD",os.environ["PSD"],"H",H,"twin %.4f no_inst %.4f | by_type"%(full.rmse.mean(),ni.rmse.mean()),full.groupby("atype").rmse.mean().round(4).to_dict(),"| corr(|phi|, err-err_ni)=%.2f"%np.corrcoef(d.dphi,d.rmse-d.rmse_ni)[0,1],flush=True)
    raise SystemExit
for H in (100,200,300):
    t0=time.time(); full=evaluate(H); r={"n":len(full),"twin":full.rmse.mean(),"twin_end":full.end.mean(),"twin_R":full.rmseR.mean()}
    for nm,kw in [("no_instance",dict(use_phi=False)),("no_cycles",dict(use_cyc=False)),("no_impedance",dict(use_R=False)),("checkups_only_no_instance",dict(use_phi=False,use_cyc=False))]:
        e=evaluate(H,**kw); r[nm]=e.rmse.mean(); r[nm+"_end"]=e.end.mean()
    l=linear(H); r["linear"]=l.rmse.mean(); r["linear_end"]=l.end.mean(); r["by_type"]=full.groupby("atype").rmse.mean().round(4).to_dict(); r["by_T"]=full.groupby("T").rmse.mean().round(4).to_dict()
    r["Lpl_frac_by_T"]=full.groupby("T").Lpl_frac.mean().round(3).to_dict()
    res[H]={k:(round(float(v),4) if not isinstance(v,dict) else v) for k,v in r.items()}; print(H,res[H],"%.0fs"%(time.time()-t0),flush=True)
    if H==200: full.to_csv("twin2_H200.csv",index=False)
json.dump(dict(TH=TH.tolist(),names=NAMES,res=res,test_cells=list(map(str,test_cells))),open("twin2.json","w"),indent=1)
# example trajectories with MC bands for the figure
ex=[]
for c in test_cells:
    g=T[T.cell==c]
    if cells.loc[c].age_type==2 and g.days.max()>400 and (g.soh.iloc[-1]<0.7): ex.append(c)
pickle_rows={}
for c in ex[:4]:
    g=T[T.cell==c]; pr,_=run(c,g,200,mc=200); pickle_rows[c]=pr
pd.concat(pickle_rows,names=["cell"]).reset_index(level=0).to_csv("twin2_examples.csv",index=False); print("examples",ex[:4])
