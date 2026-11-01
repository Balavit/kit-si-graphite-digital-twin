import json,pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
rows=[];pol=[];cal=[]
for k in range(5):
    d=f"seeds/s{k}"
    try: tw=json.load(open(d+"/twin2.json"))["res"]; ke=pd.read_csv(d+"/knee_eval.csv"); A=pd.read_csv(d+"/policy_eval.csv"); X=json.load(open(d+"/xai.json"))
    except Exception as e: print("seed",k,"missing",e); continue
    for H in ("100","200","300"):
        km=ke[ke.H==int(H)]
        rows.append(dict(seed=k,H=int(H),linear=tw[H]["linear"],twin_pop=tw[H]["no_instance"],twin_inst=tw[H]["twin"],knee_mix=km.rmse_mix.mean(),knee_mix_nok=km.rmse_nok.mean(),
            cal=tw[H]["by_type"]["1.0"],cyc=tw[H]["by_type"]["2.0"],prof=tw[H]["by_type"]["3.0"],lpl0=tw[H]["Lpl_frac_by_T"]["0.0"],lpl40=tw[H]["Lpl_frac_by_T"]["40.0"],
            cov=X[f"calib_H{H}"]["coverage_conformal"],width=X[f"calib_H{H}"]["mean_width"],factor=X[f"calib_H{H}"]["factor"]))
    m=A.Ah_obs.notna(); rho=spearmanr(A.mean_Ah[m],A.Ah_obs[m]).statistic
    match=0; det=[]
    for T in (0,10,25,40):
        a=A[A["T"]==T]; rec=a.loc[a.cvar20.idxmax()]; best=a.loc[a.Ah_obs.idxmax()]; ok=(rec.crate==best.crate)&(rec.vwin==best.vwin); match+=int(ok); det.append((T,rec.crate,rec.vwin,best.crate,best.vwin,best.Ah_obs))
    pol.append(dict(seed=k,rho=rho,n=int(m.sum()),matches=match,detail=det))
    kj=json.load(open(d+"/knee.json")); cal.append(dict(seed=k,**{f"pk{int(t)}":v for t,v in kj["p_knee"].items()},Lstar=kj["Lstar"],G=kj["G"],att=X["attribution_by_T"],sens=X["sensitivity_by_T"]))
R=pd.DataFrame(rows); R.to_csv("seeds/seed_table.csv",index=False)
print(R.groupby("H")[["linear","twin_pop","twin_inst","knee_mix","cal","cyc","prof","lpl0","lpl40","cov","width","factor"]].agg(["mean","std"]).round(4).T.to_string())
P=pd.DataFrame(pol); print(P[["seed","rho","n","matches"]].to_string()); print(P.detail.iloc[0])
json.dump(dict(table=R.to_dict("records"),policy=[{k:(v if k!="detail" else [list(map(float,x)) for x in v]) for k,v in p.items()} for p in pol],knee=cal),open("seeds/seeds_summary.json","w"),indent=1,default=float)
