import os, sys
import pandas as pd, numpy as np
cy=pd.read_csv('cycles.csv'); cu=pd.read_csv('checkups.csv')
cu=cu[cu.age_type.isin([2,3])]  # cycling and profile cells
rows=[]
for cell,u in cu.groupby('cell'):
    u=u.sort_values('t'); k=cy[(cy.cell==cell)&(cy.cyc_charged==0)].sort_values('timestamp_s')  # discharge half-cycles
    if len(k)<20: continue
    ts=u.t.values; caps=u.cap.values
    for i in range(len(ts)-1):
        blk=k[(k.timestamp_s>ts[i])&(k.timestamp_s<ts[i+1])]
        if len(blk)<6: continue
        first=blk.cap_aged_est_Ah.iloc[1:4].mean(); last=blk.cap_aged_est_Ah.iloc[-3:].mean()
        nxt=k[(k.timestamp_s>ts[i+1])]
        if i+2<len(ts): nxt=nxt[nxt.timestamp_s<ts[i+2]]
        if len(nxt)<4: continue
        first_next=nxt.cap_aged_est_Ah.iloc[1:4].mean()
        rows.append(dict(cell=cell,block=i,day=u.days.values[i+1],cap_cu_before=caps[i],cap_cu_after=caps[i+1],n_cyc=len(blk),
            drop=first-last, recovered=first_next-last, T_op=blk.t_start_degC.median(),
            age_temp=u.age_temp.iloc[0],age_chg_rate=u.age_chg_rate.iloc[0],age_type=u.age_type.iloc[0],Vmax=u.V_max_cyc_V.iloc[0],Vmin=u.V_min_cyc_V.iloc[0],
            soh_cu=u.soh_cap.values[i+1]))
r=pd.DataFrame(rows); r['permanent']=r['drop']-r.recovered; r['rev_frac']=r.recovered/r['drop']
r.to_csv('reversible_blocks.csv',index=False)
ok=r[(r['drop']>0.01)]
print(len(r),len(ok))
g=ok.groupby(['age_temp','age_chg_rate','Vmin','Vmax']).agg(n=('rev_frac','size'),drop_med=('drop','median'),rec_med=('recovered','median'),rev_frac_med=('rev_frac','median'),rev_frac_iqr=('rev_frac',lambda x: x.quantile(.75)-x.quantile(.25))).round(3)
print(g.to_string())
print(ok.groupby('age_temp').rev_frac.describe().round(3))
