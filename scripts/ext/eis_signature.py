import os, sys
import pandas as pd, numpy as np, json
E=pd.read_csv('eis_decomp.csv'); U=pd.read_csv('checkups.csv')[['cell','t','cap','soh_cap','days']]
E=E.sort_values('timestamp_s'); U=U.sort_values('t')
M=pd.merge_asof(E,U.rename(columns={'t':'timestamp_s','days':'days_cu'}),on='timestamp_s',by='cell',direction='nearest',tolerance=3*86400)
M=M.dropna(subset=['cap']); M['soh']=M.cap/3.0
rows=[]
for (cell,soc),g in M.groupby(['cell','soc']):
    g=g.sort_values('timestamp_s'); g=g[g.soh>0.75]   # pre-knee
    if len(g)<6: continue
    L=1-g.cap/g.cap.iloc[0]
    if L.iloc[-1]<0.03: continue
    r=dict(cell=cell,soc=soc,age_type=g.age_type.iloc[0],T=g.age_temp.iloc[0],C=g.age_chg_rate.iloc[0],Vmin=g.V_min_cyc_V.iloc[0],Vmax=g.V_max_cyc_V.iloc[0],soc_cal=g.age_soc.iloc[0],loss=L.iloc[-1],days=g.days.iloc[-1])
    for k in ['R0','R_sei','R_ct','R_diff','W','Rpol']:
        sl=np.polyfit(L.values,g[k].values,1)[0]; r['d'+k+'_dL']=sl; r[k+'_0']=g[k].iloc[0]; r[k+'_rel']=g[k].iloc[-1]/g[k].iloc[0]
    rows.append(r)
S=pd.DataFrame(rows); S.to_csv('eis_signature.csv',index=False)
pd.set_option('display.width',250)
print("slope of band resistance (mOhm) per unit capacity loss (fraction), median by group, SoC 10/50/90:")
for soc in [10,50,90]:
    print("--- SoC",soc); print(S[S.soc==soc].groupby(['age_type','T'])[['dR0_dL','dR_sei_dL','dR_ct_dL','dR_diff_dL']].median().round(1).to_string())
# statistics: cold (0,10) vs warm (25,40) cycling cells at SoC 50
from scipy import stats
c=S[(S.age_type.isin([2,3]))&(S.soc==50)]
out={}
for k in ['dR0_dL','dR_sei_dL','dR_ct_dL']:
    a=c[c['T']<=10][k]; b=c[c['T']>=25][k]; p=stats.mannwhitneyu(a,b).pvalue
    out[k]=dict(cold_median=float(a.median()),warm_median=float(b.median()),p=float(p),n_cold=len(a),n_warm=len(b))
# calendar: SoC dependence of where growth appears
cal=S[S.age_type==1]
out['calendar']={int(s):{k:float(cal[cal.soc==s][k].median()) for k in ['dR0_dL','dR_sei_dL','dR_ct_dL']} for s in [10,50,90]}
# fresh SoC dependence
f=E.sort_values('days').groupby(['cell','soc']).first().reset_index()
out['fresh_by_soc']={int(s):{k:float(f[f.soc==s][k].median()) for k in ['R0','R_sei','R_ct']} for s in [10,30,50,70,90]}
json.dump(out,open('eis_signature.json','w'),indent=1); print(json.dumps(out,indent=1))
