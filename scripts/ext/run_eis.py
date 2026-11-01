import os, sys
import numpy as np, pandas as pd, glob, os, sys
from multiprocessing import Pool
sys.path.insert(0,'ext'); from ecm import fit, drt
cfg=pd.read_csv('fleet_cfg.csv').set_index('cell')
BANDS={'R_sei':(2e-4,5e-3),'R_ct':(5e-3,5e-1),'R_diff':(5e-1,50)}
def one(path):
    cell=os.path.basename(path)[11:-4]
    if cell.startswith('P000'): return None
    z=pd.read_csv(path,sep=';'); z=z[(z.is_rt==1)&(z.valid==1)&z.z_re_comp_mOhm.notna()]
    if len(z)==0: return None
    t0=z.timestamp_s.min(); rows=[]; p0=None
    for (ts,soc),s in z.groupby(['timestamp_s','soc_nom']):
        s=s.sort_values('freq_Hz',ascending=False)
        if len(s)<20: continue
        f=s.freq_Hz.values; zr=s.z_re_comp_mOhm.values; zi=-s.z_im_comp_mOhm.values
        try:
            p,rm=fit(f,zr,zi,p0); 
            tau,g,_=drt(f,zr-zr.min(),zi)
        except Exception as e: continue
        d=dict(cell=cell,timestamp_s=ts,days=(ts-t0)/86400,soc=soc,T_eis=s.t_avg_degC.iloc[0],ocv=s.ocv_est_avg_V.iloc[0],
               R0=p[0],L=p[1],R1=p[2],tau1=p[3],a1=p[4],R2=p[5],tau2=p[6],a2=p[7],W=p[8],rmse=rm,
               Rpol=zr[np.argmin(np.abs(f-1.0))]-p[0], Rlow=zr[-1]-p[0])
        for k,(a,b) in BANDS.items(): d[k]=g[(tau>=a)&(tau<b)].sum()
        rows.append(d)
    r=pd.DataFrame(rows)
    for c in ['age_type','age_temp','age_soc','age_chg_rate','age_dischg_rate','age_profile','V_max_cyc_V','V_min_cyc_V']: r[c]=cfg.loc[cell,c] if cell in cfg.index else np.nan
    return r
if __name__=='__main__':
    files=sorted(glob.glob(os.environ['KIT_RAW']+'/eis/cell_eisv2_P*.csv'))
    with Pool(8) as pool: out=[r for r in pool.imap_unordered(one,files) if r is not None]
    df=pd.concat(out,ignore_index=True); df.to_csv('eis_decomp.csv',index=False); print(df.shape, df.rmse.describe())
