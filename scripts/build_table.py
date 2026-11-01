import pandas as pd, numpy as np, glob, re
import os; R = os.environ.get("KIT_RAW", "/home/claude/data/raw")
cfg = pd.read_csv("fleet_cfg.csv").set_index("cell")
rows = []; cyc_rows = []
for f in sorted(glob.glob(f"{R}/eoc/cell_eocv2_P*.csv")):
    cell = re.search(r"cell_eocv2_(P\d+_\d+_S\d+_C\d+)", f).group(1)
    if cell.startswith("P000"): continue
    e = pd.read_csv(f, sep=";"); cu = e[(e.cap_aged_est_Ah.notna()) & (e.cyc_condition==2)].copy()
    cyc = e[(e.cap_aged_est_Ah.notna()) & (e.cyc_condition==1)].copy()
    if len(cu) == 0: continue
    cu["cu"] = (cu.num_cycles_checkup).astype(int)
    g = cu.groupby("cu").agg(t=("timestamp_s","mean"), cap=("cap_aged_est_Ah","mean"), soh_cap=("soh_cap","mean"), q_chg=("total_q_chg_sum_Ah","max"), q_dis=("total_q_dischg_sum_Ah","max"), e_dis=("total_e_dischg_sum_Wh","max"), n_op=("num_cycles_op","max"), ce=("coulomb_efficiency","mean")).reset_index()
    t0 = e.timestamp_s.min(); g["days"] = (g.t - t0)/86400
    # EIS at RT, 50 % SoC
    zf = glob.glob(f"{R}/eis/cell_eisv2_{cell}.csv")
    if zf:
        z = pd.read_csv(zf[0], sep=";"); z = z[(z.is_rt==1)&(z.valid==1)]
        for soc in (10, 50, 90):
            zs = z[z.soc_nom==soc]
            feats = []
            for ts, s in zs.groupby("timestamp_s"):
                s = s.sort_values("freq_Hz"); fr = s.freq_Hz.values; re_ = s.z_re_comp_mOhm.values; im = s.z_im_comp_mOhm.values
                ok = np.isfinite(re_)&np.isfinite(im)
                if ok.sum() < 10: continue
                fr, re_, im = fr[ok], re_[ok], im[ok]
                # R0: real part at the zero-crossing of the imaginary part (interpolate), else min |Im| above 100 Hz
                hi = fr >= 100
                if (im[hi] > 0).any() and (im[hi] < 0).any():
                    i = np.where(np.diff(np.sign(im[hi])))[0][0]; a, b = np.where(hi)[0][i], np.where(hi)[0][i+1]
                    r0 = re_[a] + (re_[b]-re_[a]) * (0 - im[a])/(im[b]-im[a])
                else:
                    r0 = re_[hi][np.argmin(np.abs(im[hi]))]
                def at(fq): j = np.argmin(np.abs(np.log(fr)-np.log(fq))); return re_[j], im[j]
                r1, i1 = at(1.0); r01, i01 = at(0.1); r005, i005 = at(0.05)
                feats.append(dict(t=ts, soc=soc, R0=r0, Rmid=r1-r0, Rlow=r005-r0, Imin_1Hz=-i1, Im_005=-i005, T_eis=s.t_avg_degC.iloc[0], zref=s.z_ref_now_mOhm.iloc[0], soh_imp=s.soh_imp.iloc[0]))
            F = pd.DataFrame(feats)
            if len(F)==0: continue
            # match each check-up to nearest EIS within 3 days
            for k in ("R0","Rmid","Rlow","Imin_1Hz","Im_005","T_eis","zref","soh_imp"):
                g[f"{k}_s{soc}"] = [F.loc[(F.t-tt).abs().idxmin(), k] if (F.t-tt).abs().min() < 3*86400 else np.nan for tt in g.t]
    pf = glob.glob(f"{R}/pls/cell_plsv2_{cell}.csv")
    if pf:
        p = pd.read_csv(pf[0], sep=";"); p = p[p.is_rt==1]
        for soc in (10, 50, 90):
            ps = p[p.soc_nom==soc].groupby("timestamp_s")[["r_ref_10ms_mOhm","r_ref_1s_mOhm"]].first().reset_index()
            for k, nm in (("r_ref_10ms_mOhm","Rp10ms"),("r_ref_1s_mOhm","Rp1s")):
                g[f"{nm}_s{soc}"] = [ps.loc[(ps.timestamp_s-tt).abs().idxmin(), k] if len(ps) and (ps.timestamp_s-tt).abs().min() < 3*86400 else np.nan for tt in g.t]
    c = cfg.loc[cell]
    for k in ("age_type","age_temp","age_soc","age_chg_rate","age_dischg_rate","age_profile","V_max_cyc_V","V_min_cyc_V"): g[k] = c[k]
    g["cell"] = cell; g["cond"] = f"{int(c.age_type)}_T{int(c.age_temp)}_S{int(c.age_soc)}_C{c.age_chg_rate:.2f}_D{c.age_dischg_rate:.2f}_P{int(c.age_profile)}"
    rows.append(g)
    if len(cyc):
        cyc = cyc[["timestamp_s","cyc_charged","cap_aged_est_Ah","delta_q_Ah","delta_e_Wh","coulomb_efficiency","energy_efficiency","ocv_est_start_V","ocv_est_end_V","soc_est_start","soc_est_end","t_start_degC","t_end_degC","cyc_duration_s","total_q_chg_sum_Ah","num_cycles_op"]].copy()
        cyc["days"] = (cyc.timestamp_s - t0)/86400; cyc["cell"] = cell; cyc_rows.append(cyc)
T = pd.concat(rows, ignore_index=True)
C = pd.concat(cyc_rows, ignore_index=True); C.to_csv("cycles.csv", index=False); print("cycle-level rows", C.shape, "cells with cycling", C.cell.nunique())
T.to_csv("checkups.csv", index=False)
print(T.shape, "cells", T.cell.nunique(), "conditions", T.cond.nunique())
print(T.groupby("age_type").agg(cells=("cell","nunique"), cus=("cu","max"), soh_end=("soh_cap", lambda s: s.min())).to_string())
print(T[["days","cap","soh_cap","q_chg","R0_s50","Rmid_s50","Rlow_s50","Rp10ms_s50","Rp1s_s50","soh_imp_s50"]].describe().round(3).to_string())
print("missing EIS at checkups:", T.R0_s50.isna().mean().round(3), "missing pulse:", T.Rp1s_s50.isna().mean().round(3))
T["soh"] = T.cap/3.0
print(T.groupby("cell").soh.min().describe().round(3).to_string())
