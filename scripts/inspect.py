import pandas as pd, numpy as np, glob, sys
import os; R = os.environ.get("KIT_RAW", "/home/claude/data/raw")
f = sorted(glob.glob(f"{R}/eoc/cell_eocv2_P012_2*"))[0]
e = pd.read_csv(f, sep=";"); print(e.shape); print(list(e.columns)[-12:])
print(e["cyc_condition"].value_counts().to_dict(), "cyc_charged", e["cyc_charged"].value_counts().to_dict())
cu = e[e["cap_aged_est_Ah"].notna()]; print("rows with capacity:", len(cu)); print(cu[["timestamp_s","cyc_condition","cyc_charged","cap_aged_est_Ah","soh_cap","total_q_chg_sum_Ah","t_start_degC","num_cycles_op","num_cycles_checkup"]].head(12).to_string())
print("time span days", (e.timestamp_s.max()-e.timestamp_s.min())/86400)
z = pd.read_csv(sorted(glob.glob(f"{R}/eis/*P012_2*"))[0], sep=";")
print("EIS: rows", len(z), "is_rt", z.is_rt.value_counts().to_dict(), "soc_nom", sorted(z.soc_nom.unique()), "valid", z.valid.value_counts().to_dict())
g = z.groupby(["timestamp_s","is_rt","soc_nom"]).size(); print("spectra:", len(g), "freqs per spectrum", g.iloc[0]); print(z[(z.is_rt==1)&(z.soc_nom==50)].groupby("timestamp_s")[["z_ref_now_mOhm","soh_imp","t_avg_degC"]].first().head(8).to_string())
print(z.freq_Hz.unique()[:40])
p = pd.read_csv(sorted(glob.glob(f"{R}/pls/*P012_2*"))[0], sep=";")
print("PLS rows", len(p), "soc", sorted(p.soc_nom.unique()), "is_rt", p.is_rt.value_counts().to_dict()); print(p.groupby(["timestamp_s","soc_nom"])[["r_ref_10ms_mOhm","r_ref_1s_mOhm","t_avg_degC"]].first().head(8).to_string())
# fleet config
rows=[]
for f in sorted(glob.glob(f"{R}/cfg/cell_cfg_P*.csv")):
    c = pd.read_csv(f, sep=";").iloc[0]; rows.append(dict(cell=f.split("cell_cfg_")[1][:-4], **{k: c[k] for k in ["cell_used","age_type","age_temp","age_soc","age_chg_rate","age_dischg_rate","age_profile","V_max_cyc_V","V_min_cyc_V"]}))
cfg = pd.DataFrame(rows); print(cfg.cell_used.value_counts().to_dict()); print(cfg[cfg.cell_used==1].groupby(["age_type","age_temp"]).size().to_string()); print(cfg[cfg.cell_used==1].drop_duplicates(["age_type","age_temp","age_soc","age_chg_rate","age_dischg_rate","age_profile"]).shape)
cfg.to_csv("fleet_cfg.csv", index=False)
