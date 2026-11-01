import pandas as pd, numpy as np, json
from scipy import stats, optimize
T = pd.read_csv("checkups.csv"); T["soh"] = T.cap/3.0; T = T.sort_values(["cell","days"])
T["Tk"] = T.age_temp + 273.15
out = {}
# ---------------- H1: calendar aging law
cal = T[T.age_type==1]
rows = []
for c, g in cal.groupby("cell"):
    g = g[g.days > 1]; 
    if len(g) < 6: continue
    q0 = g.cap.iloc[0]; loss = 1 - g.cap/q0; t = g.days.values
    # power law loss = k t^n ; fit n and k in log space (loss>0)
    m = loss > 0.002
    if m.sum() < 5: continue
    n, lnk = np.polyfit(np.log(t[m]), np.log(loss[m]), 1)
    ksq = np.polyfit(np.sqrt(t), loss, 1)[0]
    r0 = g.Rmid_s50.dropna(); rr = (r0.iloc[-1]/r0.iloc[0]) if len(r0) > 2 else np.nan
    rows.append(dict(cell=c, T=g.age_temp.iloc[0], soc=g.age_soc.iloc[0], n=n, k=np.exp(lnk), ksq=ksq, days=t[-1], loss_end=loss.iloc[-1], rmid_ratio=rr))
H1 = pd.DataFrame(rows)
out["H1_n_exponent"] = dict(mean=float(H1.n.mean()), sd=float(H1.n.std()), by_T=H1.groupby("T").n.mean().round(2).to_dict())
# Arrhenius per SoC on sqrt(t) rate constant
arr = {}
for soc, g in H1.groupby("soc"):
    gg = g.groupby("T").ksq.mean(); x = 1/(gg.index.values+273.15); y = np.log(gg.values)
    sl, ic, r, p, se = stats.linregress(x, y); arr[int(soc)] = dict(Ea_kJ=float(-sl*8.314/1000), r2=float(r**2), k_by_T=gg.round(5).to_dict())
# joint model: ln k = a + b*soc - Ea/RT
X = np.column_stack([np.ones(len(H1)), H1.soc/100, 1/(8.314e-3*(H1["T"]+273.15))]); beta = np.linalg.lstsq(X, np.log(H1.ksq.clip(1e-5)), rcond=None)[0]
pred = X@beta; out["H1_arrhenius_by_soc"] = arr; out["H1_joint"] = dict(Ea_kJ=float(-beta[2]), soc_coeff=float(beta[1]), r2=float(1-np.var(np.log(H1.ksq.clip(1e-5))-pred)/np.var(np.log(H1.ksq.clip(1e-5)))), n_cells=len(H1))
H1.to_csv("H1_calendar.csv", index=False)
# ---------------- H2: knee precursor in resistance
def knee(days, soh):
    # Bacon-Watts style: find breakpoint maximising fit improvement of two-line model
    best = (np.inf, None)
    for i in range(4, len(days)-3):
        a = np.polyfit(days[:i+1], soh[:i+1], 1); b = np.polyfit(days[i:], soh[i:], 1)
        sse = ((np.polyval(a, days[:i+1]) - soh[:i+1])**2).sum() + ((np.polyval(b, days[i:]) - soh[i:])**2).sum()
        if sse < best[0]: best = (sse, i, a[0], b[0])
    if best[1] is None: return None
    _, i, s1, s2 = best
    sse1 = ((np.polyval(np.polyfit(days, soh, 1), days) - soh)**2).sum()
    return dict(i=i, ratio=s2/s1 if s1 < 0 else np.nan, improve=1 - best[0]/sse1)
rows = []
for c, g in T[T.age_type.isin([2,3])].groupby("cell"):
    g = g.dropna(subset=["Rmid_s50"]); 
    if len(g) < 10: continue
    d, s, r = g.days.values, g.soh.values, g.Rmid_s50.values
    k = knee(d, s)
    if k is None: continue
    has = (k["improve"] > 0.6) and (k["ratio"] > 2.0) and (s[-1] < 0.75)
    i = k["i"]
    rows.append(dict(cell=c, T=g.age_temp.iloc[0], chg=g.age_chg_rate.iloc[0], atype=g.age_type.iloc[0], knee=has, soh_knee=s[i] if has else np.nan, rmid_knee_rel=r[i]/r[0] if has else np.nan, rmid_rel_end=r[-1]/r[0], soh_end=s[-1], day_knee=d[i] if has else np.nan, slope_ratio=k["ratio"], r0_knee_rel=(g.R0_s50.values[i]/g.R0_s50.values[0]) if has else np.nan))
H2 = pd.DataFrame(rows); H2.to_csv("H2_knee.csv", index=False)
kn = H2[H2.knee]
out["H2"] = dict(n_cells=len(H2), n_knee=int(H2.knee.sum()), soh_at_knee=dict(mean=float(kn.soh_knee.mean()), sd=float(kn.soh_knee.std()), cv=float(kn.soh_knee.std()/kn.soh_knee.mean())),
               rmid_rel_at_knee=dict(mean=float(kn.rmid_knee_rel.mean()), sd=float(kn.rmid_knee_rel.std()), cv=float(kn.rmid_knee_rel.std()/kn.rmid_knee_rel.mean())),
               r0_rel_at_knee=dict(mean=float(kn.r0_knee_rel.mean()), cv=float(kn.r0_knee_rel.std()/kn.r0_knee_rel.mean())),
               knee_by_T=H2.groupby("T").knee.mean().round(2).to_dict(), rmid_rel_knee_by_T=kn.groupby("T").rmid_knee_rel.mean().round(2).to_dict(),
               # does Rmid threshold crossing precede knee? fraction of knee cells where Rmid_rel exceeded 1.5 at or before knee
               frac_rmid_cross_before_knee=float((kn.rmid_knee_rel >= 1.5).mean()), nonknee_rmid_end=float(H2[~H2.knee].rmid_rel_end.mean()))
# ---------------- H3: cold cycling / plating signature
cyc = T[T.age_type==2].copy()
rows = []
for c, g in cyc.groupby("cell"):
    g = g.dropna(subset=["Rmid_s50"]); g = g[g.soh > 0.75]   # pre-knee regime only
    if len(g) < 5: continue
    sl = np.polyfit(1-g.soh, g.Rmid_s50/g.Rmid_s50.iloc[0], 1)[0]   # resistance growth per unit capacity loss
    q = g.q_chg.values; loss = 1-g.soh.values
    kq = np.polyfit(q, loss, 1)[0]*1000  # loss per 1000 Ah throughput
    rows.append(dict(cell=c, T=g.age_temp.iloc[0], chg=g.age_chg_rate.iloc[0], vmin=g.V_min_cyc_V.iloc[0], vmax=g.V_max_cyc_V.iloc[0], dR_dQ=sl, loss_per_kAh=kq))
H3 = pd.DataFrame(rows); H3.to_csv("H3_cold.csv", index=False)
C = pd.read_csv("cycles.csv"); C = C[(C.cyc_charged==0)&(C.coulomb_efficiency.between(95,100.5))]
ce = C.merge(T.drop_duplicates("cell")[["cell","age_type","age_temp","age_chg_rate"]], on="cell"); ce = ce[ce.age_type==2]
ce_tab = ce.groupby(["age_temp","age_chg_rate"]).coulomb_efficiency.median().unstack().round(3)
out["H3"] = dict(dR_dQ_by_T_chg=H3.pivot_table(index="T", columns="chg", values="dR_dQ", aggfunc="median").round(2).to_dict(), loss_per_kAh_by_T_chg=H3.pivot_table(index="T", columns="chg", values="loss_per_kAh", aggfunc="median").round(4).to_dict(),
               CE_median_by_T_chg=ce_tab.to_dict(), kruskal_dRdQ_T=float(stats.kruskal(*[g.dR_dQ.values for _, g in H3.groupby("T")]).pvalue))
# ---------------- H4: variability decomposition
end = T.groupby("cell").last(); end = end[end.age_type.isin([1,2])]
# SoH at a common reference: interpolate SoH at 300 days (cells that reach it)
ref = []
for c, g in T.groupby("cell"):
    if g.days.max() >= 300 and g.days.min() <= 5: ref.append(dict(cell=c, cond=g.cond.iloc[0], atype=g.age_type.iloc[0], soh300=np.interp(300, g.days, g.soh), soh0=g.soh.iloc[0], rmid0=g.Rmid_s50.dropna().iloc[0] if g.Rmid_s50.notna().any() else np.nan, r0_0=g.R0_s50.dropna().iloc[0] if g.R0_s50.notna().any() else np.nan, loss60=np.interp(60, g.days, 1-g.soh)))
H4 = pd.DataFrame(ref); H4.to_csv("H4_var.csv", index=False)
grp = H4.groupby("cond").soh300; ss_between = sum(len(g)*(g.mean()-H4.soh300.mean())**2 for _, g in grp); ss_total = ((H4.soh300-H4.soh300.mean())**2).sum()
within = H4.soh300 - H4.cond.map(grp.mean())
# within-condition prediction of soh300 from early-life features (demeaned per condition)
feat = H4[["soh0","rmid0","r0_0","loss60"]].copy()
for k in feat: feat[k] = feat[k] - H4.cond.map(H4.groupby("cond")[k].mean())
m = feat.notna().all(1)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
pr = cross_val_predict(LinearRegression(), feat[m], within[m], cv=10)
out["H4"] = dict(n_cells=len(H4), frac_var_between_conditions=float(ss_between/ss_total), within_sd=float(within.std()), between_sd=float(np.sqrt(ss_between/len(H4))),
                 within_r2_from_early_features_cv=float(1-np.var(within[m]-pr)/np.var(within[m])), corr_each={k: float(stats.spearmanr(feat.loc[m,k], within[m]).statistic) for k in feat},
                 within_r2_loss60_only=float(1-np.var(within[m]-cross_val_predict(LinearRegression(), feat.loc[m,["loss60"]], within[m], cv=10))/np.var(within[m])))
json.dump(out, open("hyp.json","w"), indent=1, default=float)
import pprint; pprint.pprint(out, width=140)
