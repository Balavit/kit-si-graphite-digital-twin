# kit-si-graphite-digital-twin

Digital twin of 228 commercial NMC/graphite–SiO cells on the public KIT aging dataset: hidden degradation routes, route-level explanations, calibrated Monte Carlo forecasts and an offline-validated operating map. Companion code for two manuscripts: *From drifting sensors to hidden states: instance-specific digital twins for nano–micro devices* (simulation benchmark, folder `simulation_benchmark/`) and *A digital twin of silicon–graphite lithium-ion cells reveals how operating conditions steer degradation routes, and learns to operate around them* (real-cell study, `scripts/`).

## Publish this repository (two commands)
```bash
gh repo create kit-si-graphite-digital-twin --public --source=. --push     # with the GitHub CLI, from this folder
# or: create an empty repo on github.com, then:  git remote add origin <url> && git push -u origin main
```
Then archive a release on Zenodo to obtain a DOI and cite it in the paper.

Everything in the paper is produced by five Python scripts run in order on two public archives (CC BY 4.0):

1. Result data v2 (333 MB): https://radar.kit.edu/radar/en/dataset/CoNQplSNoVeXExyV  (DOI 10.35097/1969)
2. Fixed EIS addendum (17 MB): https://radar.kit.edu/radar/en/dataset/krk531nmj4bsshha  (DOI 10.35097/krk531nmj4bsshha)

## Requirements
Python ≥ 3.10 with numpy, pandas, scipy, scikit-learn, matplotlib, openpyxl. One CPU core; ~15 minutes end to end; ~2 GB disk after extraction.

## Steps
```bash
bash run_all.sh /path/10_35097-1969.tar /path/10.35097-krk531nmj4bsshha.tar
```
or run the scripts one by one from the `work/` directory (each writes its outputs there):

| Step | Script | Produces | Paper section |
| --- | --- | --- | --- |
| 0 | `scripts/inspect.py` | `fleet_cfg.csv` | 2.1 |
| 1 | `scripts/build_table.py` | `checkups.csv` (4,006 check-ups: capacity, throughput, EIS features R0/Rmid/Rlow at 10/50/90 % SoC, pulse resistances), `cycles.csv` (569,675 cycle-level rows) | 2.1, Fig. 1 |
| 2 | `scripts/hyp.py`, `scripts/fig_hyp.py` | `hyp.json`, `H1_calendar.csv`, `H2_knee.csv`, `H3_cold.csv`, `H4_var.csv`, `fig_findings_H1-H4.png` | 3, Fig. 2 |
| 3 | `scripts/twin2.py` (env `PSD=0.15`) | population model fit `twin2.json`, prospective evaluation, `twin2_examples.csv` | 5, Fig. 4 |
| 4 | `scripts/knee_rl.py` | knee stage `knee.json`, `knee_eval.csv`, offline policy evaluation `policy_eval.csv` | 5.2, 6, Table 2 |
| 5 | `scripts/xai_calib.py` | conformal calibration, `attribution.csv`, `counterfactuals.csv`, `sensitivity.csv`, `xai.json`, `fig_xai_materials.png` | 4, 5.3, Fig. 3 |

The scripts read the raw folder from the environment variable `KIT_RAW` (default `/home/claude/data/raw`) and write figures to `KIT_OUT` (default `/mnt/user-data/outputs`). The train/test split is one held-out cell per condition, seed 0 (`np.random.default_rng(0)` in `twin2.py`); change the seed there to repeat the split. `results/` holds the outputs of the run reported in the paper so that any number can be checked without rerunning.

| 6 | `scripts/ga.py`, `scripts/dash.py` | graphical abstract, `fig5_operating_map.png`, `fig6_cell_dashboard.png` | GA, Fig. 5, Fig. 6 |

`simulation_benchmark/` holds the NM-HiddenBench simulator, the PRISM-TRACE estimator, eleven baselines and the experiment scripts of the Supplementary Information (`main.py`, `closed.py`, `figs.py`; requires `jax` and `optax`, ~40 min on one core).

## What each model term means
`L = L_cal + L_cyc + L_pl` with `L_cal = kc·exp(s_soc·SoC − Ec/RT)·t^n`, `L_cyc = kq·exp(−Eq/RT + s_c·C + s_w·ΔV + s_prof·profile)·Q`, `L_pl = kp·exp(+Ep/RT + s_cp·C)·Q`; resistance `Rmid/Rmid0 = 1 + a·(L_cal+L_cyc) + b·(L_cal+L_cyc)²`. Knee: once `L ≥ L*`, rates ×g with probability p_knee(T). Fitted values are in `results/twin2.json` (`names`, `TH`) and `results/knee.json`.

## Known limitations encoded in the code
- `Eq` (activation energy of the SEI-type cycling route) converges to its lower bound 0, so warm-side counterfactuals (+10 K at 25–40 °C) are not trustworthy.
- Raw Monte Carlo bands under-cover; `xai_calib.py` applies split-conformal scaling (factor 6–8) fitted on training cells.
- Instance adaptation (`PSD` > 0) does not improve forecasts on this dataset; the paper reports the population-parameter twin as the main result.

## Further findings (Sections 3.1 and 3.5–3.8 of the paper)

`scripts/ext/` holds the analyses that let algorithms search the fleet, run by `run_all.sh` after the four findings:

| Script | Output | Paper |
|---|---|---|
| `calendar_regime.py` | `calendar_conditions.csv`, `calendar_cells.csv` | two calendar regimes (Section 3.1, Fig. 2a–b) |
| `symreg.py` | `symreg.json` | sparse-regression rate laws (Section 3.5, Fig. 3a–b) |
| `calendar_twin_v2.py` | `calendar_twin_v2.json` | regime law vs single-exponent forecast (Fig. 3c) |
| `run_eis.py`, `ecm.py` | `eis_decomp.csv` | DRT band + equivalent-circuit decomposition of all 17,548 spectra |
| `eis_signature.py` | `eis_signature.csv/json` | impedance fingerprint of each route (Section 3.6, Fig. 4a–c) |
| `reversible.py` | `reversible_blocks.csv` | reversible vs permanent loss (Section 3.7, Fig. 4d–e) |
| `knee_warn.py` | `knee_warn.json`, `knee_features.csv` | knee early warning, Shapley (Section 3.8, Fig. 4f) |
| `fig_new.py` | `fig9_discovered_laws.png` (Fig. 3), `fig8_impedance_reversible_knee.png` (Fig. 4) | |

`results/ext/` holds the outputs of the reported run (`eis_decomp.csv.gz` is the full spectrum decomposition).

## Version 2 (Scientific Reports submission)

* The calendar route of the twin now has the two regimes found in Section "Four further findings" (`scripts/twin2.py`: √t term plus a thermally activated linear term switched on above 77% SoC); all twin-dependent numbers and figures were regenerated.
* `scripts/seeds/run_seeds.sh` repeats the twin fit, knee stage, policy evaluation and calibration for five random train/test splits (`SEED=0..4`); `scripts/seeds/aggregate.py` writes `seeds/seed_table.csv` and `seeds/seeds_summary.json`, which feed Table 1 and Supplementary Tables S3–S4. `results/seeds/` holds the outputs of the reported runs; `results/` holds split 0.
* `manuscript/scirep/` holds the manuscript sources: `paper.md` and `si.md` (Markdown with LaTeX equations), `build.py` (pandoc → .docx with native Word equations; `python3 build.py paper.md out.docx`), `fill.py` (writes the numbers from the result files into the text placeholders), `refs.py` (the verified reference list with DOIs) and `figs/`.
