#!/bin/bash
# Reproduces every number and figure of the paper from the two public KIT archives.
# Usage:  bash run_all.sh /path/to/10_35097-1969.tar /path/to/10.35097-krk531nmj4bsshha.tar
set -e
TAR1="$1"; TAR2="$2"; ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/data/v2" "$ROOT/data/eisfix" "$ROOT/data/raw" "$ROOT/work" "$ROOT/out"
tar -xf "$TAR1" -C "$ROOT/data/v2"; tar -xf "$TAR2" -C "$ROOT/data/eisfix"
D="$ROOT/data/v2/10.35097-1969/data/dataset"
unzip -qo "$D/cell_eocv2.zip" -d "$ROOT/data/raw/eoc"; unzip -qo "$D/cell_plsv2.zip" -d "$ROOT/data/raw/pls"; unzip -qo "$D/cfg.zip" -d "$ROOT/data/raw/cfg"
unzip -qo "$ROOT/data/eisfix/10.35097-krk531nmj4bsshha/data/dataset/cell_eisv2_fixed.zip" -d "$ROOT/data/raw/eis"
cd "$ROOT/work"
export KIT_RAW="$ROOT/data/raw" KIT_OUT="$ROOT/out"
python3 -I "$ROOT/scripts/inspect.py"       # fleet_cfg.csv  (configuration of every cell)
python3 -I "$ROOT/scripts/build_table.py"   # checkups.csv, cycles.csv  (step 1: tables)
python3 -I "$ROOT/scripts/hyp.py"           # hyp.json, H1..H4 csv      (step 2: findings Q1-Q4)
python3 -I "$ROOT/scripts/ext/calendar_regime.py"   # calendar_conditions.csv, calendar_cells.csv (Section 3.1, two regimes)
python3 -I "$ROOT/scripts/fig_hyp.py"       # fig_findings_H1-H4.png
python3 -I "$ROOT/scripts/ext/symreg.py"            # symreg.json (Section 3.5, sparse regression)
python3 -I "$ROOT/scripts/ext/calendar_twin_v2.py"  # calendar_twin_v2.json (regime law forecast)
python3 -I "$ROOT/scripts/ext/run_eis.py"           # eis_decomp.csv (all 17,548 spectra; ~20 min on 2 cores)
python3 -I "$ROOT/scripts/ext/eis_signature.py"     # eis_signature.csv/json (Section 3.6)
python3 -I "$ROOT/scripts/ext/reversible.py"        # reversible_blocks.csv (Section 3.7)
python3 -I "$ROOT/scripts/ext/knee_warn.py"         # knee_warn.json, knee_features.csv (Section 3.8)
python3 -I "$ROOT/scripts/ext/fig_new.py"; KIT_OUT="$KIT_OUT" python3 -I "$ROOT/scripts/ext/figS1.py"           # fig9_discovered_laws.png (Fig. 3), fig8_impedance_reversible_knee.png (Fig. 4)
PSD=0.15 python3 -I "$ROOT/scripts/twin2.py" # twin2.json, twin2_H200.csv, twin2_examples.csv (step 3: twin, prospective evaluation)
python3 -I "$ROOT/scripts/knee_rl.py"       # knee.json, knee_eval.csv, policy_eval.csv      (step 4: knee stage + offline policy evaluation)
python3 -I "$ROOT/scripts/xai_calib.py"     # xai.json, attribution.csv, counterfactuals.csv, sensitivity.csv, fig_xai_materials.png (step 5)
KIT_OUT="$KIT_OUT" python3 -I "$ROOT/scripts/ga.py"; KIT_OUT="$KIT_OUT" python3 -I "$ROOT/scripts/dash.py"   # graphical abstract, Fig. 5, Fig. 6 (step 6)
KIT_OUT="$KIT_OUT" python3 -I "$ROOT/scripts/fig4_forecast.py"; KIT_OUT="$KIT_OUT" PSD=0.15 python3 -I "$ROOT/scripts/fig3d.py"   # Fig. 6 (forecasts), Supplementary Fig. S2 (3D twin)
bash "$ROOT/scripts/seeds/run_seeds.sh"     # five random train/test splits -> seeds/seed_table.csv (Table 1 spread, Supplementary Tables S3-S4)
echo "done: results in $ROOT/work, figures in $KIT_OUT"
