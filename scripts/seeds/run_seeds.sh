#!/bin/bash
# Repeats the twin fit, knee stage, policy evaluation and calibration/explanations for five random train/test splits.
# Usage (from the work directory after run_all.sh): bash scripts/seeds/run_seeds.sh
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; W="$(pwd)"
for k in 0 1 2 3 4; do
  mkdir -p "$W/seeds/s$k"; cd "$W/seeds/s$k"
  for f in checkups.csv cycles.csv fleet_cfg.csv H2_knee.csv hyp.json; do ln -sf "$W/$f" "$f"; done
  cp "$ROOT/scripts/twin2.py" "$ROOT/scripts/knee_rl.py" "$ROOT/scripts/xai_calib.py" .
  SEED=$k PSD=0.15 python3 -I twin2.py > twin2.log 2>&1 && SEED=$k PSD=0.15 python3 -I knee_rl.py > knee.log 2>&1 && SEED=$k PSD=0.15 python3 -I xai_calib.py > xai.log 2>&1 && echo "seed $k done"
  cd "$W"
done
python3 -I "$ROOT/scripts/seeds/aggregate.py"   # seed_table.csv, seeds_summary.json
