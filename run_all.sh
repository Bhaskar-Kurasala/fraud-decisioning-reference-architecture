#!/usr/bin/env bash
# Full pipeline. ~15 min wall clock on 1 core / 3GB. Logs to outputs/.
set -euo pipefail
mkdir -p outputs
bash 00_download_data.sh
for s in 01_profile 02_features 03_model 03b_calibrate 04b_ltv 05_economics 06_full; do
  echo "=== $s"
  python3 "$s.py" 2>&1 | tee "outputs/$s.log"
done
echo "done -- see outputs/*.log"
