#!/usr/bin/env bash
# Full research pipeline. ~15 min wall clock. Logs every stage to outputs/.
#
# This regenerates every number in fraud-decisioning-findings.md from the raw
# data. It is the reproducibility guarantee behind the findings, so it stays
# runnable as a single command.
set -euo pipefail
cd "$(dirname "$0")"

# The stages moved to research/ (ADR-0001) but config.py stays at the root,
# because it resolves data/ and outputs/ relative to its own location. Keeping
# it there means artifact paths are identical whether a stage runs via this
# script or directly — and the published logs were produced the second way.
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

mkdir -p outputs
bash 00_download_data.sh

for s in 01_profile 02_features 03_model 03b_calibrate 04b_ltv 05_economics 06_full; do
  echo "=== $s"
  python3 "research/$s.py" 2>&1 | tee "outputs/$s.log"
done

echo "=== figures"
python3 "research/figures.py" 2>&1 | tee "outputs/figures.log"

echo "done -- see outputs/*.log"
