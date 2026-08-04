#!/usr/bin/env bash
# IEEE-CIS Fraud Detection (Vesta Corporation e-commerce transactions).
# Mirrored on HuggingFace; the Kaggle original requires competition acceptance.
set -euo pipefail
BASE="https://huggingface.co/datasets/aliceczr/ieee-fraud-detection/resolve/main"
mkdir -p data && cd data
for f in train_transaction.csv train_identity.csv; do
  [ -f "$f" ] && { echo "$f present, skipping"; continue; }
  echo "downloading $f ..."
  curl -fL --retry 3 -o "$f" "$BASE/$f"
done
echo "verifying checksums ..."
cat > .md5 << 'CHK'
58b4038d8715f5e11007b826bef00ce7  train_transaction.csv
8487db5001c8bad139f3318d5d3db416  train_identity.csv
CHK
md5sum -c .md5
echo "OK"
