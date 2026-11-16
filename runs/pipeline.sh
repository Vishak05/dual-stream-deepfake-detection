#!/usr/bin/env bash
# Full results pipeline: waits for StyleGAN3 generation, builds manifests,
# trains the 4 runs, evaluates them, and produces explanation snapshots.
set -euo pipefail
REPO="/mnt/c/Users/visha/Desktop/college/sem 7/ai/project/implementation/dual-stream-deepfake-detection"
PY="$HOME/venvs/dsfnet/bin/python"
DATA="$HOME/dsfnet-data"
RUNS="$HOME/dsfnet-runs"
EPOCHS=10
WORKERS=8

while pgrep -f gen_sg3.py >/dev/null; do sleep 30; done
echo "[pipeline] generation done: $(ls "$DATA/stylegan3_train" | wc -l) train, $(ls "$DATA/stylegan3_test" | wc -l) test StyleGAN3 images"

"$PY" "$REPO/scripts/build_manifests.py" \
  --dataset-root "$DATA/real_vs_fake/real-vs-fake" \
  --stylegan3-dir "$DATA/stylegan3_train" \
  --stylegan3-test-dir "$DATA/stylegan3_test" \
  --out-dir "$DATA/manifests"

train() {  # name model manifest
  echo "[pipeline] training $1"
  "$PY" "$REPO/scripts/train_model.py" --manifest "$DATA/manifests/$3" --model "$2" \
    --out-dir "$RUNS/$1" --epochs $EPOCHS --num-workers $WORKERS 2>&1 | grep -v Warning
}
train baseline_original  spatial_only original_manifest.csv
train baseline_augmented spatial_only augmented_manifest.csv
train dual_original      dual_stream  original_manifest.csv
train dual_augmented     dual_stream  augmented_manifest.csv

echo "[pipeline] evaluating"
"$PY" "$REPO/scripts/evaluate_runs.py" --test-manifest "$DATA/manifests/test_manifest.csv" \
  --run baseline_original=spatial_only:"$RUNS/baseline_original/spatial_only.pth" \
  --run baseline_augmented=spatial_only:"$RUNS/baseline_augmented/spatial_only.pth" \
  --run dual_original=dual_stream:"$RUNS/dual_original/dual_stream.pth" \
  --run dual_augmented=dual_stream:"$RUNS/dual_augmented/dual_stream.pth" \
  --reference baseline_augmented --num-workers $WORKERS \
  --out-report "$RUNS/evaluation_report.json"

echo "[pipeline] explanations"
# 4 real, 4 StyleGAN, 4 StyleGAN3 test images so the snapshots cover every source.
"$PY" - "$DATA/manifests/test_manifest.csv" "$DATA/manifests/explain_manifest.csv" <<'EOF'
import sys
from dsfnet.data.manifest import read_manifest_csv, write_manifest_csv
entries = read_manifest_csv(sys.argv[1])
picked = [e for s in ("FFHQ", "StyleGAN", "StyleGAN3") for e in [x for x in entries if x.source == s][:4]]
write_manifest_csv(picked, sys.argv[2])
EOF
for run in baseline_augmented:spatial_only dual_augmented:dual_stream; do
  name=${run%%:*}; model=${run##*:}
  "$PY" "$REPO/scripts/generate_explanations.py" --checkpoint "$RUNS/$name/$model.pth" --model "$model" \
    --manifest "$DATA/manifests/explain_manifest.csv" --out-dir "$RUNS/explanations/$name" --num-samples 12
done
echo "[pipeline] DONE"
