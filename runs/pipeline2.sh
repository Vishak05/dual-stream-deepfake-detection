#!/usr/bin/env bash
# Base-paper comparison pipeline: perturbed test set (base paper Sec. III-A3)
# with unseen generators, paper-composition reproduction, and the proposed
# DSF-Net v2 + label-symmetric robust augmentation, then calibrated evaluation.
set -euo pipefail
REPO="/mnt/c/Users/visha/Desktop/college/sem 7/ai/project/implementation/dual-stream-deepfake-detection"
PY="$HOME/venvs/dsfnet/bin/python"
DATA="$HOME/dsfnet-data"
M="$DATA/manifests"
RUNS="$HOME/dsfnet-runs"
EPOCHS=10
WORKERS=8

while pgrep -f gen_sg3.py >/dev/null || pgrep -f build_perturbed_set.py >/dev/null; do sleep 30; done
echo "[pipeline2] inputs ready: $(ls "$DATA/stylegan3t_test" | wc -l) StyleGAN3-t, $(ls "$DATA/stylegan2_test" | wc -l) StyleGAN2 test images"

"$PY" "$REPO/scripts/build_manifests.py" \
  --dataset-root "$DATA/real_vs_fake/real-vs-fake" \
  --stylegan3-dir "$DATA/stylegan3_train" \
  --stylegan3-test-dir "$DATA/stylegan3_test" \
  --paper-degraded-dir "$DATA/stylegan3_train_degraded" \
  --out-dir "$M"

# Clean test set + the two unseen generators, then its perturbed copy.
"$PY" - "$M" "$DATA" <<'EOF'
import sys
from pathlib import Path
from dsfnet.data.manifest import read_manifest_csv, scan_source_folder, write_manifest_csv
m, d = Path(sys.argv[1]), Path(sys.argv[2])
full = read_manifest_csv(m / "test_manifest.csv") \
    + scan_source_folder(d / "stylegan3t_test", label=1, source="StyleGAN3-t") \
    + scan_source_folder(d / "stylegan2_test", label=1, source="StyleGAN2")
write_manifest_csv(full, m / "test_full_manifest.csv")
print(f"test_full_manifest.csv: {len(full)} images")
EOF
"$PY" "$REPO/scripts/build_perturbed_set.py" --mode test --manifest "$M/test_full_manifest.csv" \
  --out-dir "$DATA/test_perturbed" --out-manifest "$M/test_perturbed_manifest.csv" --seed 3 --workers $WORKERS

train() {  # name model manifest [extra flags]
  echo "[pipeline2] training $1"
  "$PY" "$REPO/scripts/train_model.py" --manifest "$M/$3" --model "$2" \
    --out-dir "$RUNS/$1" --epochs $EPOCHS --num-workers $WORKERS "${@:4}" 2>&1 | grep -v Warning
}
train paper_repro      spatial_only   paper_augmented_manifest.csv
train baseline_robust  spatial_only   augmented_manifest.csv --robust-aug
train dsfnet_v2_robust dual_stream_v2 augmented_manifest.csv --robust-aug
train dsfnet_v2        dual_stream_v2 augmented_manifest.csv

RUN_ARGS=(
  --run baseline_original=spatial_only:"$RUNS/baseline_original/spatial_only.pth"
  --run baseline_augmented=spatial_only:"$RUNS/baseline_augmented/spatial_only.pth"
  --run dual_augmented=dual_stream:"$RUNS/dual_augmented/dual_stream.pth"
  --run paper_repro=spatial_only:"$RUNS/paper_repro/spatial_only.pth"
  --run baseline_robust=spatial_only:"$RUNS/baseline_robust/spatial_only.pth"
  --run dsfnet_v2=dual_stream_v2:"$RUNS/dsfnet_v2/dual_stream_v2.pth"
  --run dsfnet_v2_robust=dual_stream_v2:"$RUNS/dsfnet_v2_robust/dual_stream_v2.pth"
)
echo "[pipeline2] evaluating on the perturbed (base-paper protocol) test set"
"$PY" "$REPO/scripts/evaluate_runs.py" --test-manifest "$M/test_perturbed_manifest.csv" "${RUN_ARGS[@]}" \
  --reference paper_repro --calibration-manifest "$M/valid_perturbed_manifest.csv" --target-fpr 0.05 \
  --ops-csv "$M/test_perturbed_manifest_ops.csv" --save-scores "$RUNS/scores_perturbed.npz" \
  --num-workers $WORKERS --out-report "$RUNS/evaluation_perturbed.json"
echo "[pipeline2] evaluating on the clean test set"
"$PY" "$REPO/scripts/evaluate_runs.py" --test-manifest "$M/test_full_manifest.csv" "${RUN_ARGS[@]}" \
  --reference paper_repro --calibration-manifest "$M/valid_manifest.csv" --target-fpr 0.05 \
  --save-scores "$RUNS/scores_clean.npz" --num-workers $WORKERS --out-report "$RUNS/evaluation_clean.json"

echo "[pipeline2] explanations"
"$PY" - "$M/test_perturbed_manifest.csv" "$M/explain_perturbed_manifest.csv" <<'EOF'
import sys
from dsfnet.data.manifest import read_manifest_csv, write_manifest_csv
entries = read_manifest_csv(sys.argv[1])
sources = ("FFHQ", "StyleGAN", "StyleGAN3", "StyleGAN3-t", "StyleGAN2")
write_manifest_csv([e for s in sources for e in [x for x in entries if x.source == s][:3]], sys.argv[2])
EOF
for run in paper_repro:spatial_only dsfnet_v2_robust:dual_stream_v2; do
  name=${run%%:*}; model=${run##*:}
  "$PY" "$REPO/scripts/generate_explanations.py" --checkpoint "$RUNS/$name/$model.pth" --model "$model" \
    --manifest "$M/explain_perturbed_manifest.csv" --out-dir "$RUNS/explanations_v2/$name" --num-samples 15
done
echo "[pipeline2] DONE"
