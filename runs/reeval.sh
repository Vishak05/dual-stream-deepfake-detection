#!/usr/bin/env bash
# Re-runs both evaluations with logit-margin scoring once pipeline2 is done.
set -euo pipefail
REPO="/mnt/c/Users/visha/Desktop/college/sem 7/ai/project/implementation/dual-stream-deepfake-detection"
PY="$HOME/venvs/dsfnet/bin/python -u"; export PYTHONWARNINGS=ignore; M="$HOME/dsfnet-data/manifests"; RUNS="$HOME/dsfnet-runs"
# (pipeline2 already finished when this was launched)
for f in evaluation_perturbed evaluation_clean; do mv "$RUNS/$f.json" "$RUNS/${f}_softmax.json"; done
RUN_ARGS=(
  --run baseline_original=spatial_only:"$RUNS/baseline_original/spatial_only.pth"
  --run baseline_augmented=spatial_only:"$RUNS/baseline_augmented/spatial_only.pth"
  --run dual_augmented=dual_stream:"$RUNS/dual_augmented/dual_stream.pth"
  --run paper_repro=spatial_only:"$RUNS/paper_repro/spatial_only.pth"
  --run baseline_robust=spatial_only:"$RUNS/baseline_robust/spatial_only.pth"
  --run dsfnet_v2=dual_stream_v2:"$RUNS/dsfnet_v2/dual_stream_v2.pth"
  --run dsfnet_v2_robust=dual_stream_v2:"$RUNS/dsfnet_v2_robust/dual_stream_v2.pth"
)
echo "[reeval] perturbed"
$PY "$REPO/scripts/evaluate_runs.py" --test-manifest "$M/test_perturbed_manifest.csv" "${RUN_ARGS[@]}" \
  --reference paper_repro --calibration-manifest "$M/valid_perturbed_manifest.csv" --target-fpr 0.05 \
  --ops-csv "$M/test_perturbed_manifest_ops.csv" --save-scores "$RUNS/scores_perturbed.npz" \
  --num-workers 8 --out-report "$RUNS/evaluation_perturbed.json" 2>&1
echo "[reeval] clean"
$PY "$REPO/scripts/evaluate_runs.py" --test-manifest "$M/test_full_manifest.csv" "${RUN_ARGS[@]}" \
  --reference paper_repro --calibration-manifest "$M/valid_manifest.csv" --target-fpr 0.05 \
  --save-scores "$RUNS/scores_clean.npz" --num-workers 8 --out-report "$RUNS/evaluation_clean.json" 2>&1
echo "[reeval] DONE"
