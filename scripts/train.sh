#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "${TRACELOCK_HOME:-}" ]]; then
  echo "TRACELOCK_HOME is not set. Run setup.sh or source \$TRACELOCK_HOME/env.sh." >&2
  exit 2
fi

PYTHON="${TRACELOCK_PYTHON:-${TRACELOCK_HOME}/conda/tracelock/bin/python}"
SAMPLES_DIR="${TRACELOCK_SAMPLES_DIR:-${TRACELOCK_HOME}/traces/dream_math_code/samples}"
OUTPUT_ROOT="${TRACELOCK_OUTPUT_ROOT:-${TRACELOCK_HOME}/checkpoints}"
RUN_NAME="tracelock-dream-math-code"
MAX_STEPS=36000
DEVICE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --samples-dir)
      SAMPLES_DIR="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --device)
      DEVICE_ARGS=(--device "$2")
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m tracelock.training.pretrain \
  --samples-dir "$SAMPLES_DIR" \
  --output-root "$OUTPUT_ROOT" \
  --run-name "$RUN_NAME" \
  --use-ae-data \
  --train-batch-size 64 \
  --eval-batch-size 64 \
  --num-workers 4 \
  --max-steps "$MAX_STEPS" \
  --eval-every 200 \
  --learning-rate 0.0001 \
  --weight-decay 0.01 \
  --pretrained-proj-checkpoint "$TRACELOCK_HOME/checkpoints/dream-ae-v1/best_val_loss.pt" \
  --model-type transformer \
  --d-model 4096 \
  --d-tracelock 256 \
  --d-tracelock-delta 32 \
  --d-x 384 \
  --num-encoder-layers 5 \
  --num-attention-heads 8 \
  --encoder-ffn-dim 768 \
  --max-gen-len 256 \
  --position-encoding \
  --no-state-encoding \
  --dynamic-threshold \
  --dynamic-threshold-decision-only \
  --target-mask-prob 0 \
  --target-mask-keep-loss-weight 1 \
  --input-noise-ratio 0 \
  --feature-noise-apply-probability 0 \
  --enable-random-crop \
  --random-crop-min-ratio 0.25 \
  --random-crop-max-ratio 1.0 \
  --no-random-crop-apply-to-val \
  --val-state2-only \
  --no-two-pass-reweight \
  --no-token-class-balance \
  --negative-loss-weight 2 \
  --frontier-positive-bias \
  --frontier-positive-bias-strength 0.5 \
  --threshold-center-loss-weight 0.001 \
  --rollout-proxy-threshold 0.99 \
  "${DEVICE_ARGS[@]}"
