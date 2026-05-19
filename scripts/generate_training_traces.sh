#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "${TRACELOCK_HOME:-}" ]]; then
  echo "TRACELOCK_HOME is not set. Run setup.sh or source \$TRACELOCK_HOME/env.sh." >&2
  exit 2
fi

PYTHON="${TRACELOCK_PYTHON:-${TRACELOCK_HOME}/conda/tracelock/bin/python}"
RUN_DIR="${TRACELOCK_TRACE_RUN_DIR:-${TRACELOCK_HOME}/traces/dream_math_code}"
NUM_SAMPLES=7000
DEVICES=(cuda:0)
LOCAL_FILES_ONLY=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --devices)
      shift
      DEVICES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        DEVICES+=("$1")
        shift
      done
      ;;
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --allow-downloads)
      LOCAL_FILES_ONLY=0
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

LOCAL_FLAG=(--local-files-only)
if [[ "$LOCAL_FILES_ONLY" -eq 0 ]]; then
  LOCAL_FLAG=()
fi

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m tracelock.dream.generate_pretrain_samples \
  --run-dir "$RUN_DIR" \
  --datasets gsm8k alpaca_cleaned kodcode_humaneval_like \
  --split train \
  --num-samples "$NUM_SAMPLES" \
  --batch-size 1 \
  --seed 42 \
  --model-id "$TRACELOCK_HOME/models/dream-v0-instruct-7b" \
  --devices "${DEVICES[@]}" \
  --torch-dtype bfloat16 \
  --feature-storage-dtype float16 \
  --projection-checkpoint "$TRACELOCK_HOME/checkpoints/dream-ae-v1/best_val_loss.pt" \
  --gen-length-choices 128 256 \
  --block-divisor-choices 8 16 32 \
  --temperature 0.0 \
  --top-p 0.95 \
  --alg entropy \
  --alg-temp 0.0 \
  --step-sample-ratio 0.3 \
  --val-ratio 0.1 \
  --max-run-dir-gb 600 \
  "${LOCAL_FLAG[@]}"

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "$SCRIPT_DIR/prepare_training_samples.py" \
  --trace-run-dir "$RUN_DIR" \
  --output-samples-dir "$RUN_DIR/samples" \
  --overwrite-links
