#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "${TRACELOCK_HOME:-}" ]]; then
  echo "TRACELOCK_HOME is not set. Run setup.sh or source \$TRACELOCK_HOME/env.sh." >&2
  exit 2
fi

PYTHON="${TRACELOCK_PYTHON:-${TRACELOCK_HOME}/conda/tracelock/bin/python}"
export TRACELOCK_CHECKPOINT_DIR="${TRACELOCK_CHECKPOINT_DIR:-${TRACELOCK_HOME}/checkpoints/tracelock-dream-math-code}"
NUM_SAMPLES=""
GPUS=()
RUN_NAME=""
SETS=()
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --gpus)
      shift
      GPUS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        GPUS+=("$1")
        shift
      done
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --sets)
      shift
      SETS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        SETS+=("$1")
        shift
      done
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

TMP_CONFIG="$TRACELOCK_HOME/runs/configs/eval_humaneval.$(date +%Y%m%d_%H%M%S).json"
MATERIALIZE_ARGS=(--template "$REPO_ROOT/configs/eval_humaneval.json" --output "$TMP_CONFIG")
if [[ -n "$NUM_SAMPLES" ]]; then
  MATERIALIZE_ARGS+=(--num-samples "$NUM_SAMPLES")
fi
if [[ ${#GPUS[@]} -gt 0 ]]; then
  MATERIALIZE_ARGS+=(--gpus "${GPUS[@]}")
fi
if [[ ${#SETS[@]} -gt 0 ]]; then
  MATERIALIZE_ARGS+=(--sets "${SETS[@]}")
fi

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "$SCRIPT_DIR/materialize_config.py" "${MATERIALIZE_ARGS[@]}"

RUN_ARGS=(--config "$TMP_CONFIG")
if [[ -n "$RUN_NAME" ]]; then
  RUN_ARGS+=(--run-name "$RUN_NAME")
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  RUN_ARGS+=(--dry-run)
fi

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m tracelock.dream.run_eval_code "${RUN_ARGS[@]}"
