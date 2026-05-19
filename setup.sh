#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash TraceLock/setup.sh --workspace /path/to/workspace [--download-assets]

Creates a conda environment under the workspace and optionally downloads
Dream, datasets, and TraceLock checkpoints into the same workspace.
EOF
}

WORKSPACE=""
DOWNLOAD_ASSETS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)
      WORKSPACE="${2:-}"
      shift 2
      ;;
    --download-assets)
      DOWNLOAD_ASSETS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$WORKSPACE" ]]; then
  echo "--workspace is required." >&2
  usage >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRACELOCK_HOME="$(mkdir -p "$WORKSPACE" && cd "$WORKSPACE" && pwd)"
export TRACELOCK_REPO="$REPO_ROOT"
export HF_HOME="$TRACELOCK_HOME/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

mkdir -p \
  "$TRACELOCK_HOME/conda" \
  "$TRACELOCK_HOME/hf_cache" \
  "$TRACELOCK_HOME/models" \
  "$TRACELOCK_HOME/datasets" \
  "$TRACELOCK_HOME/checkpoints" \
  "$TRACELOCK_HOME/traces" \
  "$TRACELOCK_HOME/runs" \
  "$TRACELOCK_HOME/logs"

ENV_PREFIX="$TRACELOCK_HOME/conda/tracelock"
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  conda env create --prefix "$ENV_PREFIX" --file "$REPO_ROOT/environment.yml"
else
  conda env update --prefix "$ENV_PREFIX" --file "$REPO_ROOT/environment.yml" --prune
fi

cat > "$TRACELOCK_HOME/env.sh" <<EOF
export TRACELOCK_HOME="$TRACELOCK_HOME"
export TRACELOCK_REPO="$TRACELOCK_REPO"
export HF_HOME="$HF_HOME"
export HF_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET"
export KODCODE_HUMANEVAL_SOURCE="hf"
export KODCODE_HUMANEVAL_REPO_ID="BOB12311/kodcode-humaneval-like"
export TRACELOCK_AE_REPO_ID="${TRACELOCK_AE_REPO_ID:-BOB12311/tracelock-dream-ae}"
EOF

if [[ "$DOWNLOAD_ASSETS" -eq 1 ]]; then
  "$ENV_PREFIX/bin/python" "$REPO_ROOT/scripts/download_assets.py" \
    --workspace "$TRACELOCK_HOME"
fi

echo "TraceLock workspace: $TRACELOCK_HOME"
echo "No conda activation is required by the TraceLock scripts."
echo "Load workspace variables with: source $TRACELOCK_HOME/env.sh"
echo "Scripts will use: $ENV_PREFIX/bin/python"
