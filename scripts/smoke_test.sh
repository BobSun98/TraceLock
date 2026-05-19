#!/usr/bin/env bash
set -euo pipefail

bash "$(dirname "${BASH_SOURCE[0]}")/generate_training_traces.sh" --num-samples 8 --devices "${1:-cuda:0}"
bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" --max-steps 20
bash "$(dirname "${BASH_SOURCE[0]}")/eval_math.sh" --num-samples 8 --gpus "${1:-cuda:0}"
bash "$(dirname "${BASH_SOURCE[0]}")/eval_code.sh" --num-samples 8 --gpus "${1:-cuda:0}"

