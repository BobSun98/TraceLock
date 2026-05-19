---
license: apache-2.0
library_name: pytorch
tags:
  - tracelock
  - dream
  - diffusion-language-model
  - activation-autoencoder
  - pytorch
---

# TraceLock Dream Activation Autoencoder

This repository contains the projection autoencoder checkpoint used to reproduce TraceLock on Dream.

TraceLock is a token-level acceptance policy for Dream-style masked diffusion generation. Dream proposes candidate tokens during the denoising loop, and TraceLock decides which positions should be locked now versus kept masked for later refinement.

## What This Checkpoint Is

`best_val_loss.pt` is an activation autoencoder for Dream hidden states. It compresses the last three Dream hidden-state snapshots and two hidden-state deltas into compact features consumed by the TraceLock policy model.

This checkpoint is not a text generation model and does not contain Dream model weights. Users still need to download Dream from its original repository:

```text
Dream-org/Dream-v0-Instruct-7B
```

## How It Is Used

After downloading this repository into a TraceLock workspace, the expected local path is:

```text
$TRACELOCK_HOME/checkpoints/dream-ae-v1/best_val_loss.pt
```

TraceLock uses this checkpoint in two places:

1. `generate_training_traces.sh`: projects Dream activations while building training traces.
2. `train.sh` / evaluation: reconstructs the same projection stack expected by the TraceLock policy.

## Architecture

The released checkpoint was trained with:

```json
{
  "d_model": 3584,
  "d_hidden_bottleneck": 256,
  "d_delta_bottleneck": 32,
  "dropout": 0.1
}
```

The exported projection state contains:

- hidden-state normalization
- delta-state normalization
- hidden-state projection encoder
- delta-state projection encoder

## Files

- `best_val_loss.pt`: projection autoencoder checkpoint.
- `config.json`: training/configuration metadata for this autoencoder run.
- `data_stats.json`: basic sample count and batch metadata from the run.

## Citation

If you use this checkpoint, please cite the TraceLock paper/repository once available.

