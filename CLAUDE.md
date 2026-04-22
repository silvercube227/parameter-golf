# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

Parameter Golf is an OpenAI competition to train the best language model within **16MB of storage** in **under 10 minutes on 8×H100 GPUs**. Models are evaluated by **bits-per-byte (BPB)** on the FineWeb validation set — a tokenizer-agnostic metric. Lower BPB = better.

Challenge runs March 18 – April 30, 2026.

## Development Setup

### Mac (Apple Silicon) — local iteration
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm

# Download 1 shard for smoke testing
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

# Smoke test (~2 min)
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py
```

### CUDA (Runpod / H100s) — full training
```bash
pip install -r requirements.txt

# Download full dataset
python3 data/cached_challenge_fineweb.py --variant sp1024

# Single GPU
RUN_ID=run1 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 torchrun --standalone --nproc_per_node=1 train_gpt.py

# 8×H100 (leaderboard run)
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key Hyperparameters (set via environment variables)

| Variable | Default | Notes |
|---|---|---|
| `ITERATIONS` | 20000 | Training steps |
| `TRAIN_BATCH_TOKENS` | 524288 | Global batch size in tokens |
| `TRAIN_SEQ_LEN` | 1024 | Sequence length during training |
| `MAX_WALLCLOCK_SECONDS` | 600 | 10-minute cap for leaderboard |
| `WARMUP_STEPS` | 20 | LR warmup steps |
| `WARMDOWN_ITERS` | 1200 | Cosine warmdown steps at end |
| `NUM_LAYERS` | 9 | Transformer blocks |
| `MODEL_DIM` | 512 | Hidden dim |
| `NUM_HEADS` / `NUM_KV_HEADS` | 8 / 4 | GQA config |
| `MLP_MULT` | 2 | MLP expansion ratio |
| `TIE_EMBEDDINGS` | 1 | Share embed/lm_head weights |
| `VOCAB_SIZE` | 1024 | Must match tokenizer |
| `ROPE_BASE` | 10000.0 | RoPE frequency base |
| `LOGIT_SOFTCAP` | 30.0 | tanh softcap on output logits |
| `QAT` | 1 | Enable quantization-aware training |
| `EVAL_STRIDE` | 64 | Sliding window stride for validation |
| `EVAL_BATCH_SEQS` | 32 | Sequences per validation batch |
| `MATRIX_LR` | 0.04 | Muon optimizer LR |
| `EMBED_LR` | 0.6 | Adam LR for tied embeddings |
| `HEAD_LR` | 0.008 | Adam LR for untied lm_head |
| `SCALAR_LR` | 0.04 | Adam LR for scalar/vector params |
| `MUON_MOMENTUM` | 0.95 | Muon momentum (after warmup) |
| `SEED` | 1337 | Leaderboard requires 3+ seeds |

## Code Architecture

### Training Scripts

- **`train_gpt.py`** — PyTorch/CUDA baseline. Hard 1500-line limit for readability (submissions must stay under this). This is the canonical leaderboard script.
- **`train_gpt_mlx.py`** — Apple MLX port for local M-series development. Mirrors CUDA logic but slower to iterate on for final scores.

### Model (`train_gpt.py`)

`GPT` class with:
- Token embedding → RMSNorm → N transformer blocks → RMSNorm → logits
- **U-Net skip connections**: first half of layers store residuals, second half adds them back in reverse order
- **Logit softcap** via tanh (default 30.0) for training stability
- Learnable per-layer residual scales (`attn_scale`, `mlp_scale`, `resid_mix`)

`CausalSelfAttention`:
- Grouped Query Attention (GQA)
- RoPE positional embeddings
- Per-head Q-gain scaling
- Flash Attention 2 backend

`MLP`: ReLU² activation (`relu(x)²`), configurable expansion ratio.

### Optimizer Strategy

Four separate optimizers run simultaneously:
1. **Muon** — matrix-shaped parameters in transformer blocks (Newton-Schulz orthogonalization)
2. **Adam** — embeddings (`EMBED_LR`)
3. **Adam** — lm_head when not tied (`HEAD_LR`)
4. **Adam** — scalar/vector params (`SCALAR_LR`)

### Quantization & Compression Pipeline

Post-training steps (end of `train_gpt.py`):
1. Quantize to **int8** (per-row for matrices, per-tensor for vectors)
2. Small tensors (<65k params) kept as fp16
3. Control tensors (`attn_scale`, `mlp_scale`, `resid_mix`, `q_gain`, `skip_weights`) kept as fp32
4. Zlib compress at level 9
5. Round-trip validation ensures decompression works
6. Output: `final_model.int8.ptz` — must be ≤16MB **including** the training script bytes

An uncompressed `final_model.pt` is also saved for debugging but does not count toward the size limit. Some record submissions swap zlib for lzma or zstd to squeeze extra bytes.

### Data Pipeline

- **TokenStream** — sequential shard reader with infinite wrap-around
- **DistributedTokenLoader** — shards across GPU ranks
- Validation uses **BPB** (bits-per-byte), computed via SentencePiece byte metadata to be tokenizer-agnostic

## Submission Process

1. Fork the repo and create `records/track_10min_16mb/<date>_<name>/`
2. Include: `README.md`, `train_gpt.py` (≤1500 lines), `submission.json`, `train.log`
3. Run 3+ seeds, report mean val_bpb
4. Final score = mean BPB across seeds at the 10-minute mark
5. Open a PR — the leaderboard in README.md is updated on merge

`submission.json` required fields (see existing entries for examples):
```json
{
  "author": "...",
  "github_id": "...",
  "name": "...",
  "blurb": "...",
  "date": "YYYY-MM-DD",
  "track": "10min_16mb",
  "val_loss": 1.882,
  "val_bpb": 1.115,
  "val_loss_std": 0.0006,
  "val_bpb_std": 0.00035,
  "seeds": [314, 42, 999],
  "seed_results": {
    "314": { "val_loss": ..., "val_bpb": ..., "artifact_bytes": ..., "steps": ..., "step_avg_ms": ... },
    ...
  },
  "hardware": "8xH100 80GB SXM",
  "pytorch_version": "...",
  "cuda_version": "..."
}
```
Records that beat a prior submission also include `comparison_baseline_pr`, `delta_vs_prN_bpb`, `t_statistic`, and `welch_df`.

## Records Structure

```
records/
├── track_10min_16mb/        # Leaderboard submissions (time-constrained)
└── track_non_record_16mb/   # Experimental approaches (no time limit)
```

Each submission folder has its own `train_gpt.py` — to reproduce any result, run that script with the same seeds shown in its `train.log`.

## Evaluation Metric

BPB = `cross_entropy_loss / log(2) / bytes_per_token`

`bytes_per_token` is computed from SentencePiece piece lengths, making the metric independent of vocabulary size or tokenizer choice.
