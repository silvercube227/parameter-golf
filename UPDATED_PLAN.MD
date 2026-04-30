Plan: Fix Artifact Size (20MB→<16MB) + Improve BPB (1.15→<1.07)

 Context

 Parameter Golf competition: train best LM in ≤16MB artifact, ≤10min on 8×H100. Evaluated by BPB (lower=better). As of April 23, 2026:
 - Merged SOTA: 1.0810 BPB (bigbag, PR #1493, April 9)
 - Best open PRs: 1.0638 BPB (PR #1787 Polar Express), 1.0645 BPB (PR #1736 CaseOps+GatedAttn)
 - Wild-card PR: 1.01925 BPB (PR #1785 PPM mixture, legality under "no n-gram cache" rule TBD)

 Current state: artifact ~20MB (too large), BPB ~1.15 (gap of ~0.09 vs realistic open-PR SOTA ~1.064).

 ---
 Root-Cause Analysis

 Why artifact is 20MB (limit: 16MB)

 1. Brotli not installed → falls back to LZMA → ~1.3× vs Brotli's ~1.7× (with byte-shuffle)
 2. Missing byte-shuffle transform → SOTA uses byte-interleave before Brotli for much better compression of int6 data
 3. Code bytes ~90KB → counts toward 16MB limit; SOTA compresses code via lzma+base85 to ~16.6KB (saves ~74KB)
 4. Missing weight decay → weights are larger/less zero-heavy → harder to compress

 Why BPB is 1.15 (gap vs merged SOTA 1.081)

 ┌─────────────────────────────┬────────────────────┬────────────────────────────┐
 │           Feature           │        Ours        │   Merged SOTA (PR #1493)   │
 ├─────────────────────────────┼────────────────────┼────────────────────────────┤
 │ Weight decay (Muon)         │ None               │ WD=0.095                   │
 ├─────────────────────────────┼────────────────────┼────────────────────────────┤
 │ Weight decay (Adam scalars) │ None               │ 0.02                       │
 ├─────────────────────────────┼────────────────────┼────────────────────────────┤
 │ Weight decay (embeddings)   │ None               │ 0.085                      │
 ├─────────────────────────────┼────────────────────┼────────────────────────────┤
 │ Grad clip                   │ 0.0 (off)          │ 0.3                        │
 ├─────────────────────────────┼────────────────────┼────────────────────────────┤
 │ Warmdown                    │ 4000 steps (~20%)  │ 72% of training            │
 ├─────────────────────────────┼────────────────────┼────────────────────────────┤
 │ EMA decay                   │ 0.997              │ 0.9965                     │
 ├─────────────────────────────┼────────────────────┼────────────────────────────┤
 │ Matrix LR                   │ 0.04               │ 0.022                      │
 ├─────────────────────────────┼────────────────────┼────────────────────────────┤
 │ Recurrence                  │ L4-5 × 2×          │ L3-5 × 3×                  │
 ├─────────────────────────────┼────────────────────┼────────────────────────────┤
 │ Progressive recur           │ Always on          │ Enabled at 35% of training │
 ├─────────────────────────────┼────────────────────┼────────────────────────────┤
 │ SDClip for matrices         │ Fixed 99.99984 pct │ Sigma-search (k=12.85)     │
 └─────────────────────────────┴────────────────────┴────────────────────────────┘

 Missing weight decay is the single largest gap. Then warmdown length, grad clip, and recurrence config.

 What PR #1736 / #1787 add beyond merged SOTA (gap 1.081→1.064)

 - CaseOps tokenizer — bijective case-folding transformation (TITLE, ALLCAPS, CAPNEXT, ESC tokens) + byte sidecar accounting; better vocab use.
  Complex, requires preprocessing change.
 - Learned attention output gate — small gated output scaling per block, initialized ~0.
 - Phased TTT with LoRA — LoRA adapters on attention, reset per-document, scores first then adapts; more powerful than our SGD-on-all-params
 TTT.
 - Polar Express Newton-Schulz — per-step minimax NS coefficients in Muon (replaces fixed (3.4445, -4.7750, 2.0315)). Small but free gain.
 - MIN_LR warmdown floor — LR floors at 10% of max instead of 0 (reclaims final training efficiency).
 - Recurrence Depth Curriculum — train with depth 1 → depth 3 → depth 4 across three phases (rather than fixed depth or single switch-on).
 ~0.001-0.003 BPB.
 - Sparse Attn Head-Output Gate — reduce gate from 4096→96 params per layer, saves 44KB.

 ---
 Implementation Plan

 File: /Users/bennettye/Documents/GitHub/parameter-golf/train_gpt.py (1496 lines; 1500-line limit)
 Note: code compression (item 10) reduces the submitted file to 2 lines; 1500-line limit applies to the wrapper.

 ---
 Tier 1 — Critical (fixes size + closes most of BPB gap)

 1. Weight decay — Muon (lines 109–168)

 Add weight_decay: float = 0.0 to Muon.__init__. In step(), after applying the update:
 p.mul_(1.0 - lr * weight_decay)
 Add to Hyperparameters:
 muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", 0.095))
 Pass to Muon(...) constructor at line 1183.

 2. Weight decay — Adam optimizers (lines 1177–1214)

 Add:
 adam_wd   = float(os.environ.get("ADAM_WD",   0.02))
 embed_wd  = float(os.environ.get("EMBED_WD",  0.085))
 Add weight_decay=args.embed_wd to optimizer_tok, weight_decay=args.adam_wd to optimizer_scalar and optimizer_head.

 3. Gradient clipping default (line 91)

 grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))  # was 0.0

 4. Warmdown fraction (lines 43, 1253–1261)

 Add:
 warmdown_frac = float(os.environ.get("WARMDOWN_FRAC", 0.72))
 Update get_lr_scale(step) to compute warmdown_start as iterations * (1 - warmdown_frac) when warmdown_frac > 0, overriding the fixed
 warmdown_iters fallback.

 5. EMA / LR defaults (lines 43, 66, 81)

 warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 0))   # 0 = use warmdown_frac
 ema_decay      = float(os.environ.get("EMA_DECAY",    0.9965))
 matrix_lr      = float(os.environ.get("MATRIX_LR",    0.022))

 6. Recurrence defaults (lines 72–74)

 recur_start = int(os.environ.get("RECUR_START", 3))  # was 4
 recur_end   = int(os.environ.get("RECUR_END",   6))  # unchanged
 recur_depth = int(os.environ.get("RECUR_DEPTH", 3))  # was 2

 7. Progressive recurrence enable (new, ~6 lines in training loop)

 Add:
 enable_recur_frac = float(os.environ.get("ENABLE_RECUR_FRAC", 0.35))
 In the training loop, track elapsed_frac. Before enable_recur_frac, call base_model.set_recur_depth(1). At threshold, call
 base_model.set_recur_depth(args.recur_depth). Add trivial set_recur_depth(d) method to GPT.

 ---
 Tier 2 — Important for BPB (gets toward 1.07)

 8. SDClip for all 2D matrices (lines 335–348)

 Add MATRIX_QUANT_CLIP_SIGMAS env var (default "12.85"). In quantize_float_tensor, for 2D tensors, try sigma-based clip candidates:
 MATRIX_QUANT_CLIP_SIGMAS = tuple(float(x) for x in os.environ.get("MATRIX_QUANT_CLIP_SIGMAS", "12.85").split(",") if x)
 In the 2D branch: compute row_std, try each sigma as clip_abs = max(std * sigma, absmax / qmax), score by reconstruction MSE, keep best.
 (Similar to existing quantize_embedding_gptq but for all matrices.)

 9. MIN_LR warmdown floor (lines 1253–1261)

 Add:
 min_lr_frac = float(os.environ.get("MIN_LR_FRAC", 0.1))
 In get_lr_scale, clamp return value: return max(scale, min_lr_frac).

 10. Recurrence Depth Curriculum (lines ~57, training loop)

 Add:
 recur_phase_depths = [int(x) for x in os.environ.get("RECUR_PHASE_DEPTHS", "1,3,3").split(",")]
 In training loop, determine current phase (thirds of total iterations/wallclock), call base_model.set_recur_depth(recur_phase_depths[phase]).
 Supersedes the simpler progressive enable in Tier 1 if both are used.

 11. Polar Express Newton-Schulz (lines 94–107)

 Replace the single (a, b, c) = (3.4445, -4.7750, 2.0315) triple with per-iteration minimax tuples from PR #1344:
 _NS_COEFFS = [(3.4445, -4.7750, 2.0315), ...]  # 5 optimized triples for steps 0-4
 Each NS iteration uses its own coefficients. Drop-in replacement, no other changes.

 ---
 Tier 3 — Compression (critical for size budget)

 12. Byte-shuffle before Brotli (lines 1430–1434)

 Add byte_shuffle / byte_unshuffle helpers. Before compressing with Brotli, apply byte-interleave:
 def byte_shuffle(buf: bytes, stride: int = 4) -> bytes:
     a = np.frombuffer(buf + bytes(-len(buf) % stride), dtype=np.uint8)
     return a.reshape(-1, stride).T.ravel().tobytes()
 The inverse reorders back in dequantize / round-trip load path. This transforms int6-packed weights so zero-biased high bits cluster together
 → Brotli achieves ~1.7× vs ~1.3× without shuffle.

 13. Code compression wrapper (applied last)

 After all changes stabilize, compress train_gpt.py to a 2-line wrapper:
 import lzma as L, base64 as B
 exec(L.decompress(B.b85decode("<base85-lzma-of-source>"), format=L.FORMAT_RAW, filters=[{"id": L.FILTER_LZMA2}]))
 Reduces code bytes from ~90KB → ~16KB (saves ~74KB). Run pip install brotli on the training machine.

 ---
 Tier 4 — Advanced (gets toward 1.064 if time permits)

 14. Learned Attention Output Gate (moderate complexity)

 Add attn_out_gate = nn.Parameter(torch.zeros(num_heads)) to CausalSelfAttention. After computing y, apply y = y *
 torch.tanh(attn_out_gate)[None, :, None, None]. Zero-init ensures stability. Add to control tensors (keep as fp32).

 15. Phased TTT with LoRA adapters (high complexity)

 Replace current eval_val_sliding SGD-on-all-params with LoRA-adapter TTT:
 - Add small LoRA rank-4 adapters for Q/V projections (separate from training LoRAs)
 - During TTT: only adapt LoRA params; reset per-document boundary
 - Score first, then update LoRA — same score-first compliance
 - Removes the risk of large weight perturbations from full-param SGD

 16. PPM adaptive-λ mixture (very complex, legality uncertain)

 Byte-level Prediction by Partial Matching (order-5) mixed with NN logits at eval time. Achieves ~−0.12 BPB gain (PR #1785 gets 1.019).
 However, this may violate the "no n-gram cache" rule — verify competition compliance before implementing.

 ---
 Implementation Order

 1. Items 1–7 (Tier 1) — implement together in one pass, test locally
 2. Items 8–11 (Tier 2) — add next, validate size still fits
 3. Item 12 (byte-shuffle) — validate round-trip correctness first
 4. Item 13 (code compression) — final step
 5. Items 14–15 (Tier 4) — if time and size budget allow

 ---
 Critical Files

 ┌────────────────────────┬─────────────────────────────────────────────┐
 │         Lines          │                   Purpose                   │
 ├────────────────────────┼─────────────────────────────────────────────┤
 │ train_gpt.py:109–168   │ Muon optimizer → add weight_decay           │
 ├────────────────────────┼─────────────────────────────────────────────┤
 │ train_gpt.py:1177–1214 │ Optimizer construction → add wd to Adam     │
 ├────────────────────────┼─────────────────────────────────────────────┤
 │ train_gpt.py:91        │ GRAD_CLIP_NORM default                      │
 ├────────────────────────┼─────────────────────────────────────────────┤
 │ train_gpt.py:43,66,81  │ WARMDOWN, EMA, MATRIX_LR defaults           │
 ├────────────────────────┼─────────────────────────────────────────────┤
 │ train_gpt.py:72–74     │ RECUR_START/END/DEPTH defaults              │
 ├────────────────────────┼─────────────────────────────────────────────┤
 │ train_gpt.py:335–348   │ quantize_float_tensor → SDClip              │
 ├────────────────────────┼─────────────────────────────────────────────┤
 │ train_gpt.py:94–107    │ zeropower_via_newtonschulz5 → Polar Express │
 ├────────────────────────┼─────────────────────────────────────────────┤
 │ train_gpt.py:1430–1434 │ Compress path → byte-shuffle                │
 ├────────────────────────┼─────────────────────────────────────────────┤
 │ train_gpt.py:883–901   │ GPT._run_blocks → recur curriculum          │
 ├────────────────────────┼─────────────────────────────────────────────┤
 │ train_gpt.py:1350–1360 │ Training loop → progressive recur gate      │
 └────────────────────────┴─────────────────────────────────────────────┘

 ---
 Expected Outcomes

 ┌────────────────────────────────────┬────────────────┬───────────────────┐
 │               Change               │ Est. BPB delta │  Est. size delta  │
 ├────────────────────────────────────┼────────────────┼───────────────────┤
 │ Weight decay (WD=0.095/0.02/0.085) │ −0.030         │ −0.5MB compressed │
 ├────────────────────────────────────┼────────────────┼───────────────────┤
 │ Extended warmdown (72%)            │ −0.010         │ —                 │
 ├────────────────────────────────────┼────────────────┼───────────────────┤
 │ Grad clip 0.3                      │ −0.005         │ —                 │
 ├────────────────────────────────────┼────────────────┼───────────────────┤
 │ Recurrence L3-5 × 3 + curriculum   │ −0.007         │ —                 │
 ├────────────────────────────────────┼────────────────┼───────────────────┤
 │ Progressive recur (0.35)           │ −0.003         │ —                 │
 ├────────────────────────────────────┼────────────────┼───────────────────┤
 │ EMA 0.9965 + matrix LR 0.022       │ −0.004         │ —                 │
 ├────────────────────────────────────┼────────────────┼───────────────────┤
 │ SDClip for matrices                │ −0.005         │ −0.3MB            │
 ├────────────────────────────────────┼────────────────┼───────────────────┤
 │ MIN_LR floor (10%)                 │ −0.002         │ —                 │
 ├────────────────────────────────────┼────────────────┼───────────────────┤
 │ Polar Express NS                   │ −0.002         │ —                 │
 ├────────────────────────────────────┼────────────────┼───────────────────┤
 │ Byte-shuffle + Brotli              │ —              │ −2.0MB            │
 ├────────────────────────────────────┼────────────────┼───────────────────┤
 │ Code compression                   │ —              │ −74KB             │
 ├────────────────────────────────────┼────────────────┼───────────────────┤
 │ Tier 1–3 total                     │ ~−0.068        │ ~−2.9MB           │
 ├────────────────────────────────────┼────────────────┼───────────────────┤
 │ Tier 4 (gated attn + LoRA-TTT)     │ −0.010         │ —                 │
 ├────────────────────────────────────┼────────────────┼───────────────────┤
 │ Grand total                        │ ~1.082→~1.07   │ ~17.1→~14.2MB     │
 └────────────────────────────────────┴────────────────┴───────────────────┘

 Note: Our baseline ~20MB is likely final_model.pt (fp32 uncompressed). The actual final_model.quant.ptz should already be closer to 16-17MB.
 After byte-shuffle + Brotli, target is ~15MB model + ~16KB code = ~15MB total.