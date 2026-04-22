 ---
 Current Leaderboard (as of 2026-04-21)

 ┌────────┬─────────────────────────────────────────────────────────────────────────────┬──────────────────┐
 │  BPB   │                               Key techniques                                │      Author      │
 ├────────┼─────────────────────────────────────────────────────────────────────────────┼──────────────────┤
 │ 1.0810 │ SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT │ bigbag           │
 ├────────┼─────────────────────────────────────────────────────────────────────────────┼──────────────────┤
 │ 1.0822 │ SP8192 + Parallel Residuals + Legal TTT                                     │ aryanbhosale     │
 ├────────┼─────────────────────────────────────────────────────────────────────────────┼──────────────────┤
 │ 1.0828 │ SP8192 + QK-Gain 5.0 + Legal TTT                                            │ dexhunter        │
 ├────────┼─────────────────────────────────────────────────────────────────────────────┼──────────────────┤
 │ 1.0835 │ SP8192 + Hessian SDClip + Progressive Recurrence                            │ Robby Sneiderman │
 ├────────┼─────────────────────────────────────────────────────────────────────────────┼──────────────────┤
 │ 1.0856 │ SP8192 + GPTQ Embeddings + Depth Recurrence (layers 4-5 ×2) + SDClip        │ Kevin Clark      │
 ├────────┼─────────────────────────────────────────────────────────────────────────────┼──────────────────┤
 │ 1.0897 │ SP4096 + Depth Recurrence + Parallel Residuals + MuonEq-R + QK-Gain 5.0     │ aryanbhosale     │
 ├────────┼─────────────────────────────────────────────────────────────────────────────┼──────────────────┤
 │ 1.0912 │ MuonEq-R + Depth Recurrence + WD=0.090 + All-Int6 GPTQ                      │ dexhunter        │
 ├────────┼─────────────────────────────────────────────────────────────────────────────┼──────────────────┤
 │ 1.0979 │ SP4096 + 4× MLP + high WD (TTT, hash, SmearGate removed)                    │ Kevin Clark      │
 ├────────┼─────────────────────────────────────────────────────────────────────────────┼──────────────────┤
 │ 1.1063 │ Parallel Residuals + Mini Depth Recurrence                                  │ Marko Sisovic    │
 ├────────┼─────────────────────────────────────────────────────────────────────────────┼──────────────────┤
 │ 1.1147 │ AR Self-Gen GPTQ + XSA-all + BigramHash 3072×112                            │ abaybektursun    │
 ├────────┼─────────────────────────────────────────────────────────────────────────────┼──────────────────┤
 │ 1.1194 │ LeakyReLU² + Legal TTT + Parallel Muon                                      │ abaybektursun    │
 ├────────┼─────────────────────────────────────────────────────────────────────────────┼──────────────────┤
 │ 1.2244 │ Naive Baseline                                                              │ —                │
 └────────┴─────────────────────────────────────────────────────────────────────────────┴──────────────────┘

 The local repo contains submissions through 2026-03-25. The April breakthroughs (SP8192, depth recurrence, parallel residuals, MuonEq-R) are
 only in the README leaderboard from merged PRs not yet cloned locally.

 ---
 Technique Wave Overview

 The competition evolved in identifiable waves:

 ┌──────┬──────────────┬─────────────┬──────────────────────────────────────────────────────────────────────────────────┐
 │ Wave │    Dates     │  BPB range  │                                  Dominant ideas                                  │
 ├──────┼──────────────┼─────────────┼──────────────────────────────────────────────────────────────────────────────────┤
 │ 1    │ Mar 18-20    │ 1.22 → 1.15 │ Int6 quant, 3× MLP, SmearGate, BigramHash, OrthoInit, Muon WD, SWA, sliding eval │
 ├──────┼──────────────┼─────────────┼──────────────────────────────────────────────────────────────────────────────────┤
 │ 2    │ Mar 20-22    │ 1.15 → 1.12 │ XSA, U-Net skips, Partial RoPE, LN scale, EMA, GPTQ-lite                         │
 ├──────┼──────────────┼─────────────┼──────────────────────────────────────────────────────────────────────────────────┤
 │ 3    │ Mar 22-25    │ 1.12 → 1.11 │ Full Hessian GPTQ, AR calibration, LeakyReLU², Parallel Muon, Legal TTT          │
 ├──────┼──────────────┼─────────────┼──────────────────────────────────────────────────────────────────────────────────┤
 │ 4    │ Mar 25-Apr 5 │ 1.11 → 1.09 │ SP4096/8192, depth recurrence, parallel residuals, MuonEq-R, SDClip              │
 ├──────┼──────────────┼─────────────┼──────────────────────────────────────────────────────────────────────────────────┤
 │ 5    │ Apr 5-9      │ 1.09 → 1.08 │ SP8192 + 3-layer recurrence + parallel residuals + QK-Gain 5.25 + TTT            │
 └──────┴──────────────┴─────────────┴──────────────────────────────────────────────────────────────────────────────────┘

 ---
 Deep-Dive Explanations (with Math)

 1. Quantization: Int6 → Full Hessian GPTQ

 Standard int8 per-row quantization (baseline):

 scale_i = max(|W[i, :]|) / 127
 W_q[i, j] = round(W[i, j] / scale_i) × scale_i

 This minimizes element-wise error but ignores that weights have different impact on layer output.

 GPTQ (Frantar et al., 2022) minimizes the layer output error δL = ||WX − W_qX||²_F using the Hessian H = 2XX^T:

 δL ≈ ½ δw^T H δw

 For each column q, the optimal quantization error δw_q* and the correction applied to remaining columns is:

 δw_q* = round(w_q, Δ) − w_q
 update: w_{q'} ← w_{q'} − (δw_q* / H^{-1}_{qq}) × H^{-1}_{qq'} for q' > q

 The Cholesky reformulation makes this numerically stable and O(d³) instead of naively O(d⁴):

 H = LL^T  (Cholesky decomp)
 H^{-1} = L^{-T} L^{-1}  (computed once)

 Why GPTQ beats per-row MSE clipping:
 - Percentile/MSE clipping (GPTQ-lite) reduces large weights but ignores covariance — two correlated weights may both be clipped, compounding
 error
 - Full Hessian GPTQ accounts for covariance: if w_q is large but H^{-1}{qq} is small, the correction is negligible; if H^{-1}{qq} is large,
 compensation is spread across remaining columns
 - BPB gain: ~−0.005 on the March stack

 AR Self-Generated Calibration:
 GPTQ requires calibration data to compute H = Σᵢxᵢxᵢᵀ. Using the validation set is illegal (you'd be training on test data). The solution:
 after training, run the model autoregressively to generate N sequences, collect activations, and compute Hessians from those. This is legal
 because you're not accessing external data during quantization.

 Int6 vs Int8:
 Int6 uses 6 bits per weight (range −31…31) instead of 8 (range −127…127). The 2-bit saving per weight reduces model size by 25%, enabling
 either: (a) a larger model within 16 MB, or (b) using the freed bytes for bigger BigramHash tables.

 SDClip (Std-based GPTQ Clipping):
 Rather than testing fixed percentiles, SDClip clips the weight range to ±k×σ where σ is the per-row standard deviation, then sweeps k ∈ {2.5,
 3.0, 3.5, 4.0, 4.5} and picks the k that minimizes reconstruction MSE. This is more theoretically motivated (Gaussian weights → clipping at 3σ
  captures 99.73%) and adapts to the actual weight distribution.

 GPTQ Embeddings:
 Embedding rows are accessed only once per token in a forward pass — they are "lookup tables," not weight matrices in the traditional sense.
 But embedding tables are large (8192 × 512 = 4M weights at fp16 = 8 MB alone). Applying GPTQ to embeddings with per-row Hessian estimates
 (from token frequency weighting) can reduce embedding size to int4 or int6, freeing megabytes for other parameters.

 ---
 2. Vocabulary Size: SP1024 → SP4096 → SP8192

 BPB is tokenizer-agnostic by definition:

 BPB = cross_entropy_loss / log(2) / bytes_per_token

 where bytes_per_token is the average UTF-8 bytes per SentencePiece token. This cancels out the tokenizer compression, making BPB a pure
 measure of the model's conditional language modeling ability.

 Why bigger vocab helps:
 - SP1024 gives ~1.0-1.5 bytes/token (very coarse; every subword is 1-1.5 characters)
 - SP8192 gives ~3-4 bytes/token (closer to standard BPE)
 - A larger vocab means the model sees fewer tokens per second of training, but each token carries more information
 - The model must predict fewer total tokens to cover the same text, and each prediction is "easier" (less ambiguous)
 - Empirically: going 1024→4096 gave ~−0.018 BPB; 4096→8192 gave another ~−0.008 BPB

 Model size constraint:
 With vocab=8192, the tied embedding/head matrix is 8192 × 512 × 2 bytes = 8 MB in fp16 — over half the budget. This forces using GPTQ on
 embeddings and careful size budgeting. Kevin Clark (1.0856) uses GPTQ embeddings to handle this.

 Why 8192 is the sweet spot (so far):
 Going beyond 8192 (e.g., 16384) would make the embedding table too large. The challenge at 8192 is: GPTQ embeddings + careful quantization to
 fit remaining weights in the budget.

 ---
 3. Depth Recurrence (Looped Layers)

 Standard transformer: Each layer l runs once per forward pass. Effective depth = num_layers.

 Depth recurrence: Specific layers run multiple times, sharing weights across loops:

 h₀ = embedding(x)
 for t in range(recurrence_depth):
     h_{t+1} = transformer_block_4(h_t)   # same weights, different pass
     h_{t+1} = transformer_block_5(h_{t+1})  # same weights
 h_final = remaining_layers(h_recur_out)

 Why this helps under a 16 MB constraint:
 - You get "effective depth" of, say, 13 layers using the weights of only 11 layers
 - Weight sharing means recurrent layers cost 0 extra bytes in the model artifact
 - The iterative refinement allows the recurrent block to perform gradient descent-like computations (implicitly — this is related to unrolled
 optimization)
 - BPB gain over non-recurrent: ~−0.008 to −0.015

 Parcae scaling law (Together AI, 2026): Optimal recurrence scales as C^0.40 where C is total compute budget. For 10 minutes on 8×H100s, this
 suggests 2-3 recurrence steps is near-optimal.

 Stability: Residual explosion is a risk. Solutions used in the competition:
 - Spectral norm constraints on recurrent block weights
 - Progressive recurrence: start with 1 loop, gradually increase to 3 during training warmup
 - Initialize recurrent blocks to near-identity (skip connections help)

 Key insight: The recurrent block essentially becomes an iterative refinement step — like running a "smaller transformer" multiple times to
 converge to a good representation, instead of stacking more unique layers.

 ---
 4. Parallel Residuals

 Standard transformer (sequential):

 h = h + Attn(LN(h))     # attention adds to residual stream
 h = h + MLP(LN(h))      # MLP adds to residual stream

 Parallel residuals (from Kimi Attention Residuals, 2026):

 h_attn = Attn(LN(h))
 h_mlp  = MLP(LN(h))
 h_new = h + α × h_attn + β × h_mlp   # both computed from same h

 where α, β can be fixed or learnable scalars.

 Why this helps:
 1. Training dynamics: In sequential residuals, the MLP always sees attention-processed input. In parallel, both attention and MLP see the same
  pre-block representation — neither is "downstream" of the other
 2. Gradient flow: Both paths receive direct gradient signal from the loss, avoiding the "gradient thinning" that occurs when gradients must
 pass through sequential non-linearities
 3. Step time: Attention and MLP can be computed simultaneously (overlapping their CUDA streams) — no serialization penalty on modern hardware
 4. BPB gain: ~−0.005 to −0.010

 Mathematical analysis:
 In sequential blocks, each block applies f: h → h + g(h). After L layers:
 h_L = h₀ + Σₗ gₗ(hₗ₋₁)

 The function gₗ(hₗ₋₁) depends on all previous transformations, making optimization a chain of dependencies. In parallel, gₗ(h₀) (or gₗ(hₗ₋₁)
 where hₗ₋₁ is the output of the previous full block, not intermediate) allows each block to "reason independently" about the same input.

 This is related to PaLM's parallel attention+FFN formulation:
 y = x + attn(x) + ffn(x)  [PaLM parallel]

 ---
 5. Exclusive Self Attention (XSA)

 Problem: Standard self-attention has a "self-attention bias" — each token's query attends heavily to its own key/value, because the token
 contains the most information about itself. This wastes attention capacity on trivially attending to self.

 XSA removes the self-aligned component from each head's output:

 # Standard attention output per head:
 y = Attn(Q, K, V)   # shape [B, T, H, D]

 # XSA correction (efficient GQA-aware):
 v_norm = F.normalize(V, dim=-1)            # [B, T, Hkv, D]
 group = H // Hkv
 y_g = y.reshape(B, T, Hkv, group, D)
 proj = (y_g * v_norm.unsqueeze(-2)).sum(-1, keepdim=True) * v_norm.unsqueeze(-2)
 y_xsa = (y_g - proj).reshape(B, T, H, D)  # subtract self-projection

 Mathematically: for each head h and query position t, let v̂ = V[t] / ‖V[t]‖. The XSA output is:
 y_h[t] = Attn_h[t] − (Attn_h[t] · v̂[t]) v̂[t]

 This is an orthogonal projection onto the complement of the value direction — it forces the attention output to be "orthogonal to self,"
 meaning the head can only carry information about other positions.

 Why it works: Early layers can still use full self-attention (they need to build per-token representations). Applying XSA to all 11 layers (as
  in the best submission) forces every layer to extract cross-token information, maximizing the model's use of context.

 Cost: ~2 ms/step overhead for 11-layer XSA. Zero new parameters.

 ---
 6. LeakyReLU(0.5)² Activation

 Standard ReLU² in MLP:
 out = relu(Wx + b)² = max(0, Wx+b)²

 This zeroes all negative pre-activations, creating "dead neurons" — units that never activate for any input. In a 16 MB constrained model,
 wasting neurons is costly.

 LeakyReLU(0.5)²:
 out = leaky_relu(Wx + b, slope=0.5)² = max(0.5x, x)² for x = Wx+b

 For x > 0: same as ReLU² (quadratic growth)
 For x < 0: (0.5x)² = 0.25x² (still quadratic, just scaled down by 0.25)

 Key properties:
 - No dead neurons: all units contribute gradient everywhere
 - Output is always non-negative (preserves the inductive bias of ReLU²)
 - Negative pre-activations get a "second chance" — they influence subsequent layers with a smaller but nonzero contribution
 - Gradient: d/dx max(0.5x, x)² = 2×(negative activations get slope 0.5 = 0.5²×2x = x)
 - BPB gain: −0.003 (confirmed in ablation)

 Why slope=0.5 specifically: This preserves 25% of the negative activation magnitude in the output (0.5² = 0.25). Lower slopes approach ReLU²
 (all negative info lost); slope=1.0 is just x² (no rectification, poor inductive bias).

 ---
 7. BigramHash Embedding + SmearGate

 BigramHash: Standard embeddings give each token a vector based only on its identity. BigramHash additionally looks up a vector based on the
 pair (prev_token, curr_token):

 hash_idx = (prev_tok * 36313) XOR (curr_tok * 27191) % bigram_vocab_size
 bigram_emb = bigram_table[hash_idx]   # [B, T, bigram_dim]
 bigram_proj = linear(bigram_emb)      # [B, T, model_dim]
 h = tok_emb + bigram_proj

 The XOR-based hash avoids modular arithmetic collisions while being fast. With 3072 buckets × dim=112, this table is (3072 × 112) × 2 bytes ≈
 689 KB in fp16, giving rich bigram statistics within a small memory budget.

 Why it helps: Language has strong bigram statistics (e.g., "the" almost always follows a specific set of words). Encoding this at the
 embedding level gives every layer downstream access to this context without attention.

 SmearGate: A learned per-dimension sigmoid gate that "smears" each token's embedding with its predecessor:

 g = sigmoid(gate)       # [model_dim], gate is a learnable parameter
 x_prev = shift_right(x) # [B, T, model_dim], pad first position with zeros
 h = (1 - g) * x + g * x_prev

 This allows the model to softly blend adjacent tokens at the embedding level. With only 512 parameters (one gate per dimension), it captures
 strong local dependencies without attention overhead.

 Combined effect: Both techniques give the transformer's first layer "pre-processed" input that already encodes bigram statistics, reducing the
  number of attention layers needed to capture basic local context.

 ---
 8. Partial RoPE + LN Scale

 Standard RoPE: Rotates all d_head dimensions of Q and K by position-dependent angles:
 θᵢ = base^(-2i/d),  i = 0, ..., d/2-1
 Q_rotated[t] = Q[t] × R(θ, t)

 Partial RoPE (rope_dims=16): Only rotates the first 16 of 64 head dimensions:
 Q[:16] = rotate(Q[:16], θ, t)
 Q[16:] = Q[16:]   # unchanged (position-invariant)

 The 48 unchanged dimensions can attend position-invariantly — useful for content-based retrieval that doesn't depend on where the token
 appears. This is a soft prior toward learning position-sensitive vs position-invariant features in different subspaces.

 LN Scale factor 1/√(layer+1):
 After each RMSNorm, multiply by a damping factor:
 LN_output *= 1 / sqrt(layer_idx + 1)

 Layer 0: scale = 1.0, Layer 5: scale ≈ 0.41, Layer 10: scale ≈ 0.30.

 Why this helps: Deep networks have a tendency for the norms of residual streams to grow as signals accumulate across layers. The damping
 prevents this, stabilizing training and improving convergence in models where the residual stream might otherwise become dominated by a few
 early layers.

 ---
 9. MuonEq-R Optimizer

 Standard Muon:
 1. Compute gradient G (shape [m, n])
 2. Update momentum: M ← μM + G
 3. Orthogonalize: M̃ = Newton-Schulz(M) (approximate singular value whitening)
 4. Update weights: W ← W − η × M̃

 MuonEq-R adds a per-row equilibration step before orthogonalization:

 # Row-equilibration:
 row_norms = (M × M^T).diag().sqrt()   # [m], row L2 norms
 M_eq = M / row_norms.unsqueeze(1)     # rescale each row to unit L2 norm

 # Then orthogonalize as usual:
 M̃ = Newton-Schulz(M_eq)
 W ← W − η × M̃

 Mathematical justification:
 Standard Muon whitens the singular spectrum of M via Newton-Schulz (which approximates M → U where UU^T ≈ I). But if some rows of M have much
 larger norm than others, the orthogonalization is "uneven" — large-norm rows dominate the Gram matrix M M^T used in Newton-Schulz.

 Row-normalization is a zeroth-order whitening surrogate: it makes the row-norms uniform before the Newton-Schulz iteration, so the
 orthogonalization operates on a better-conditioned matrix. This improves convergence, especially for weight matrices where gradient norms vary
  strongly across output dimensions (e.g., the last MLP projection layer).

 Empirically: MuonEq-R with 130M/350M LLaMA-2 on C4 outperforms standard Muon consistently.

 ---
 10. Legal Score-First TTT (Test-Time Training)

 Setting: After all 600s of training, during evaluation of the validation set.

 Legal TTT protocol (Backward-looking, score-first):
 for each chunk in validation_set:
     # SCORE first (with inference_mode — no gradient tracking, no weight change)
     with torch.inference_mode():
         score[chunk] = sliding_window_eval(model, chunk)  # this is what counts

     # TRAIN after (update model on already-scored chunk)
     optimizer = SGD(model.params(), lr=0.002, momentum=0.9)
     for epoch in range(3):
         loss = model(chunk).mean()
         loss.backward()
         optimizer.step()

 Why it's legal: Each chunk is scored before the model sees it as training data. This is equivalent to a sequential online learning protocol —
 you predict before you learn, so there's no leakage.

 Why it works: The validation set is from FineWeb, which has consistent stylistic patterns. Adapting the model to the "style" of already-seen
 chunks helps predict future similar chunks better. SGD with momentum is used (not Adam) to avoid overfitting in 3 epochs.

 Gain: ~−0.0025 BPB, costs ~410s of wall time (within the evaluation budget, which has its own 10-minute cap separate from training).

 Why it failed on the current SOTA stack: The #1 submission (1.1147) found TTT neutral or negative on the Full GPTQ stack. This suggests TTT
 and GPTQ interact poorly — GPTQ quantizes the model, and then TTT fine-tunes the quantized model with SGD, which can undo the carefully
 calibrated quantization.

 ---
 11. EMA + Tight SWA Weight Averaging

 Exponential Moving Average (EMA):
 θ_ema = 0.997 × θ_ema + 0.003 × θ
 Updated every training step. At evaluation, use θ_ema instead of θ.

 Standard SWA (Stochastic Weight Averaging): Average checkpoints at periodic intervals during warmdown.

 Tight SWA (every 50 steps): Take 30+ checkpoints during warmdown, average them uniformly:
 θ_swa = (1/N) × Σᵢ θᵢ

 Why both? EMA provides continuous smoothing; SWA captures the "bottom of the loss bowl" during warmdown. Together, they produce weights that:
 1. Are "flatter" (less sensitive to small parameter perturbations)
 2. Quantize better (smoother weight distributions have less quantization error)
 3. Generalize better (averaging reduces overfitting to specific training trajectories)

 Mathematical intuition (Izmailov et al., 2018): SGD with cyclic/decaying LR traverses the loss surface, exploring a neighborhood of the
 minimum. The average of multiple points on this trajectory is closer to the geometric center of the loss basin than any single checkpoint,
 giving better generalization.

 ---
 12. U-Net Skip Connections

 Standard transformer: information flows residually forward through L layers.

 U-Net topology:
 Encoder layers 0..L/2-1: push residual h to skip_stack
 Decoder layers L/2..L-1: pop from skip_stack, mix with learned weight:
     h = h + skip_weight × h_skip

 skip_weight is a learnable [model_dim] vector (not a scalar), allowing per-dimension mixing.

 Why it helps: Information from early layers (token identity, local syntax) can bypass the middle layers and inject directly into late layers.
 This is especially useful when late layers have "forgotten" early context due to repeated mixing. The learned skip_weight allows the model to
 choose how much early-layer information to retain.

 Parameter cost: Only L/2 learnable vectors of dimension d, total = L/2 × d = 5 × 512 = 2,560 parameters (negligible).

 ---
 13. Parallel Muon with Parameter Banking

 Problem: Running Newton-Schulz (5 iterations of matrix multiplications) on 66 separate [m, n] weight matrices is slow because each is launched
  as a separate CUDA kernel with small matrix sizes.

 Parameter Banking: Group all 66 weight matrices into 4 contiguous 3D "banks":
 bank_attn = stack([W_q, W_k, W_v for all layers])  # [num_layers, in, out]
 bank_mlp  = stack([W_fc, W_proj for all layers])
 # etc.

 Parallel Newton-Schulz via torch.bmm:
 # 5 NS iterations, all matrices in parallel:
 for _ in range(5):
     M = torch.bmm(bank, bank.transpose(-2, -1))  # [B, in, in] for all in batch
     bank = bank + bank * M                        # update all matrices in one kernel

 Speedup: From ~85 ms/step to ~83.3 ms/step — about 2% faster. Over 7,000 steps in 600s, this is ~84 extra training steps, equivalent to ~1.4%
 more training compute.

 DDP interaction: Banks replace per-matrix DDP gradient sync with rank-local bank update, then async all-reduce. This reduces communication
 overhead.

 ---
 Priority Ranking: What to Implement

 Based on verified BPB gains and implementation feasibility, here is a stack-ranked list:

 Tier 1: Critical (Proven, High Impact)

 ┌───────────────────────────────────────┬──────────┬───────────────────────────────┬────────────────────┐
 │               Technique               │ BPB gain │             Cost              │ Status in baseline │
 ├───────────────────────────────────────┼──────────┼───────────────────────────────┼────────────────────┤
 │ SP8192 vocabulary                     │ ~−0.026  │ Medium (new dataset download) │ Not present        │
 ├───────────────────────────────────────┼──────────┼───────────────────────────────┼────────────────────┤
 │ Depth recurrence (2-3× on layers 4-5) │ ~−0.015  │ Low (few lines)               │ Not present        │
 ├───────────────────────────────────────┼──────────┼───────────────────────────────┼────────────────────┤
 │ Parallel residuals (attn ‖ MLP)       │ ~−0.008  │ Low                           │ Not present        │
 ├───────────────────────────────────────┼──────────┼───────────────────────────────┼────────────────────┤
 │ MuonEq-R (row equilibration)          │ ~−0.003  │ Low                           │ Uses standard Muon │
 ├───────────────────────────────────────┼──────────┼───────────────────────────────┼────────────────────┤
 │ QK-Gain tuning (4.0→5.25)             │ ~−0.002  │ Trivial (1 param)             │ Default 1.5        │
 └───────────────────────────────────────┴──────────┴───────────────────────────────┴────────────────────┘

 These five changes alone account for the gap from 1.1147 → 1.0810 (−0.034 BPB). They should be implemented first.

 Tier 2: High-Value (Proven, Medium Impact)

 ┌────────────────────────────────────┬──────────┬─────────────────────────────┐
 │             Technique              │ BPB gain │            Cost             │
 ├────────────────────────────────────┼──────────┼─────────────────────────────┤
 │ Full Hessian GPTQ + AR calibration │ ~−0.005  │ Medium (post-training pass) │
 ├────────────────────────────────────┼──────────┼─────────────────────────────┤
 │ LeakyReLU(0.5)²                    │ ~−0.003  │ Trivial (1-line change)     │
 ├────────────────────────────────────┼──────────┼─────────────────────────────┤
 │ XSA on all layers                  │ ~−0.003  │ Low                         │
 ├────────────────────────────────────┼──────────┼─────────────────────────────┤
 │ Legal TTT                          │ ~−0.002  │ Medium (eval-time)          │
 ├────────────────────────────────────┼──────────┼─────────────────────────────┤
 │ BigramHash 3072×112                │ ~−0.001  │ Low                         │
 ├────────────────────────────────────┼──────────┼─────────────────────────────┤
 │ GPTQ on embeddings                 │ ~−0.002  │ Medium                      │
 └────────────────────────────────────┴──────────┴─────────────────────────────┘

 Tier 3: Experimental (Research-Grade)

 ┌───────────────────────────────────┬────────────────────────┬─────────────────┬────────────────────────┐
 │             Technique             │     Expected gain      │      Risk       │         Source         │
 ├───────────────────────────────────┼────────────────────────┼─────────────────┼────────────────────────┤
 │ Differential attention            │ Unknown (−0.005 est.)  │ Medium          │ Microsoft ICLR 2025    │
 ├───────────────────────────────────┼────────────────────────┼─────────────────┼────────────────────────┤
 │ AQLM 1-2 bit quantization         │ Enables larger model   │ High complexity │ Frantar et al.         │
 ├───────────────────────────────────┼────────────────────────┼─────────────────┼────────────────────────┤
 │ Hadamard rotation (QuIP#)         │ −0.002 est.            │ High            │ Tseng et al. ICLR 2025 │
 ├───────────────────────────────────┼────────────────────────┼─────────────────┼────────────────────────┤
 │ MLA (Multi-head Latent Attention) │ Enables longer context │ Medium          │ DeepSeek 2024          │
 ├───────────────────────────────────┼────────────────────────┼─────────────────┼────────────────────────┤
 │ YaRN extended context             │ −0.003 est. for 2048+  │ Low             │ Peng et al. ICLR 2024  │
 ├───────────────────────────────────┼────────────────────────┼─────────────────┼────────────────────────┤
 │ KAN layers                        │ Unknown                │ Very high       │ Liu et al. ICLR 2025   │
 └───────────────────────────────────┴────────────────────────┴─────────────────┴────────────────────────┘

 ---
 Frontier AI Research Improvements (with Math)

 A. Hadamard Rotation Before Quantization (QuIP# / SpinQuant)

 Problem: Weight matrices in small models have activations with outlier dimensions — a small number of feature dimensions carry
 disproportionately large values. These outliers dominate the quantization range, causing other dimensions to be quantized with very coarse
 granularity.

 Solution (QuIP#, Tseng et al., ICLR 2025): Apply a random Hadamard rotation to the weight matrix before quantization:

 W_rot = H @ W @ H^T

 where H is a normalized Hadamard matrix (H_n × H_n = nI, so H/√n is orthogonal). A key property: multiplying by a Hadamard matrix spreads
 outlier energy uniformly across all dimensions.

 Math: For a vector w with one large component w_k, after rotation:
 (Hw)_i = Σⱼ H_{ij} w_j ≈ (1/√n) Σⱼ ±w_j
 Each rotated component sees a ±sum of all original components, distributed roughly as N(0, ‖w‖²/n). The per-component variance is ‖w‖²/n
 regardless of which original components are large.

 Why this matters for quantization: If original w has max value 100 and typical value 1, the quantization range must cover 100, so the
 resolution is 100/63 ≈ 1.6 per int6 step. After rotation, max value ≈ 3 (since values spread to ~‖w‖/√n), resolution is 3/63 ≈ 0.05. All
 quantization steps are spent on actual signal.

 Implementation:
 from scipy.linalg import hadamard
 import torch

 def apply_hadamard_rotation(W, group=64):
     # W: [out, in], group: dimension to rotate within
     H = torch.tensor(hadamard(group), dtype=torch.float32) / math.sqrt(group)
     # Apply to input dimension groups
     W_reshaped = W.reshape(W.shape[0], -1, group)
     W_rot = torch.einsum('...g,gh->...h', W_reshaped, H)
     return W_rot.reshape(W.shape)

 Expected gain: In large models, QuIP# gets 3-bit quality better than standard 4-bit. For small 16 MB models, this could allow 4-bit or even
 3-bit quantization with same reconstruction quality as current 6-bit.

 SpinQuant variant: Instead of fixed Hadamard, learn R on the Stiefel manifold:
 min_{R ∈ St(d,d)} L(Q(R @ W))
 where Q is the quantization operator. Optimization via Riemannian gradient descent on the manifold of orthogonal matrices.

 ---
 B. Vector Quantization: AQLM for Extreme Compression

 Standard scalar quantization: Each weight gets its own n-bit integer code.

 AQLM (Additive Quantization of Language Models, Frantar et al., ICML 2024): Groups of weights are jointly coded as a sum of vectors:

 w_group ≈ Σ_{k=1}^{K} codebook_k[code_k]

 where each codebook_k is a learned table of M vectors of dimension g (group size), and code_k is an M-bit index. With K=2, M=256, g=8:
 - Original: g × 16 bits = 128 bits for 8 weights
 - AQLM: K × log₂(M) = 2 × 8 = 16 bits for 8 weights → 8× compression, effectively 2 bits/weight

 Training AQLM codebooks:
 The codebooks are learned to minimize the layer output error (similar to GPTQ):
 min_{C₁...Cₖ, codes} ‖(W - Σₖ Cₖ[codeₖ])X‖²_F

 This is solved with alternating optimization: fix codes, update codebooks via least squares; fix codebooks, update codes via beam search.

 Why this could matter here: At 2 bits/weight vs 6 bits/weight (current GPTQ), a model could have 3× as many parameters within the same 16 MB
 budget. A 3× larger model at equivalent quantization quality would dramatically improve BPB.

 Practical challenge: AQLM inference requires hardware-efficient dequantization, which CUDA kernels must implement. The reference
 implementation uses lookup tables. For 16 MB competition, the codebooks themselves consume bytes (K × M × g × 2 bytes ≈ 2 × 256 × 8 × 2 = 8 KB
  per codebook pair).

 ---
 C. Differential Transformer (DiffT)

 Standard attention (noise-prone):
 A = softmax(QK^T / √d) V

 Each attention head must output meaningful signal AND suppress irrelevant tokens. In practice, attention distributions have diffuse noise
 across all positions.

 Differential attention (Microsoft, ICLR 2025): Split Q and K projections into two halves; compute two attention maps and subtract them:

 [Q₁, Q₂] = split(W_Q × X)     # [B, T, H, D/2] each
 [K₁, K₂] = split(W_K × X)
 A₁ = softmax(Q₁K₁^T / √(D/2))
 A₂ = softmax(Q₂K₂^T / √(D/2))
 A_diff = λ × A₁ - (1-λ) × A₂  # λ is learnable, init ≈ 0.8
 output = A_diff × V

 Why subtraction cancels noise:
 Both A₁ and A₂ share common noise components (uniform attention baseline ≈ 1/T for all positions). Subtracting yields:
 A_diff[i, j] ≈ (signal_ij + noise) - noise ≈ signal_ij

 This induces sparse attention — tokens that aren't relevant get near-zero attention from both heads, and the subtraction makes the difference
 even smaller.

 Gains (from paper):
 - 7.5% average accuracy improvement on downstream tasks
 - Better long-context retrieval (passkey tasks)
 - Reduced activation outliers (helps quantization)
 - Fewer heads needed for equivalent performance

 Parameter cost: Essentially the same as standard attention (Q, K, V projections of the same total size). The only overhead is the scalar λ per
  layer (negligible).

 ---
 D. Multi-Head Latent Attention (MLA) for KV Compression

 Standard MHA KV cache: Stores K and V separately per layer per position:
 cache_size ∝ 2 × n_heads × d_head × seq_len × n_layers

 For 11 layers, 8 heads, 64 d_head, 1024 seq_len: 11 × 8 × 64 × 1024 × 2 = 91 MB (impractical for a 16 MB model budget).

 MLA (DeepSeek-V2, 2024): Instead of storing K and V, store a compressed latent C:
 C = W_down × h      # [batch, seq, c_kv], c_kv << n_heads × d_head
 K = W_k_up × C      # reconstruct K
 V = W_v_up × C      # reconstruct V

 Parameters: W_down is [model_dim, c_kv], W_k_up and W_v_up are [c_kv, n_heads × d_head].

 Cache compression: Only C (size c_kv) is stored per position, instead of K+V (size 2 × n_kv_heads × d_head). With c_kv = 128 vs standard
 2×4×64=512, that's 4× KV cache reduction.

 Additional benefit: The latent C contains shared information for both K and V, effectively enforcing that K and V derive from the same
 compressed representation. This acts as a regularizer, preventing K and V from overfitting to training artifacts.

 For 16 MB competition: MLA's main value is allowing larger seq_len or more inference-time context without increasing the model's stored
 parameter count (W_down, W_k_up, W_v_up are stored in the model, but they're typically smaller than separate K/V projections).

 ---
 E. YaRN / NTK-Aware Extended RoPE for Longer Context

 Current baseline: Training seq_len=1024, eval with sliding window stride=64.

 YaRN (Yet Another RoPE extensioN, Peng et al., ICLR 2024):
 Instead of interpolating all RoPE frequencies equally, YaRN applies a per-frequency scale factor:

 For frequency f (= θᵢ = base^(-2i/d)):
   - If f > high_freq_threshold: no scaling (keep as is)
   - If f < low_freq_threshold: scale by 1/s (pure interpolation)
   - Otherwise: NTK-aware ramp (smooth interpolation between the two)

 base' = base × (s^(d/(d-2)))^(1/T)

 where s is the extension factor (e.g., 2 for 2× context extension) and T is the temperature.

 Attention temperature scaling (YaRN's key innovation):
 A = softmax(QK^T / (√d × α(s)))
 where α(s) = 0.1 × ln(s) + 1. This compensates for attention entropy changes when extending context.

 Why this helps beyond context extension: Even at training seq_len, YaRN's per-frequency scaling with NTK-aware interpolation lets the model
 better utilize different positional frequency bands. High-frequency components (local position) remain sharp; low-frequency components (global
  position) are more robustly encoded.

 Expected gain for BPB: If training at seq_len=2048 with YaRN (so the model sees 2× more context per training sample), this can improve BPB by
 ~0.003-0.005 at the cost of 2× memory per step.

 ---
 F. Improved Compression: Arithmetic Coding + Better Entropy Modeling

 Current: LZMA preset=9 (best LZ-based compression) or zstd level 22.

 Arithmetic coding achieves the theoretical entropy limit for a given source distribution:
 code_length ≈ -log₂(P(message)) bits

 where P(message) is the probability under the source model. LZMA uses an approximate entropy model; true arithmetic coding with an optimal
 predictor can approach Shannon entropy.

 For quantized int6 weights: The weight distribution after GPTQ is not uniform — most weights cluster near zero (due to magnitude penalization
 during training). A Laplacian or logistic distribution fits better:
 P(w) = (1/2b) × exp(-|w-μ|/b)   # Laplace

 Coding weights under their empirical distribution (instead of uniform) can save 10-30% of bits for weight matrices where most values are
 small.

 Practical approaches:
 1. Custom Huffman codes per layer: Compute empirical frequencies of each int6 value (0-63) per weight matrix row, build optimal prefix codes.
 Combined with zstd, this can squeeze an extra 2-5% vs zstd alone.
 2. ANS (Asymmetric Numeral Systems): Used in zstd internally; accessible via the zstandard Python package with custom context mixing.
 3. Per-layer entropy coding with weight grouping: Cluster weights by magnitude, code each cluster with different precision.

 Structural delta coding: If using depth recurrence, the recurrent block runs 3 times with the same weights. Within a recurrent block, the
 "skip" from loop iteration t to t+1 is often small. Coding the delta between iterations (always zero for shared weights) costs nothing beyond
 acknowledging they're the same.

 ---
 G. SwiGLU vs LeakyReLU² for Small Models

 SwiGLU (Noam Shazeer, 2020):
 SwiGLU(x, W, V, W₂) = W₂ × (Swish(xW) ⊗ xV)
 Swish(x) = x × sigmoid(βx)

 This requires two input projection matrices (W and V) where standard ReLU needs one. To keep parameter count equal, SwiGLU uses 2/3 of the
 hidden dimension:
 SwiGLU: 2 × (d → 2d/3) + 1 × (2d/3 → d) = 2d²×(2/3) + d²×(2/3) ≈ 2d²
 ReLU²:  1 × (d → d) + 1 × (d → d) = 2d² (same count if hidden=d)

 With 3× expansion: ReLU² uses hidden=1536; SwiGLU with equivalent params uses hidden=1024 but with gate+up projections.

 Empirical comparison for small constrained models:
 - Large models (7B+): SwiGLU reliably outperforms ReLU² by ~1-3% perplexity
 - Small models (<100M params): Less clear. ReLU² has sparser activations (better memory access patterns, more saturating gradients that help
 optimization); SwiGLU has smoother gradients
 - LeakyReLU² is currently winning in this competition; SwiGLU untested

 Recommendation: LeakyReLU² is already the Tier-1 choice. SwiGLU is an experimental alternative that may help or hurt — needs ablation.

Step 1: Match Current SOTA (1.0810 BPB)

 Implement these four techniques that account for the entire gap from 1.1147 to 1.0810:

 1. SP8192 vocabulary: Download fineweb10B_sp8192 dataset. Set VOCAB_SIZE=8192. This requires GPTQ embedding quantization (the embedding table
 at 8192×512×fp16 = 8 MB would otherwise overflow the budget).
 2. GPTQ Embeddings: Extend the GPTQ quantization pass to cover the embedding table in addition to weight matrices. Use frequency-weighted
 Hessians (common tokens get higher weight in H = Σᵢfᵢxᵢxᵢᵀ where fᵢ is token frequency).
 3. Depth recurrence (layers 4-5, 2×): Add a recurrence_depth parameter (default 2). In the forward pass, run blocks 4 and 5 twice with shared
 weights. Keep the remaining 9 layers non-recurrent.
 4. Parallel residuals: Change the Block forward from sequential:
 h = h + attn(norm(h))
 h = h + mlp(norm(h))
 4. to parallel:
 h_in = norm(h)
 h = h + attn(h_in) + mlp(h_in)
 5. MuonEq-R: Add row-equilibration before Newton-Schulz in the Muon optimizer.
 6. QK-Gain 5.0: Change qk_gain_init from 1.5 to 5.0.

 Step 2: Push Beyond 1.080 BPB

 Experimental frontier techniques with highest potential:

 1. Differential attention — replace standard attention with DiffT in all layers. Zero parameters, potential −0.005 BPB.
 2. Hadamard rotation before GPTQ — implement QuIP#-style rotation, should improve int6 reconstruction quality and allow moving to int4 for MLP
  weights.
 3. Larger context training (YaRN 2048) — train at 2048 seq_len with YaRN to enable better sliding window BPB.
 4. Progressive recurrence — start with 1 recurrence loop at step 0, linearly increase to 3 by step 2000.