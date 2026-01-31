# Conditional Transformer-UNet: Architecture Summary

This document summarizes the current conditional architecture, the reasoning behind each choice, and practical next steps.

## High-level idea

We train a **conditional flow-matching model** where:

- a **Transformer encoder** embeds the DtN map (represented as tokenized columns) into a **conditioning representation**
- a **FiLM-conditioned UNet** predicts `x_pred` from the current state `z_t`, time `t`, and that conditioning representation

In short:

- **TransformerEncoder** = condition encoder
- **ConditionalUnet** = time-dependent backbone
- **FiLM** = conditioning injection mechanism

---

## Data / inputs

- **Condition**: `x_cond` (DtN map), shape `(B, n_bdy, n_bdy)`
- **Tokenization**: treat columns as tokens:
  - `v = x_cond.transpose(1, 2)` giving `(B, N_tokens=n_bdy, D=n_bdy)`
- **State**: `z` (current image/state), shape `(B, 1, H, W)`
- **Time**: `t` shape `(B,)`

---

## Model flow (end-to-end)

### 1) ConditionalTransformerUnet.forward
File: `models/transformer/conditional.py`

Inputs:
- `z`: current state image
- `v`: tokenized DtN condition
- `t`: time

Steps:
1. `cond_rep = TransformerEncoder.encode(v, t)`
2. `cond_emb = LayerNorm(Linear(cond_rep))` to get a stable conditioning vector
3. `x_pred = ConditionalUnet(z, t, cond_emb)`

Output:
- `x_pred` (predicted clean image / state used by the flow-matching objective)

---

## TransformerEncoder
File: `models/transformer/default.py`

### Purpose
Learn a **good conditioning representation** of the DtN tokens, optionally time-aware.

### Encoder details
1. **Token projection**:
   - `x = Linear(in_dim -> d_model)(v)` => `(B, N, d_model)`
2. **Time embedding**:
   - `temb = sinusoidal_embedding(t, d_model)`
   - `tfeat = temb_proj(temb)`
   - add to all tokens: `x += tfeat[:, None, :]`
3. **Token positional embedding**:
   - `pos = sinusoidal_embedding(arange(N), d_model)`
   - `x += pos`
4. **Transformer blocks**:
   - standard self-attention + MLP blocks
5. **Attention pooling**:
   - compute token weights with a small learned scoring head:
     - `attn_logits = pool(x)` where `pool = LayerNorm(d_model) -> Linear(d_model,1)`
     - `attn = softmax(attn_logits, dim=1)`
   - produce a single vector:
     - `cond_rep = sum_i attn_i * x_i` => `(B, d_model)`

### Why these decisions
- **Encoder-only**: focuses on representation learning and avoids auxiliary image-decoding distractions.
- **Sinusoidal time embedding**: stable, standard, consistent with diffusion/flow UNets.
- **Sinusoidal token positions**: essential for “columns as tokens”; otherwise tokens are permutation-symmetric.
- **Attention pooling**: learns which DtN columns matter per sample; avoids dubious mean pooling.

---

## Shared sinusoidal embedding
File: `models/embeddings.py`

- `sinusoidal_embedding(timesteps, embedding_dim, ...)`
- UNet timestep embedding function in `models/unet/modules.py` calls this utility.

This keeps time embedding logic consistent across the project.

---

## ConditionalUnet (FiLM conditioned)
File: `models/unet/conditional.py`

### Purpose
Predict `x_pred` from `z` using:
- time embedding (always present)
- conditioning embedding (from the transformer)

### Time + conditioning embedding creation
- `time_emb = time_proj(get_timestep_embedding(t, ch))`  => `(B, temb_dim)`
- `cond_raw = cond_proj(cond)` => `(B, temb_dim)`
- apply a bounded, per-channel gate:
  - `gate = sigmoid(cond_gate)` where `cond_gate` is a learnable vector of size `(temb_dim,)`
  - `cond_emb = gate * cond_raw`

We initialize `cond_gate` to a negative value so `sigmoid(cond_gate)` is near 0 initially (conditioning nearly off), improving stability.

---

## Improved FiLM block: ConditionalResBlock
File: `models/unet/modules.py`

### Core idea
Avoid entangling time and conditioning too early.

Instead of forming one mixed embedding `emb = time + cond`, we keep two pathways:

- `t = time_cond(time_emb)`
- `c = ctx_cond(cond_emb)`
- `emb = t + c`

Then split into FiLM params:

- `(scale, shift) = emb.chunk(2)`
- apply: `GN(h) * (1 + scale) + shift`

### Zero-init for conditioning FiLM
We **zero-initialize** the linear layer inside `ctx_cond`.

Effect:
- initially, `ctx_cond(cond_emb) ≈ 0`
- conditioning starts as a true no-op
- training begins like a stable time-conditioned UNet
- the model learns to use conditioning gradually

---

## Summary of stabilization decisions

- **LayerNorm on conditioning embedding** before it hits FiLM.
- **Per-channel bounded gating** (`sigmoid(cond_gate)`) for conditioning strength.
- **Separate time and condition FiLM paths** in `ConditionalResBlock`.
- **Zero-init conditioning FiLM projection** so conditioning starts off.

These are standard diffusion/FiLM stabilization patterns.

---

## What to do next (recommended)

### 1) Verify conditioning is actually being used
Add light logging during training:
- mean of `sigmoid(cond_gate)` over training (should increase from ~0.02)
- norms of time FiLM vs condition FiLM contributions
- histogram / mean of attention pooling weights (do a few tokens dominate?)

### 2) Add conditioning dropout (CFG-style)
During training, with probability `p` set conditioning to zero:
- `cond_emb = 0`

Benefits:
- robustness
- enables classifier-free guidance later if desired

### 3) Improve transformer representation further
- try attention pooling depth (2-layer scoring head)
- token dropout/masking
- consider token-wise outputs to avoid a single-vector bottleneck

### 4) Stronger conditioning mechanism: cross-attention in the UNet
Instead of collapsing to `(B, d_model)`, keep transformer token outputs `(B, N, d_model)` and use cross-attention blocks inside the UNet.

This is typically the highest-fidelity conditioning approach, at the cost of more compute and code.

### 5) Pretrain the encoder (once stable)
Now that the transformer is cleanly an encoder, pretraining is clearer:
- supervised pretrain (DtN -> media) using an auxiliary head
- then fine-tune end-to-end with the flow objective

---

## Files involved

- `models/transformer/default.py` — `TransformerEncoder`
- `models/transformer/conditional.py` — `ConditionalTransformerUnet`
- `models/unet/conditional.py` — `ConditionalUnet`
- `models/unet/modules.py` — `ConditionalResBlock`, FiLM utilities, timestep embedding wrapper
- `models/embeddings.py` — shared sinusoidal embedding
- `train/train_transformer.py` — training script (flow-matching objective)
