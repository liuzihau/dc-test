import math
import typing

import einops
from functools import partial
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from functools import lru_cache

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
import torch._inductor.config as inductor_cfg
inductor_cfg.triton.cudagraphs = True
inductor_cfg.coordinate_descent_tuning = True

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

import os
FORCE_NAIVE_ATTENTION = bool(int(os.getenv("ESOLM_FORCE_NAIVE_ATTENTION", "0")))

BLOCK_SIZE = 128

@lru_cache
def _causal_mask(b, h, q_idx, kv_idx):
  causal = q_idx >= kv_idx
  return causal


@lru_cache
def _get_causal_mask(seq_len):
  return create_block_mask(
    _causal_mask,
    B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, BLOCK_SIZE=BLOCK_SIZE)


@lru_cache
def _bidirectional_mask(b, h, q_idx, kv_idx):
  bidirectional = q_idx == q_idx
  return bidirectional


@lru_cache
def _get_bidirectional_mask(seq_len):
  return create_block_mask(
    _bidirectional_mask,
    B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, BLOCK_SIZE=BLOCK_SIZE)

def _causal_context_mask(b, h, q_idx, kv_idx, cutoffs, mask_cutoffs):
  causal = q_idx >= kv_idx
  block_identity = q_idx >= cutoffs[b]
  base_mask = causal | block_identity
  # Forbid attending to positions >= mask_cutoffs[b]
  return base_mask & (kv_idx < mask_cutoffs[b])

def _get_causal_context_mask(seq_len, cutoffs, mask_cutoffs):
  # Note: B must be set to batch_size for per-batch masks to work correctly
  batch_size = len(cutoffs)
  return create_block_mask(
    partial(_causal_context_mask, cutoffs=cutoffs, mask_cutoffs=mask_cutoffs),
    B=batch_size, H=None, Q_LEN=seq_len, KV_LEN=seq_len, BLOCK_SIZE=BLOCK_SIZE)


def _causal_output_mask(b, h, q_idx, kv_idx, cutoffs, mask_cutoffs):
  causal = q_idx >= kv_idx
  block_identity = (q_idx < cutoffs[b]) & (kv_idx < cutoffs[b])
  base_mask = causal | block_identity
  # Forbid attending to positions >= mask_cutoffs[b]
  return base_mask & (kv_idx < mask_cutoffs[b])


def _get_causal_output_mask(seq_len, cutoffs, mask_cutoffs):
  # Note: B must be set to batch_size for per-batch masks to work correctly
  batch_size = len(cutoffs)
  return create_block_mask(
    partial(_causal_output_mask, cutoffs=cutoffs, mask_cutoffs=mask_cutoffs),
    B=batch_size, H=None, Q_LEN=seq_len, KV_LEN=seq_len, BLOCK_SIZE=BLOCK_SIZE)


def _full_mask(b, h, q_idx, kv_idx, mask_cutoffs):
  # Forbid attending to positions >= mask_cutoffs[b]
  return kv_idx < mask_cutoffs[b]


def _get_full_mask(seq_len, mask_cutoffs):
  # Note: B must be set to batch_size for per-batch masks to work correctly
  batch_size = len(mask_cutoffs)
  return create_block_mask(
    partial(_full_mask, mask_cutoffs=mask_cutoffs),
    B=batch_size, H=None, Q_LEN=seq_len, KV_LEN=seq_len, BLOCK_SIZE=BLOCK_SIZE)


def _solo_causal_mask(b, h, q_idx, kv_idx, cutoffs):
    q_is_clean  = q_idx < cutoffs[b]
    kv_is_clean = kv_idx < cutoffs[b]

    causal   = q_idx >= kv_idx
    same_pos = q_idx == kv_idx

    return (
        (q_is_clean & kv_is_clean & causal) |               # clean q
        ((~q_is_clean) & (kv_is_clean | (~kv_is_clean & same_pos)))  # masked q
    )


def _get_solo_causal_mask(seq_len, cutoffs):
  # Note: B must be set to batch_size for per-batch masks to work correctly
  batch_size = len(cutoffs)
  return create_block_mask(
    partial(_solo_causal_mask, cutoffs=cutoffs),
    B=batch_size, H=None, Q_LEN=seq_len, KV_LEN=seq_len)


def _solo_full_mask(b, h, q_idx, kv_idx, cutoffs):
    """
    Independent full mask: masked tokens attend to clean tokens and themselves only.
    - Clean-to-clean: Full (bidirectional)
    - Clean-to-masked: Blocked
    - Masked-to-clean: Full
    - Masked-to-masked: Diagonal (self-attend only)
    """
    q_is_clean  = q_idx < cutoffs[b]
    kv_is_clean = kv_idx < cutoffs[b]
    same_pos = q_idx == kv_idx
    return (
        (q_is_clean & kv_is_clean) |                                    # clean q: full among clean
        ((~q_is_clean) & (kv_is_clean | (~kv_is_clean & same_pos)))     # masked q: clean + self
    )


def _get_solo_full_mask(seq_len, cutoffs):
    batch_size = len(cutoffs)
    return create_block_mask(
        partial(_solo_full_mask, cutoffs=cutoffs),
        B=batch_size, H=None, Q_LEN=seq_len, KV_LEN=seq_len)


@lru_cache
def _dense_causal_mask(S, device_str: str, dtype_str: str):
  device = torch.device(device_str)
  m = torch.ones(S, S, device=device, dtype=torch.bool).tril()
  return m.view(1, 1, S, S)  # [1,1,S,S]

@lru_cache
def _dense_full_mask(S, device_str: str, dtype_str: str, mask_cutoffs: torch.Tensor):
  """
  Full attention mask with mask_cutoffs applied.
  mask_cutoffs: [B], rightmost position any query can attend to
  """
  device = torch.device(device_str)
  B = mask_cutoffs.shape[0]
  m = torch.zeros(B, 1, S, S, device=device, dtype=torch.bool)
  for b in range(B):
    mc = int(mask_cutoffs[b].item())
    # All queries can attend to keys in [0, mc)
    m[b, 0, :, :mc] = True
  return m

@lru_cache
def _dense_causal_context_mask(S, cutoffs: torch.Tensor, mask_cutoffs: torch.Tensor):
  """
  causal_context: 'causal over clean' + 'bidirectional over masked'
  Here we return [B,1,S,S] boolean mask.
  cutoffs: [B], each cutoff[b] in [0..S]
  mask_cutoffs: [B], rightmost position any query can attend to
  """
  B = cutoffs.shape[0]
  device = cutoffs.device
  m = torch.zeros(B, 1, S, S, device=device, dtype=torch.bool)
  for b in range(B):
    c = int(cutoffs[b].item())
    mc = int(mask_cutoffs[b].item())
    # clean rows/cols: causal among themselves
    if c > 0:
      tril = torch.ones(c, c, device=device, dtype=torch.bool).tril()
      m[b, 0, :c, :c] = tril
    # masked rows/cols: fully bidirectional *within masked*
    if c < S:
      m[b, 0, c:, c:] = True
      m[b, 0, c:, :c] = True
    # Apply mask_cutoffs: forbid attending to positions >= mc
    m[b, 0, :, mc:] = False
  return m

@lru_cache
def _dense_causal_output_mask(S, cutoffs: torch.Tensor, mask_cutoffs: torch.Tensor):
  """
  causal_output: 'bidirectional over clean' + 'causal over masked'
  mask_cutoffs: [B], rightmost position any query can attend to
  """
  B = cutoffs.shape[0]
  device = cutoffs.device
  m = torch.zeros(B, 1, S, S, device=device, dtype=torch.bool)
  for b in range(B):
    c = int(cutoffs[b].item())
    mc = int(mask_cutoffs[b].item())
    if c > 0:
      m[b, 0, :c, :c] = True  # clean<->clean full
    if c < S:
      tril = torch.ones(S-c, S-c, device=device, dtype=torch.bool).tril()
      m[b, 0, c:, c:] = tril   # masked causal among masked
      m[b, 0, c:, :c] = True
    # Apply mask_cutoffs: forbid attending to positions >= mc
    m[b, 0, :, mc:] = False
  return m

@lru_cache
def _dense_solo_causal_mask(S, cutoffs: torch.Tensor):
  """
  Dense version of _solo_causal_mask.
  Returns [B,1,S,S] boolean mask.
  """
  B = cutoffs.shape[0]
  device = cutoffs.device
  m = torch.zeros(B, 1, S, S, device=device, dtype=torch.bool)
  for b in range(B):
    c = int(cutoffs[b].item())
    # --- clean queries ---
    if c > 0:
      # clean-clean block: causal (tril)
      tril = torch.ones(c, c, device=device, dtype=torch.bool).tril()
      m[b, 0, :c, :c] = tril
      # clean queries cannot attend to masked keys (already zero)
    # --- masked queries ---
    if c < S:
      diag_mask = torch.eye(S - c, dtype=torch.bool, device=device)
      m[b, 0, c:, c:] = diag_mask
      m[b, 0, c:, :c] = True
  return m


@lru_cache
def _dense_solo_full_mask(S, cutoffs: torch.Tensor):
    """
    Dense version of _solo_full_mask.
    Returns [B,1,S,S] boolean mask.
    """
    B = cutoffs.shape[0]
    device = cutoffs.device
    m = torch.zeros(B, 1, S, S, device=device, dtype=torch.bool)
    for b in range(B):
        c = int(cutoffs[b].item())
        # --- clean queries ---
        if c > 0:
            m[b, 0, :c, :c] = True  # clean-clean: full bidirectional
        # --- masked queries ---
        if c < S:
            diag_mask = torch.eye(S - c, dtype=torch.bool, device=device)
            m[b, 0, c:, c:] = diag_mask  # masked-masked: diagonal only
            m[b, 0, c:, :c] = True       # masked-clean: full
    return m


def _block_diff_mask(b, h, q_idx, kv_idx, block_size=1, n=None):
  """
  Copied directly from BD3LM's codebase: https://github.com/kuleshov-group/bd3lms

  Constructs the specialized block diffusion attention mask for training
  composed of three masks:
  - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
  - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
  - **Block Causal Mask (M_BC)**: Attention to update x0

  Args:
      b, h: Batch and head indices (ignored for mask logic).
      q_idx, kv_idx: Query and Key indices.
      seq_len: Total sequence length.
      block_size: Defines the block structure.

  Returns:
      A boolean attention mask.
  """

  # Indicate whether token belongs to xt or x0
  x0_flag_q = (q_idx >= n)
  x0_flag_kv = (kv_idx >= n)

  # Compute block indices
  block_q = torch.where(x0_flag_q == 1,
                        (q_idx - n) // block_size,
                        q_idx // block_size)
  block_kv = torch.where(x0_flag_kv == 1,
                         (kv_idx - n) // block_size,
                         kv_idx // block_size)

  # **1. Block Diagonal Mask (M_BD) **
  block_diagonal = (
    block_q == block_kv) & (x0_flag_q == x0_flag_kv)

  # **2. Offset Block-Causal Mask (M_OBC) **
  offset_block_causal = ((block_q > block_kv)
                          & (x0_flag_kv == 1)
                          & (x0_flag_q == 0))

  # **3. Block-Causal Mask (M_BC) **
  block_causal = (block_q >= block_kv) & (
    x0_flag_kv == 1) & (x0_flag_q == 1)

  # **4. Combine Masks **
  return block_diagonal | offset_block_causal | block_causal


# flex_attention_compiled = torch.compile(flex_attention, dynamic=False, fullgraph=True, mode='reduce-overhead')
flex_attention_compiled = torch.compile(flex_attention, dynamic=True, fullgraph=True, mode='reduce-overhead')

# flex_attention_compiled = torch.compile(flex_attention, dynamic=False, fullgraph=True, mode='max-autotune-no-cudagraphs')
# flex_attention_compiled = flex_attention
# flex_attention_compiled = torch.compile(flex_attention, dynamic=True)


def fused_flex_attention(q, k, v, mask=None):
  return flex_attention_compiled(q, k, v, block_mask=mask)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
      freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
      emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
      # dims are: batch, seq_len, qkv, head, dim
      self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
      self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
      # This makes the transformation on v an identity.
      self.cos_cached[:,:,2,:,:].fill_(1.)
      self.sin_cached[:,:,2,:,:].fill_(0.)

    return self.cos_cached, self.sin_cached


def rotate_half(x, interleaved=False):
  """Copied and refactored from FlashAttention"""
  if interleaved:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return einops.rearrange(
      torch.stack((-x2, x1), dim=-1),
      "... d two -> ... (d two)",
      two=2)
  x1, x2 = x.chunk(2, dim=-1)
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
  """
  Copied and refactored from FlashAttention
  x: (batch_size, seq_len, nheads, headdim)
  cos, sin: (seq_len, rotary_dim / 2) or (batch_size, seq_len, rotary_dim / 2)
  """
  ro_dim = cos.shape[-1] * 2
  assert ro_dim <= x.shape[-1]
  pattern = "... d -> ... 1 (2 d)"
  if interleaved:
    pattern =  "... d -> ... 1 (d 2)"
  cos = einops.repeat(cos, pattern)
  sin = einops.repeat(sin, pattern)
  return torch.cat(
      [x[..., :ro_dim] * cos
       + rotate_half(x[..., :ro_dim],
                     interleaved) * sin, x[..., ro_dim:]],
      dim=-1)


def _split_rotary(rotary_cos_sin, dtype):
  cos, sin = rotary_cos_sin
  cos = cos.to(dtype)
  sin = sin.to(dtype)
  cos = cos[0,:,0,0,:cos.shape[-1]//2]
  sin = sin[0,:,0,0,:sin.shape[-1]//2]
  return cos, sin


def split_qkv_no_rotary(qkv):
  """Split qkv tensor without applying rotary embeddings."""
  q, k, v = qkv.chunk(3, dim=2)
  return q.squeeze(dim=2), k.squeeze(dim=2), v.squeeze(dim=2)


def split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin):
  with torch.amp.autocast('cuda', enabled=False):
    cos, sin = _split_rotary(rotary_cos_sin, dtype=qkv.dtype)
    q, k, v = qkv.chunk(3, dim=2)
    q = apply_rotary_emb_torch(
      q.squeeze(dim=2), cos, sin)
    k = apply_rotary_emb_torch(
      k.squeeze(dim=2), cos, sin)
    v = v.squeeze(dim=2)
  return q, k, v


def split_and_apply_rotary_pos_emb_batch(qkv, rotary_cos_sin):
  with torch.amp.autocast('cuda', enabled=False):
    cos, sin = rotary_cos_sin
    cos = cos.to(qkv.dtype)
    sin = sin.to(qkv.dtype)
    cos = cos[:,:,0,0,:cos.shape[-1]//2]  # difference is here
    sin = sin[:,:,0,0,:sin.shape[-1]//2]  # difference is here
    q, k, v = qkv.chunk(3, dim=2)
    q = apply_rotary_emb_torch(
      q.squeeze(dim=2), cos, sin)
    k = apply_rotary_emb_torch(
      k.squeeze(dim=2), cos, sin)
    v = v.squeeze(dim=2)
  return q, k, v


def apply_rotary_emb_per_head(x, cos, sin, interleaved=False):
  """
  Apply rotary embeddings with per-head positions.
  x: (batch_size, seq_len, nheads, headdim)
  cos, sin: (batch_size, seq_len, nheads, rotary_dim / 2)
  """
  ro_dim = cos.shape[-1] * 2
  assert ro_dim <= x.shape[-1]
  pattern = "b s h d -> b s h (2 d)"
  if interleaved:
    pattern = "b s h d -> b s h (d 2)"
  cos = einops.repeat(cos, pattern)  # repeat, not rearrange
  sin = einops.repeat(sin, pattern)
  return torch.cat(
      [x[..., :ro_dim] * cos
       + rotate_half(x[..., :ro_dim], interleaved) * sin, 
       x[..., ro_dim:]],
      dim=-1)


def split_and_apply_rotary_pos_emb_split_heads(qkv, rotary_cos_sin):
  """
  Apply rotary embeddings where different heads have different positions.
  rotary_cos_sin: tuple of (cos, sin) with shape (bs, seq_len, 3, n_heads, rotary_dim)
  qkv: (bs, seq_len, 3, n_heads, head_dim)
  
  Note: cos/sin are the same for q and k (index 0 and 1 of dim 2), so we just use index 0.
  """
  with torch.amp.autocast('cuda', enabled=False):
    cos, sin = rotary_cos_sin
    cos = cos.to(qkv.dtype)
    sin = sin.to(qkv.dtype)
    # cos shape: (bs, seq_len, 3, n_heads, rotary_dim)
    # q and k use the same rotary, just extract from index 0
    cos = cos[:, :, 0, :, :cos.shape[-1]//2]  # (bs, seq_len, n_heads, rotary_dim//2)
    sin = sin[:, :, 0, :, :sin.shape[-1]//2]
    
    q, k, v = qkv.chunk(3, dim=2)
    # q shape after squeeze: (bs, seq_len, n_heads, head_dim)
    q = apply_rotary_emb_per_head(q.squeeze(dim=2), cos, sin)
    k = apply_rotary_emb_per_head(k.squeeze(dim=2), cos, sin)
    v = v.squeeze(dim=2)
  return q, k, v


def split_and_apply_rotary_pos_emb_qk_split(qkv, rotary_cos_sin):
  """
  Apply rotary embeddings where Q and K have different positions.
  - Q (queries): use target positions (asking "what goes at target position?")
  - K (keys): use source positions (saying "I'm a token from source position")
  
  rotary_cos_sin: tuple of (cos, sin) with shape (bs, seq_len, 2, 1, rotary_dim)
                  where dim 2 is [q_rotary (target), k_rotary (source)]
  qkv: (bs, seq_len, 3, n_heads, head_dim)
  """
  with torch.amp.autocast('cuda', enabled=False):
    cos, sin = rotary_cos_sin
    cos = cos.to(qkv.dtype)
    sin = sin.to(qkv.dtype)
    # cos shape: (bs, seq_len, 2, 1, rotary_dim)
    # Extract separate cos/sin for Q (target) and K (source)
    cos_q = cos[:, :, 0, 0, :cos.shape[-1]//2]  # (bs, seq_len, rotary_dim//2)
    sin_q = sin[:, :, 0, 0, :sin.shape[-1]//2]
    cos_k = cos[:, :, 1, 0, :cos.shape[-1]//2]  # (bs, seq_len, rotary_dim//2)
    sin_k = sin[:, :, 1, 0, :sin.shape[-1]//2]
    
    q, k, v = qkv.chunk(3, dim=2)
    # q shape after squeeze: (bs, seq_len, n_heads, head_dim)
    q = apply_rotary_emb_torch(q.squeeze(dim=2), cos_q, sin_q)
    k = apply_rotary_emb_torch(k.squeeze(dim=2), cos_k, sin_k)
    v = v.squeeze(dim=2)
  return q, k, v


def split_and_apply_rotary_pos_emb_avg(qkv, rotary_cos_sin):
  """
  Apply rotary embeddings by averaging source and target positions.
  q_final = (q_source + q_target) / 2
  k_final = (k_source + k_target) / 2
  
  rotary_cos_sin: tuple of (cos, sin) with shape (bs, seq_len, 2, 1, rotary_dim)
                  where dim 2 is [target, source]
  qkv: (bs, seq_len, 3, n_heads, head_dim)
  """
  with torch.amp.autocast('cuda', enabled=False):
    cos, sin = rotary_cos_sin
    cos = cos.to(qkv.dtype)
    sin = sin.to(qkv.dtype)
    # cos shape: (bs, seq_len, 2, 1, rotary_dim)
    # Extract cos/sin for target (index 0) and source (index 1)
    cos_target = cos[:, :, 0, 0, :cos.shape[-1]//2]  # (bs, seq_len, rotary_dim//2)
    sin_target = sin[:, :, 0, 0, :sin.shape[-1]//2]
    cos_source = cos[:, :, 1, 0, :cos.shape[-1]//2]
    sin_source = sin[:, :, 1, 0, :sin.shape[-1]//2]
    
    q, k, v = qkv.chunk(3, dim=2)
    q = q.squeeze(dim=2)  # (bs, seq_len, n_heads, head_dim)
    k = k.squeeze(dim=2)
    v = v.squeeze(dim=2)
    
    # Apply rotary for both source and target, then average
    q_target = apply_rotary_emb_torch(q, cos_target, sin_target)
    q_source = apply_rotary_emb_torch(q, cos_source, sin_source)
    q = (q_target + q_source) / 2
    
    k_target = apply_rotary_emb_torch(k, cos_target, sin_target)
    k_source = apply_rotary_emb_torch(k, cos_source, sin_source)
    k = (k_target + k_source) / 2
    
  return q, k, v


def split_and_apply_rotary_pos_emb_4way(qkv, rotary_cos_sin):
  """
  Apply rotary embeddings with 4-way head split for all source/target combinations.
  - 1/4 heads: q_source × k_source
  - 1/4 heads: q_source × k_target
  - 1/4 heads: q_target × k_source
  - 1/4 heads: q_target × k_target
  
  rotary_cos_sin: tuple of (cos, sin) with shape (bs, seq_len, 2, 1, rotary_dim)
                  where dim 2 is [target, source]
  qkv: (bs, seq_len, 3, n_heads, head_dim)
  """
  with torch.amp.autocast('cuda', enabled=False):
    cos, sin = rotary_cos_sin
    cos = cos.to(qkv.dtype)
    sin = sin.to(qkv.dtype)
    # cos shape: (bs, seq_len, 2, 1, rotary_dim)
    cos_target = cos[:, :, 0, 0, :cos.shape[-1]//2]  # (bs, seq_len, rotary_dim//2)
    sin_target = sin[:, :, 0, 0, :sin.shape[-1]//2]
    cos_source = cos[:, :, 1, 0, :cos.shape[-1]//2]
    sin_source = sin[:, :, 1, 0, :sin.shape[-1]//2]
    
    q, k, v = qkv.chunk(3, dim=2)
    q = q.squeeze(dim=2)  # (bs, seq_len, n_heads, head_dim)
    k = k.squeeze(dim=2)
    v = v.squeeze(dim=2)
    
    n_heads = q.shape[2]
    n_quarter = n_heads // 4
    
    # Compute all four rotary versions
    q_source = apply_rotary_emb_torch(q, cos_source, sin_source)
    q_target = apply_rotary_emb_torch(q, cos_target, sin_target)
    k_source = apply_rotary_emb_torch(k, cos_source, sin_source)
    k_target = apply_rotary_emb_torch(k, cos_target, sin_target)
    
    # Split by heads and combine
    # Group 1: q_source × k_source (heads 0 to n_quarter-1)
    # Group 2: q_source × k_target (heads n_quarter to 2*n_quarter-1)
    # Group 3: q_target × k_source (heads 2*n_quarter to 3*n_quarter-1)
    # Group 4: q_target × k_target (heads 3*n_quarter to n_heads-1)
    
    q_out = torch.cat([
      q_source[:, :, :n_quarter, :],
      q_source[:, :, n_quarter:2*n_quarter, :],
      q_target[:, :, 2*n_quarter:3*n_quarter, :],
      q_target[:, :, 3*n_quarter:, :]
    ], dim=2)
    
    k_out = torch.cat([
      k_source[:, :, :n_quarter, :],
      k_target[:, :, n_quarter:2*n_quarter, :],
      k_source[:, :, 2*n_quarter:3*n_quarter, :],
      k_target[:, :, 3*n_quarter:, :]
    ], dim=2)
    
  return q_out, k_out, v


def split_and_apply_rotary_pos_emb_2way(qkv, rotary_cos_sin):
  """
  Apply rotary embeddings with 2-way head split.
  Keys always use source positions; queries use source or target.
  - 1/2 heads: q_source × k_source
  - 1/2 heads: q_target × k_source
  
  rotary_cos_sin: tuple of (cos, sin) with shape (bs, seq_len, 2, 1, rotary_dim)
                  where dim 2 is [target, source]
  qkv: (bs, seq_len, 3, n_heads, head_dim)
  """
  with torch.amp.autocast('cuda', enabled=False):
    cos, sin = rotary_cos_sin
    cos = cos.to(qkv.dtype)
    sin = sin.to(qkv.dtype)
    # cos shape: (bs, seq_len, 2, 1, rotary_dim)
    cos_target = cos[:, :, 0, 0, :cos.shape[-1]//2]  # (bs, seq_len, rotary_dim//2)
    sin_target = sin[:, :, 0, 0, :sin.shape[-1]//2]
    cos_source = cos[:, :, 1, 0, :cos.shape[-1]//2]
    sin_source = sin[:, :, 1, 0, :sin.shape[-1]//2]
    
    q, k, v = qkv.chunk(3, dim=2)
    q = q.squeeze(dim=2)  # (bs, seq_len, n_heads, head_dim)
    k = k.squeeze(dim=2)
    v = v.squeeze(dim=2)
    
    n_heads = q.shape[2]
    n_half = n_heads // 2
    
    # Compute rotary versions
    q_source = apply_rotary_emb_torch(q, cos_source, sin_source)
    q_target = apply_rotary_emb_torch(q, cos_target, sin_target)
    # Keys always use source
    k_source = apply_rotary_emb_torch(k, cos_source, sin_source)
    
    # Split by heads and combine
    # Group 1: q_source × k_source (first half)
    # Group 2: q_target × k_source (second half)
    q_out = torch.cat([
      q_source[:, :, :n_half, :],
      q_target[:, :, n_half:, :]
    ], dim=2)
    
    # Keys all use source
    k_out = k_source
    
  return q_out, k_out, v


def flex_attention_multi_headed(q, k, v, mask):
  q = q.transpose(1, 2).contiguous()
  k = k.transpose(1, 2).contiguous()
  v = v.transpose(1, 2).contiguous()
  attention_output = fused_flex_attention(q, k, v, mask=mask)
  attention_output = attention_output.transpose(1, 2).contiguous()
  return einops.rearrange(attention_output, 'b s h d -> b s (h d)')


def _apply_dense_mask(scores: torch.Tensor, dense_mask: torch.Tensor):
  # scores: [B, H, S_q, S_k], dense_mask: [B or 1, 1, S_q, S_k] (bool)
  if dense_mask is None:
    return scores
  # Broadcast and mask with dtype-appropriate -inf
  neg_inf = torch.finfo(scores.dtype).min
  return scores.masked_fill(~dense_mask, neg_inf)

def naive_attention_multi_headed(q, k, v, dense_mask=None):
  # q,k,v: [B, S, H, D]  -> match the interface used in flex_attention_multi_headed
  q = q.transpose(1, 2).contiguous()  # [B, H, S, D]
  k = k.transpose(1, 2).contiguous()  # [B, H, S, D]
  v = v.transpose(1, 2).contiguous()  # [B, H, S, D]
  scale = q.shape[-1] ** 0.5
  scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, S, S]
  scores = _apply_dense_mask(scores, dense_mask)         # optional
  attn = torch.softmax(scores, dim=-1)
  out = torch.matmul(attn, v).transpose(1, 2).contiguous()  # [B, S, H, D]
  return einops.rearrange(out, 'b s h d -> b s (h d)')


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
  def __init__(self, dim, use_bias=False):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.bias = nn.Parameter(torch.zeros([dim])) if use_bias else None
    self.dim = dim

  def forward(self, x):
    with torch.amp.autocast('cuda', enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    x = x * self.weight[None, None, :]
    if self.bias is not None:
      x = x + self.bias[None, None, :]
    return x


def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
      / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class LabelEmbedder(nn.Module):
  """Embeds class labels into vector representations.
  
  Also handles label dropout for classifier-free guidance.
  """
  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
    self.num_classes = num_classes

    # TODO think of initializing with 0.02 std deviation like in original DiT paper

  def forward(self, labels):
    embeddings = self.embedding_table(labels)
    return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################

class DDiTBlockCausal(nn.Module):
  def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1, use_gpt2_arch=False):
    super().__init__()
    self.n_heads = n_heads

    # When use_gpt2_arch=True, add biases to LayerNorm and attention projections
    use_bias = use_gpt2_arch
    self.dim = dim
    self.norm1 = LayerNorm(dim, use_bias=use_bias)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=use_bias)
    self.attn_out = nn.Linear(dim, dim, bias=use_bias)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim, use_bias=use_bias)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    self.past_k = None
    self.past_v = None

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def reset_kv_cache(self):
    self.past_k = None
    self.past_v = None

  def _process_and_update_kv(self, k, v):
    if (self.past_k is not None
        and self.past_v is not None):
      k = torch.cat([self.past_k, k], dim=1)
      v = torch.cat([self.past_v, v], dim=1)
    self.past_k = k
    self.past_v = v
    return k, v

  @torch.no_grad()
  def _attention_with_kv_cache(self, qkv, rotary_cos_sin):
    assert qkv.shape[1] == 1
    q, k, v = qkv.chunk(3, dim=2)
    k, v = self._process_and_update_kv(k=k, v=v)
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = _split_rotary(rotary_cos_sin, q.dtype)
      q = apply_rotary_emb_torch(
        q.squeeze(dim=2), cos[-1:, :], sin[-1:, :])
      k = apply_rotary_emb_torch(k.squeeze(dim=2), cos, sin)
      v = v.squeeze(dim=2)
    scale = q.shape[-1] ** 0.5
    # swap seq_len and num_heads
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
    attn_weights = F.softmax(attn_scores, dim=-1)
    x =  torch.matmul(attn_weights, v).transpose(1, 2)
    return x.view(x.shape[0], 1, self.dim)

  def forward(self, x, rotary_cos_sin, kv_cache=False, **kwargs):
    del kwargs
    bias_dropout_scale_fn = self._get_bias_dropout_scale()
    x_skip = x
    x = self.norm1(x)
    qkv = einops.rearrange(
      self.attn_qkv(x),
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads)
    
    if kv_cache:
      x = self._attention_with_kv_cache(qkv.detach(), rotary_cos_sin)
    else:
      if rotary_cos_sin is not None:
        q, k, v = split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin)
      else:
        q, k, v = split_qkv_no_rotary(qkv)
      # recreate the mask every time (cheap) to fit different input length
      # different input length can happen during generation
      attn_mask = _get_causal_mask(x.shape[1])
      x = flex_attention_multi_headed(q, k, v, attn_mask)

    scale = torch.ones(1, device=x.device, dtype=x.dtype)
    x = bias_dropout_scale_fn(
      self.attn_out(x), None, scale, x_skip, self.dropout)

    # mlp operation
    x = bias_dropout_scale_fn(
      self.mlp(self.norm2(x)), None, scale, x, self.dropout)
    return x


class DDiTBlock(nn.Module):
  def __init__(self, dim, n_heads, adaLN,
               cond_dim=None, mlp_ratio=4,
               dropout=0.1, use_gpt2_arch=False):
    super().__init__()
    self.n_heads = n_heads
    self.dim = dim
    self.adaLN = adaLN

    # When use_gpt2_arch=True, add biases to LayerNorm and attention projections
    use_bias = use_gpt2_arch
    self.norm1 = LayerNorm(dim, use_bias=use_bias)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=use_bias)
    self.attn_out = nn.Linear(dim, dim, bias=use_bias)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim, use_bias=use_bias)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()

    self.past_k = None
    self.past_v = None
    self.neg_infinity = -1000000.0

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def reset_kv_cache(self):
    self.past_k = None
    self.past_v = None

  def _process_and_update_kv(self, k, v, num_clean):
    if num_clean == 0:
      # no caching if all we see if mask tokens
      return k, v
    else:
      if (self.past_k is None 
          and self.past_v is None):
        self.past_k = k[:, :num_clean, :, :]
        self.past_v = v[:, :num_clean, :, :]
        return k, v
      else:
        k_so_far = torch.cat([self.past_k, k], dim=1)
        v_so_far = torch.cat([self.past_v, v], dim=1)
        # only update the kv cache with kv values from
        # clean tokens generated during the previous 
        # iteration
        self.past_k = torch.cat(
          [self.past_k, k[:, :num_clean, :, :]], dim=1)
        self.past_v = torch.cat(
          [self.past_v, v[:, :num_clean, :, :]], dim=1)
        return k_so_far, v_so_far

  @torch.no_grad()
  def _attention_with_kv_cache(self, qkv, rotary_cos_sin, 
                               num_clean, num_clean_and_mask, attn_mode):
    # TODO this should also accept attn_mask
    # num_clean: num gen last
    # num_clean_and_mask: num gen last + num to gen
    assert qkv.shape[1] == num_clean_and_mask
    # qkv shape: 
    # [bs, num gen last + num to gen, 3, h, d]
    q, k, v = qkv.chunk(3, dim=2)
    q = q.squeeze(dim=2)
    k = k.squeeze(dim=2)
    v = v.squeeze(dim=2)
    k, v = self._process_and_update_kv(
      k=k, v=v, num_clean=num_clean)
    # new kv shape: 
    # [bs, 
    #  num gen before last + num gen last + num to gen, 
    #  h, d]
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = rotary_cos_sin
      cos = cos.to(qkv.dtype)
      sin = sin.to(qkv.dtype)
      cos = cos[:,:,0,0,:cos.shape[-1]//2]
      sin = sin[:,:,0,0,:sin.shape[-1]//2]
      cos_part = cos[:, -num_clean_and_mask:]
      sin_part = sin[:, -num_clean_and_mask:]
      q = apply_rotary_emb_torch(q, cos_part, sin_part)
      k = apply_rotary_emb_torch(k, cos, sin)
    scale = q.shape[-1] ** 0.5
    # shapes after transpose:
    # q: [bs, h, num gen last + num to gen, d]
    # k: [bs, h, num gen before last + num gen last + num to gen, d]
    # v: [bs, h, num gen before last + num gen last + num to gen, d]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    # attn_scores shape: 
    # [bs, h, 
    #  num gen last + num to gen, 
    #  num gen before last + num gen last + num to gen]
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale

    # TODO these need to support mask_cutoffs
    # turn off kv cache for now

    if attn_mode == 'causal':
      # A contains very large negative values above the diagonal
      # - q attends to all v values over "num gen before last"
      # - q attends causally to v values within "num gen last
      #   + num to gen"
      ones = torch.ones(
        num_clean_and_mask, num_clean_and_mask).to(qkv.device)
      A = self.neg_infinity * torch.triu(ones, diagonal=1)
      A = A.view(1, 1, num_clean_and_mask, num_clean_and_mask)
      attn_scores[:, :, :, -num_clean_and_mask:] += A

    elif attn_mode == 'causal_context':
      # Layout of the last num_clean_and_mask tokens:
      # [ clean (num_clean) | masked (num_masked) ]
      mask_start = -num_clean_and_mask + num_clean
      if num_clean > 0:
        clean_slice = slice(-num_clean_and_mask, mask_start)
        # causal restriction over clean tokens (lower-triangular)
        ones = torch.ones(num_clean, num_clean, device=qkv.device)
        A = self.neg_infinity * torch.triu(ones, diagonal=1)
        A = A.view(1, 1, num_clean, num_clean)
        # apply causal mask within clean block
        attn_scores[:, :, clean_slice, clean_slice] += A
        # clean should not attend to masked
        attn_scores[:, :, clean_slice, mask_start:] = self.neg_infinity
      # masked queries are left as-is (whatever the base mask / mode gives)
    
    elif attn_mode == 'solo_causal':
      # clean block: same as causal_context
      mask_start = -num_clean_and_mask + num_clean
      if num_clean > 0:
        clean_slice = slice(-num_clean_and_mask, mask_start)
        ones = torch.ones(num_clean, num_clean, device=qkv.device)
        A = self.neg_infinity * torch.triu(ones, diagonal=1)
        A = A.view(1, 1, num_clean, num_clean)
        # causal over clean
        attn_scores[:, :, clean_slice, clean_slice] += A
        # clean should not attend to masked
        attn_scores[:, :, clean_slice, mask_start:] = self.neg_infinity
      # masked block: only self-attend among masked
      num_masked = num_clean_and_mask - num_clean
      if num_masked > 0:
        masked_slice = slice(mask_start, None)
        ones_m = torch.ones(num_masked, num_masked, device=qkv.device)
        eye_m = torch.eye(num_masked, device=qkv.device)
        # 0 on diag, -inf elsewhere
        B = self.neg_infinity * (ones_m - eye_m)
        B = B.view(1, 1, num_masked, num_masked)
        attn_scores[:, :, masked_slice, masked_slice] += B
    
    else:
      raise ValueError(f"Invalid attention mode for kv Cache: {attn_mode}")


    attn_weights = F.softmax(attn_scores, dim=-1)
    # matmul shape: [bs, h, num gen last + num to gen, d] 
    # shape after tranpose: [bs, num gen last + num to gen, h, d]
    attn_output = torch.matmul(attn_weights, v).transpose(1, 2)
    return einops.rearrange(attn_output, 'b s h d -> b s (h d)')

  def forward(self, x, rotary_cos_sin, c=None, attn_mask=None,
              kv_cache=False, num_clean=None, num_clean_and_mask=None, attn_mode=None,
              use_split_heads=False, use_qk_split=False, use_avg_rotary=False, use_4way=False, use_2way=False):
    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    x_skip = x
    x = self.norm1(x)
    if self.adaLN:
      # self.adaLN_modulation(c): (128, 1536)
      # self.adaLN_modulation(c)[:, None]: (128, 1, 1536)
      # "" .chunk(6, dim=2) returns 6 tuples of shapes (128, 1, 256)
      (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
       gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
      x = modulate_fused(x, shift_msa, scale_msa)

    qkv = einops.rearrange(
      self.attn_qkv(x),
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads).contiguous()

    if kv_cache:
      x = self._attention_with_kv_cache(
        qkv.detach(), rotary_cos_sin,
        num_clean=num_clean, num_clean_and_mask=num_clean_and_mask, attn_mode=attn_mode)
    else:
      if rotary_cos_sin is None:
        # No rotary embeddings - just split qkv
        q, k, v = split_qkv_no_rotary(qkv)
      elif use_split_heads:
        # Per-head rotary: different heads have different positions
        q, k, v = split_and_apply_rotary_pos_emb_split_heads(qkv, rotary_cos_sin)
      elif use_qk_split:
        # Q/K split rotary: Q uses target positions, K uses source positions
        q, k, v = split_and_apply_rotary_pos_emb_qk_split(qkv, rotary_cos_sin)
      elif use_avg_rotary:
        # Average of source and target rotary for both Q and K
        q, k, v = split_and_apply_rotary_pos_emb_avg(qkv, rotary_cos_sin)
      elif use_4way:
        # 4-way head split: all combinations of source/target for Q/K
        q, k, v = split_and_apply_rotary_pos_emb_4way(qkv, rotary_cos_sin)
      elif use_2way:
        # 2-way head split: keys always use source, queries split between source/target
        q, k, v = split_and_apply_rotary_pos_emb_2way(qkv, rotary_cos_sin)
      elif rotary_cos_sin[0].shape[0] > 1:
        q, k, v = split_and_apply_rotary_pos_emb_batch(qkv, rotary_cos_sin)
      else:
        q, k, v = split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin)

      if FORCE_NAIVE_ATTENTION:
        # here, attn_mask MUST be a dense bool tensor [B or 1, 1, S, S]
        x = naive_attention_multi_headed(q, k, v, attn_mask)
      else:
        x = flex_attention_multi_headed(q, k, v, attn_mask)


    if self.adaLN:
      x = bias_dropout_scale_fn(self.attn_out(x),
                                None,
                                gate_msa,
                                x_skip,
                                self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(
          self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x, self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, scale, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(self.norm2(x)), None, scale, x, self.dropout)

    return x


class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x):
    if x.ndim == 2:
      return self.embedding[x]
    assert x.ndim == 3
    return torch.einsum(
      "blv,ve->ble",
      torch.nn.functional.softmax(x, dim=-1).float(),
      self.embedding.float()).to(x.dtype)


class DDiTFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim,
               adaLN, use_gpt2_arch=False):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size, use_bias=use_gpt2_arch)
    self.linear = nn.Linear(hidden_size, out_channels)
    # Only zero-initialize when adaLN=True, because the adaLN modulation
    # (with scale initialized to 0) provides gradient flow through (1+scale)*x.
    # When adaLN=False, zero weights would block all gradients to earlier layers.
    if adaLN:
      self.linear.weight.data.zero_()
      self.linear.bias.data.zero_()
    self.adaLN = adaLN
    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim,
                                        2 * hidden_size,
                                        bias=True)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()


  def forward(self, x, c):
    x = self.norm_final(x)
    if self.adaLN:
      shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
      x = modulate_fused(x, shift, scale)
    x = self.linear(x)
    return x

class DecodeHead(nn.Module):
  def __init__(self, hidden_size):
    super().__init__()
    self.linear = nn.Linear(hidden_size, 1)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()

  def forward(self, x):
    return self.linear(x).squeeze(-1)

class DiT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)
    self.causal = config.algo.causal_attention
    # adaLN is typically enabled for non-causal (diffusion) models
    # Can be disabled for debugging via algo.disable_adaln
    disable_adaln = config.model.get('disable_adaln', False)
    self.adaLN = (not self.causal) and (not disable_adaln)
    self.config = config
    self.vocab_size = vocab_size
    dim = config.model.hidden_size
    cond_dim = config.model.cond_dim

    # GPT-2 architecture mode: adds biases to LayerNorm and attention projections
    use_gpt2_arch = config.model.get('use_gpt2_arch', False)

    self.vocab_embed = EmbeddingLayer(dim, vocab_size)
    # Only create sigma_map when adaLN is enabled (uses timestep conditioning)
    if self.adaLN:
      self.sigma_map = TimestepEmbedder(cond_dim)

    # RoPE is enabled by default, can be disabled via config
    self.use_rope = config.model.get('use_rope', True)
    self.rotary_dim = dim // config.model.n_heads
    if self.use_rope:
      self.rotary_emb = Rotary(self.rotary_dim)
    else:
      self.rotary_emb = None

    # Optional absolute positional embeddings (in addition to RoPE)
    if config.model.get('absolute_pos_embed', False):
      print('DiT: Using absolute positional embeddings')
      max_len = int(config.model.length)
      self.abs_pos = nn.Embedding(max_len, dim)
    else:
      self.abs_pos = None

    blocks = []
    for _ in range(config.model.n_blocks):
      if self.causal:
        block = DDiTBlockCausal(
          dim=dim,
          n_heads=config.model.n_heads,
          dropout=config.model.dropout,
          use_gpt2_arch=use_gpt2_arch)
      else:
        block = DDiTBlock(
          dim=dim,
          n_heads=config.model.n_heads,
          cond_dim=cond_dim,
          adaLN=self.adaLN,
          dropout=config.model.dropout,
          use_gpt2_arch=use_gpt2_arch)
      blocks.append(block)
    self.blocks = nn.ModuleList(blocks)

    self.output_layer = DDiTFinalLayer(
      hidden_size=dim,
      out_channels=vocab_size,
      cond_dim=cond_dim,
      adaLN=self.adaLN,
      use_gpt2_arch=use_gpt2_arch)
    self.scale_by_sigma = config.model.scale_by_sigma

    # Weight tying: share embedding weights with output layer
    if config.model.get('tie_word_embeddings', False):
      self.output_layer.linear.weight = self.vocab_embed.embedding

    # Shifted logits: position i-1's output predicts position i (diffusion-vs-ar style)
    self.shifted_logits = config.algo.get('shifted_logits', False)

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def reset_kv_cache(self):
    for block in self.blocks:
      block.reset_kv_cache()

  def forward(self, x, sigma, sort_idx=None, x0=None, kv_cache=False):
    assert x0 is None
    seq_len = x.shape[1]
    x = self.vocab_embed(x)
    
    # Add absolute positional embeddings if enabled
    if self.abs_pos is not None:
      pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
      pos_emb = self.abs_pos(pos_ids)
      # Scale like EsoLMDiT does
      pos_emb = pos_emb / (2 * math.sqrt(self.config.model.hidden_size))
      x = x + pos_emb
    
    if self.adaLN:
      t_cond = F.silu(self.sigma_map(sigma))
    else:
      t_cond = None

    rotary_cos_sin = self.rotary_emb(x) if self.rotary_emb is not None else None
    if kv_cache:
      x = x[:, -1:, :]
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](
          x, rotary_cos_sin, c=t_cond, kv_cache=kv_cache)
      x = self.output_layer(x, c=t_cond)
    if self.shifted_logits:
      assert not kv_cache, "shifted_logits is not compatible with kv_cache"
      # Shift logits: position i-1's output predicts position i
      # shifted[0] = x[0], shifted[i] = x[i-1] for i > 0
      x = torch.cat([x[:, 0:1, :], x[:, :-1, :]], dim=1)
    return x


class EsoLMDiT(DiT):
  def __init__(self, config, vocab_size: int, mask_index: int):
    super().__init__(config, vocab_size)
    # sequential not causal
    # this also makes sure that
    # - sigma_map was created
    # - DDiTBlock was used instead of DDiTBlockCausal
    assert not self.causal  # adaLN can be disabled for debugging
    self.mask_index = mask_index

    self.diffusion_attn_mode = config.algo.diffusion_attn_mode
    
    # Positional encoding strategy for DiffuParallel
    # - "target": use target position for both (original behavior)
    # - "split_abs_source": abs_pos uses source, rotary uses target
    # - "split_avg_heads": abs_pos uses avg, rotary split by heads
    self.pos_encoding_strategy = config.algo.get('pos_encoding_strategy', 'target')
    self.n_heads = config.model.n_heads

    self.mdlm_mask = None

    self.use_decode_head = config.model.get('decode_head', False)
    if self.use_decode_head:
      self.decode_head = DecodeHead(config.model.hidden_size)

    self._decode_scores = None

    # Validate shifted_logits is only used with full/bidirectional attention
    if self.shifted_logits:
      assert self.diffusion_attn_mode == 'full', (
        f"shifted_logits requires diffusion_attn_mode='full', "
        f"got '{self.diffusion_attn_mode}'"
      )

  def _apply_logits_shift(self, x):
    """Shift logits so position i-1's output predicts position i.

    This matches the diffusion-vs-ar convention where GPT-2's
    next-token prediction pattern is preserved even with bidirectional attention.
    Position 0 is duplicated (predicts itself).
    """
    # x shape: [B, L, vocab_size]
    # Result: shifted[0] = x[0], shifted[i] = x[i-1] for i > 0
    return torch.cat([x[:, 0:1, :], x[:, :-1, :]], dim=1)

  def _absolute_pos(self, seq_len: int, sort_idx: torch.Tensor = None):
    """
    Returns [bs, L, dim] absolute position embeddings.
    If sort_idx is given, positions are reindexed per-sample to match your reordered tokens.
    """
    L = seq_len
    bs = sort_idx.shape[0] if sort_idx is not None else 1
    device = sort_idx.device if sort_idx is not None else self.abs_pos.weight.device
    pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(bs, -1)  # [bs, L]
    if sort_idx is not None:
      pos_ids = torch.gather(pos_ids, dim=1, index=sort_idx)              # [bs, seq_len]
    return self.abs_pos(pos_ids)  # [bs, L, dim]
  
  def _sort_rotary_cos_sin(self, rotary_cos_sin, sort_idx):
    # example cos shape: (1, 128, 3, 1, 32)
    # 128 for seq_len, 3 for qkv, 32 for head dim
    cos, sin = rotary_cos_sin
    bs = sort_idx.shape[0]
    cos = cos.expand(bs, -1, -1, -1, -1)
    sin = sin.expand(bs, -1, -1, -1, -1)
    cos = torch.gather(
      cos, dim=1, 
      index=sort_idx[:, :, None, None, None].expand(
        -1, -1, 3, -1, self.rotary_dim)).contiguous()
    sin = torch.gather(
      sin, dim=1, 
      index=sort_idx[:, :, None, None, None].expand(
        -1, -1, 3, -1, self.rotary_dim)).contiguous()
    return cos, sin

  def _sort_rotary_cos_sin_split_heads(self, rotary_cos_sin, sort_idx_source, sort_idx_target):
    """
    Create rotary embeddings with source positions for first half of heads,
    target positions for second half of heads.
    
    Returns rotary with explicit head dimension: (bs, seq_len, 3, n_heads, rotary_dim)
    """
    cos, sin = rotary_cos_sin
    bs = sort_idx_source.shape[0]
    seq_len = sort_idx_source.shape[1]
    
    # Expand base rotary to batch size
    cos = cos.expand(bs, -1, -1, -1, -1)  # (bs, full_seq, 3, 1, rotary_dim)
    sin = sin.expand(bs, -1, -1, -1, -1)
    
    # Gather for source positions
    gather_idx = sort_idx_source[:, :, None, None, None].expand(-1, -1, 3, -1, self.rotary_dim)
    cos_source = torch.gather(cos, dim=1, index=gather_idx)  # (bs, seq_len, 3, 1, rotary_dim)
    sin_source = torch.gather(sin, dim=1, index=gather_idx)
    
    # Gather for target positions
    gather_idx = sort_idx_target[:, :, None, None, None].expand(-1, -1, 3, -1, self.rotary_dim)
    cos_target = torch.gather(cos, dim=1, index=gather_idx)
    sin_target = torch.gather(sin, dim=1, index=gather_idx)
    
    # Split heads: first half use source, second half use target
    n_heads_half = self.n_heads // 2
    # Expand to have explicit head dimension: (bs, seq_len, 3, n_heads, rotary_dim)
    cos_source = cos_source.expand(-1, -1, -1, n_heads_half, -1)
    sin_source = sin_source.expand(-1, -1, -1, n_heads_half, -1)
    cos_target = cos_target.expand(-1, -1, -1, self.n_heads - n_heads_half, -1)
    sin_target = sin_target.expand(-1, -1, -1, self.n_heads - n_heads_half, -1)
    
    # Concatenate along head dimension
    cos_split = torch.cat([cos_source, cos_target], dim=3).contiguous()
    sin_split = torch.cat([sin_source, sin_target], dim=3).contiguous()
    
    return cos_split, sin_split

  def _sort_rotary_cos_sin_qk_split(self, rotary_cos_sin, sort_idx_source, sort_idx_target):
    """
    Create rotary embeddings with:
    - Q (queries): target positions (asking "what goes at target position?")
    - K (keys): source positions (saying "I'm a token from source position")
    
    Returns: (cos, sin) each with shape (bs, seq_len, 2, rotary_dim)
             where dim 2 is [q_rotary, k_rotary]
    """
    cos, sin = rotary_cos_sin
    bs = sort_idx_source.shape[0]
    
    # Expand base rotary to batch size
    cos = cos.expand(bs, -1, -1, -1, -1)  # (bs, full_seq, 3, 1, rotary_dim)
    sin = sin.expand(bs, -1, -1, -1, -1)
    
    # Gather for source positions (for K)
    gather_idx_source = sort_idx_source[:, :, None, None, None].expand(-1, -1, 1, -1, self.rotary_dim)
    # Only need index 0 since we'll apply same to all of q or k
    cos_source = torch.gather(cos[:, :, :1, :, :], dim=1, index=gather_idx_source)  # (bs, seq_len, 1, 1, rotary_dim)
    sin_source = torch.gather(sin[:, :, :1, :, :], dim=1, index=gather_idx_source)
    
    # Gather for target positions (for Q)
    gather_idx_target = sort_idx_target[:, :, None, None, None].expand(-1, -1, 1, -1, self.rotary_dim)
    cos_target = torch.gather(cos[:, :, :1, :, :], dim=1, index=gather_idx_target)  # (bs, seq_len, 1, 1, rotary_dim)
    sin_target = torch.gather(sin[:, :, :1, :, :], dim=1, index=gather_idx_target)
    
    # Concatenate: [q_rotary (target), k_rotary (source)]
    # Shape: (bs, seq_len, 2, 1, rotary_dim)
    cos_qk = torch.cat([cos_target, cos_source], dim=2).contiguous()
    sin_qk = torch.cat([sin_target, sin_source], dim=2).contiguous()
    
    return cos_qk, sin_qk

  def _get_attention_mask(self, seq_len, attn_mode, cutoffs, mask_cutoffs):
    """
    Get attention mask for the given attention mode.
    
    Args:
      seq_len: Sequence length
      attn_mode: Attention mode ('causal', 'bidirectional', 'causal_context', 'causal_output', 'full', 'solo_causal', 'solo_full')
      cutoffs: [B] tensor indicating boundary between clean and masked tokens
      mask_cutoffs: [B] tensor indicating rightmost position any query can attend to
    """
    if attn_mode == 'causal':
      # if self.mdlm_mask is None:
        # self.mdlm_mask = _get_causal_mask(seq_len)
      # return self.mdlm_mask
      return _get_causal_mask(seq_len)
    elif attn_mode == 'bidirectional':
      # if self.mdlm_mask is None:
        # self.mdlm_mask = _get_bidirectional_mask(seq_len)
      # return self.mdlm_mask
      return _get_bidirectional_mask(seq_len)
    elif attn_mode == 'causal_context':
      # causal over clean tokens
      # bidirectional over masked tokens
      return _get_causal_context_mask(seq_len=seq_len,
                             cutoffs=cutoffs,
                             mask_cutoffs=mask_cutoffs)
    elif attn_mode == 'causal_output':
      # bidirectional over clean tokens
      # causal over masked tokens
      return _get_causal_output_mask(seq_len=seq_len,
                              cutoffs=cutoffs,
                              mask_cutoffs=mask_cutoffs)
    elif attn_mode == 'full':
      return _get_full_mask(seq_len=seq_len, mask_cutoffs=mask_cutoffs)
    elif attn_mode == 'solo_causal':
      # clean tokens: causal among themselves, cannot attend to masked tokens
      # masked tokens: can attend to all clean tokens, cannot attend to other masked tokens
      return _get_solo_causal_mask(seq_len=seq_len,
                                cutoffs=cutoffs)
    elif attn_mode == 'solo_full':
      # clean tokens: full bidirectional among themselves, cannot attend to masked tokens
      # masked tokens: can attend to all clean tokens and themselves only
      return _get_solo_full_mask(seq_len=seq_len,
                                cutoffs=cutoffs)

  def _diffusion_features(self, zt, sort_idx=None,
                          attn_mode=None, cutoffs=None, mask_cutoffs=None, rotary_range=None,
                          sort_idx_source=None, sort_idx_target=None):
    """
    Compute features for diffusion forward pass.
    
    For DiffuParallel with pos_encoding_strategy:
    - "target": uses sort_idx (or sort_idx_target) for both abs_pos and rotary
    - "split_abs_source": abs_pos uses sort_idx_source, rotary uses sort_idx_target
    - "split_avg_heads": abs_pos uses (source+target)/2, rotary split by heads
    
    Args:
      zt: input tokens
      sort_idx: single sort index (backward compat, used for 'target' strategy)
      sort_idx_source: source positions (where input tokens come from)
      sort_idx_target: target positions (what to predict)
      ... other args unchanged
    """
    if cutoffs is None:
      cutoffs = torch.sum(zt != self.mask_index, dim=1)
    if attn_mode is None:
      attn_mode = self.diffusion_attn_mode
    # Default mask_cutoffs to seq_len (no restriction) if not provided
    if mask_cutoffs is None:
      mask_cutoffs = torch.full((zt.shape[0],), zt.shape[1], device=zt.device, dtype=torch.long)
    
    # Handle backward compatibility: if only sort_idx is provided, use it as target
    if sort_idx is not None and sort_idx_target is None:
      sort_idx_target = sort_idx
    if sort_idx_source is None:
      sort_idx_source = sort_idx_target  # Fallback to target if source not provided
    
    x = self.vocab_embed(zt)

    # Handle rotary embeddings (skip if use_rope is False)
    if self.use_rope:
      if rotary_range is None:
        rotary_cos_sin = self.rotary_emb(x)
      else:
        rotary_cos_sin = self.rotary_emb(torch.ones((x.shape[0], rotary_range)).to(x.device))

      # Determine which rotary strategy to use
      use_split_heads = (self.pos_encoding_strategy == 'split_avg_heads')
      use_qk_split = (self.pos_encoding_strategy == 'qk_split')
      use_avg_rotary = (self.pos_encoding_strategy == 'avg_rotary')
      use_4way = (self.pos_encoding_strategy == '4way_heads')
      use_2way = (self.pos_encoding_strategy == '2way_heads')

      if sort_idx_target is None:
        # No shuffling: use rotary embeddings as-is (identity positions)
        # Just expand to batch size
        cos, sin = rotary_cos_sin
        bs = zt.shape[0]
        rotary_cos_sin = (cos.expand(bs, -1, -1, -1, -1), sin.expand(bs, -1, -1, -1, -1))
      elif use_split_heads:
        # Strategy: split rotary by heads (source for first half, target for second half)
        rotary_cos_sin = self._sort_rotary_cos_sin_split_heads(
          rotary_cos_sin, sort_idx_source, sort_idx_target)
      elif use_qk_split or use_avg_rotary or use_4way or use_2way:
        # These strategies need both source and target rotary embeddings
        # They use the same format: (bs, seq_len, 2, 1, rotary_dim) with [target, source]
        rotary_cos_sin = self._sort_rotary_cos_sin_qk_split(
          rotary_cos_sin, sort_idx_source, sort_idx_target)
      else:
        # Strategies 'target' and 'split_abs_source' both use target for rotary
        rotary_cos_sin = self._sort_rotary_cos_sin(
          rotary_cos_sin, sort_idx_target)
    else:
      rotary_cos_sin = None
      use_split_heads = False
      use_qk_split = False
      use_avg_rotary = False
      use_4way = False
      use_2way = False

    if FORCE_NAIVE_ATTENTION:
      # Build dense bool masks that replicate the same logic.
      if attn_mode == 'causal':
        attention_mask = _dense_causal_mask(zt.shape[1], str(zt.device), str(zt.dtype))
      elif attn_mode == 'bidirectional':
        attention_mask = _dense_full_mask(zt.shape[1], str(zt.device), str(zt.dtype), mask_cutoffs=mask_cutoffs)
      elif attn_mode == 'causal_context':
        attention_mask = _dense_causal_context_mask(zt.shape[1], cutoffs, mask_cutoffs=mask_cutoffs)
      elif attn_mode == 'causal_output':
        attention_mask = _dense_causal_output_mask(zt.shape[1], cutoffs, mask_cutoffs=mask_cutoffs)
      elif attn_mode == 'solo_causal':
        attention_mask = _dense_solo_causal_mask(zt.shape[1], cutoffs)
      elif attn_mode == 'solo_full':
        attention_mask = _dense_solo_full_mask(zt.shape[1], cutoffs)
      elif attn_mode == 'full':
        attention_mask = _dense_full_mask(zt.shape[1], str(zt.device), str(zt.dtype), mask_cutoffs=mask_cutoffs)
      else:
        raise ValueError(f"Unknown attn_mode: {attn_mode}")
    else:
      attention_mask = self._get_attention_mask(seq_len=zt.shape[1], attn_mode=attn_mode, cutoffs=cutoffs, mask_cutoffs=mask_cutoffs)

    # Handle absolute positional embeddings based on strategy
    if self.abs_pos is not None:
      abs_seq_len = rotary_range if rotary_range is not None else zt.shape[1]
      
      pos = self._absolute_pos(seq_len=abs_seq_len, sort_idx=sort_idx_source)
      # if self.pos_encoding_strategy == 'split_abs_source':
        # Strategy 1: abs_pos uses source positions
        # pos = self._absolute_pos(seq_len=abs_seq_len, sort_idx=sort_idx_source)
      # elif self.pos_encoding_strategy == 'split_avg_heads':
      #   # Strategy 2: abs_pos uses average of source and target
      #   # Need to interpolate positions - use float positions then embed
      #   # pos_source = self._absolute_pos(seq_len=abs_seq_len, sort_idx=sort_idx_source)
      #   # pos_target = self._absolute_pos(seq_len=abs_seq_len, sort_idx=sort_idx_target)
      #   # pos = (pos_source + pos_target) / 2
      # else:
      #   # Default 'target' strategy: use target positions (original behavior)
      #   pos = self._absolute_pos(seq_len=abs_seq_len, sort_idx=sort_idx_target)
      
      pos = pos / (2 * math.sqrt(self.config.model.hidden_size))
      x = x + pos
  
    return {'x': x,
            'rotary': rotary_cos_sin,
            'attention': attention_mask,
            'sorted_indices': sort_idx_target,
            'use_split_heads': use_split_heads,
            'use_qk_split': use_qk_split,
            'use_avg_rotary': use_avg_rotary,
            'use_4way': use_4way,
            'use_2way': use_2way}

  def forward(self, zt, sigma, sort_index=None, x0=None, mask_cutoffs=None, rotary_range=None,
              sort_idx_source=None, sort_idx_target=None):
    """
    Forward pass for EsoLMDiT.
    
    Args:
      zt: input tokens
      sigma: noise level (unused for DiffuParallel)
      sort_index: single sort index (backward compat)
      sort_idx_source: source positions (where input tokens come from)
      sort_idx_target: target positions (what to predict)
      ... other args unchanged
    """
    features = self._diffusion_features(
      zt, sort_idx=sort_index, 
      mask_cutoffs=mask_cutoffs, rotary_range=rotary_range,
      sort_idx_source=sort_idx_source, sort_idx_target=sort_idx_target)
    x = features['x']
    use_split_heads = features.get('use_split_heads', False)
    use_qk_split = features.get('use_qk_split', False)
    use_avg_rotary = features.get('use_avg_rotary', False)
    use_4way = features.get('use_4way', False)
    use_2way = features.get('use_2way', False)
    
    if self.adaLN:
      t_cond = F.silu(self.sigma_map(sigma))
    else:
      t_cond = None
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, features['rotary'], c=t_cond,
                           attn_mask=features['attention'],
                           use_split_heads=use_split_heads,
                           use_qk_split=use_qk_split,
                           use_avg_rotary=use_avg_rotary,
                           use_4way=use_4way,
                           use_2way=use_2way)
      self._decode_scores = self.decode_head(x) if self.use_decode_head else None
      x = self.output_layer(x, c=t_cond)
    if self.shifted_logits:
      x = self._apply_logits_shift(x)
    return x

  @torch.no_grad()
  def forward_sample(self, zt, sort_idx, attn_mode=None,
                     cutoffs=None, kv_cache=False,
                     last_k_start=None,
                     curr_k_start=None,
                     curr_k_end=None,
                     mask_cutoffs=None,
                     rotary_range=None):
    """
    zt is expected to be sorted as per sort_idx.
    
    When kv_cache is true:
    - zt will have shape (num_samples, model.length); we need its shape 
      to generate all the rotary embeddings because any of them can be
      selected by the random ordering
    - sort_idx will have shape  (num_samples, model.length) for the same reason

    Within self._diffusion_features, zt will be used
    to generate the full rotary embeddings, and sort_idx
    will be used to index the embedded zt into shape
    (num_samples, num_tokens_generated_last_time (non-mask) + num_tokens_to_gen (mask), hidden)
    
    Args:
      mask_cutoffs: Optional[int or Tensor]. If provided, restricts attention so that
                    tokens can only attend to positions < mask_cutoffs.
    """
    assert attn_mode is not None
    ones = torch.ones(zt.shape[0], device=zt.device)
    if cutoffs is not None:
      cutoffs = cutoffs * ones
      assert cutoffs.ndim == 1
    # Convert scalar mask_cutoffs to tensor if needed
    if mask_cutoffs is not None and not isinstance(mask_cutoffs, torch.Tensor):
      mask_cutoffs = torch.full((zt.shape[0],), mask_cutoffs, device=zt.device, dtype=torch.long)
    features = self._diffusion_features(
      zt=zt,
      sort_idx=sort_idx,
      attn_mode=attn_mode,
      cutoffs=cutoffs,
      mask_cutoffs=mask_cutoffs,
      rotary_range=rotary_range)
    if self.adaLN:
      zeros = torch.zeros(zt.shape[0], device=zt.device)
      t_cond = F.silu(self.sigma_map(zeros))
    else:
      t_cond = None

    x = features['x']
    rotary = features['rotary']
    attn_mask = features['attention']
    if kv_cache:
      # expect x to be sorted
      x = x[:, last_k_start:curr_k_end, :]
      # rotary is already sorted here
      # looking ahead
      if rotary is not None:
        cos, sin = rotary
        rotary = (cos[:, :curr_k_end], sin[:, :curr_k_end])
      num_clean = curr_k_start - last_k_start
      num_clean_and_mask = curr_k_end - last_k_start
    else:
      num_clean = None
      num_clean_and_mask = None

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](
          x, rotary, c=t_cond,
          attn_mask=attn_mask,
          kv_cache=kv_cache,
          num_clean=num_clean,
          num_clean_and_mask=num_clean_and_mask,
          attn_mode=attn_mode)
      self._decode_scores = self.decode_head(x) if self.use_decode_head else None
      x = self.output_layer(x, c=t_cond)

    if self.shifted_logits:
      assert not kv_cache, "shifted_logits is not compatible with kv_cache"
      x = self._apply_logits_shift(x)

    if kv_cache:
      x = x[:, num_clean:, :]
      # keep them same shape
      if self.use_decode_head:
        self._decode_scores = self._decode_scores[:, num_clean:]

    return x

  def get_decode_scores(self):
    return self._decode_scores