import math
import typing
from contextlib import nullcontext
from dataclasses import dataclass

import einops
from einops import rearrange
from functools import partial
try:
  import flash_attn
  import flash_attn.layers.rotary
except:
  pass
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
  from torch.nn.attention.flex_attention import flex_attention, create_block_mask
  FLEX_ATTN_AVAILABLE = True
except:
  FLEX_ATTN_AVAILABLE = False

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
  """
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
  block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

  # **2. Offset Block-Causal Mask (M_OBC) **
  offset_block_causal = (
    (block_q > block_kv)
    & (x0_flag_kv == 1)
    & (x0_flag_q == 0)
  )

  # **3. Block-Causal Mask (M_BC) **
  block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

  # **4. Combine Masks **
  return block_diagonal | offset_block_causal | block_causal


def sample_block_causal_mask(
    b, h, q_idx, kv_idx, block_size=None):
  """Block-causal mask for a clean prefix followed by one active block."""
  return q_idx // block_size >= kv_idx // block_size

@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def fused_flex_attention(q, k, v, mask=None):
    return flex_attention(q, k, v, block_mask=mask)


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


def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rope_coordinate(x, positions, base=10_000):
  """Apply parameter-free RoPE to one coordinate of a feature slice."""
  dim = x.shape[-1]
  if dim == 0:
    return x
  if dim % 2 != 0:
    raise ValueError(f'RoPE dimension must be even, got {dim}')
  positions = torch.as_tensor(positions, device=x.device, dtype=torch.float32)
  if positions.ndim == 0:
    positions = positions.expand(x.shape[-2])
  inv_freq = 1.0 / (
    base ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
  freqs = torch.einsum('s,d->sd', positions, inv_freq)
  embedding = torch.cat((freqs, freqs), dim=-1)
  cos = embedding.cos().to(x.dtype)[None, None, :, :]
  sin = embedding.sin().to(x.dtype)[None, None, :, :]
  return x * cos + rotate_half(x) * sin


def apply_denoising_rope_2d(
    x, spatial_positions, temporal_position, spatial_dim):
  """Apply spatial RoPE to the first slice and temporal RoPE to the rest."""
  spatial = apply_rope_coordinate(
    x[..., :spatial_dim], spatial_positions)
  temporal = apply_rope_coordinate(
    x[..., spatial_dim:], temporal_position)
  return torch.cat((spatial, temporal), dim=-1)


def split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin):
  with torch.amp.autocast('cuda', enabled=False):
    cos, sin = rotary_cos_sin
    cos = cos.to(qkv.dtype)
    sin = sin.to(qkv.dtype)
    cos = cos[0,:,0,0,:cos.shape[-1]//2]
    sin = sin[0,:,0,0,:sin.shape[-1]//2]
    q, k, v = qkv.chunk(3, dim=2)
    q = flash_attn.layers.rotary.apply_rotary_emb_torch(
      q.squeeze(dim=2), cos, sin)
    k = flash_attn.layers.rotary.apply_rotary_emb_torch(
      k.squeeze(dim=2), cos, sin)
    v = v.squeeze(dim=2)
  return q, k, v

def apply_rotary_pos_emb_torchscript(qkv, cos, sin):
    return (qkv * cos) + (rotate_half(qkv) * sin)

def apply_rotary_pos_emb(qkv, cos, sin):
  cos = cos[0,:,0,0,:cos.shape[-1]//2]
  sin = sin[0,:,0,0,:sin.shape[-1]//2]
  return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


def regular_attention_multi_headed(q, k, v):
  # Assuming qkv is a tensor with shape [batch, seq_len, 3, num_heads, head_dim]
  # where the 3 represents Q, K, V packed in that order
  attention_output = F.scaled_dot_product_attention(
    query=q.transpose(1, 2),
    key=k.transpose(1, 2),
    value=v.transpose(1, 2),
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False)
  # [batch_size, seq_len, num_heads, head_dim]
  attention_output = attention_output.transpose(1, 2)
  return einops.rearrange(attention_output, 'b s h d -> b s (h d)')


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    with torch.amp.autocast('cuda', enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None, None, :]


@dataclass
class DcachehoopingBackboneOutput:
  """Optional rich output used by recurrent workspace pretraining."""
  logits: torch.Tensor
  step_kv: typing.Optional[typing.List[torch.Tensor]]
  final_hidden: torch.Tensor
  confidence_logits: typing.Optional[torch.Tensor]


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
      * torch.arange(start=0, end=half).to(t.dtype).to(t.device)
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
  def __init__(self, n, dim, n_heads, mlp_ratio=4, dropout=0.1, max_batch_size=64, max_seqlen=1024, adaLN=False, cond_dim=None, attn_backend='flash_attn'):
    super().__init__()
    self.n_heads = n_heads
    self.max_seqlen = max_seqlen
    self.n = n

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout
    self.adaLN = adaLN
    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()
    self.attn_backend = attn_backend
    self.kv_cache = None

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def get_qkv(self, x, rotary_cos_sin, store_kv=False):
    # compute qkv (potentially use cache)
    if self.kv_cache is not None:
      new_qkv = self.attn_qkv(x[:, -1:])
      qkv = torch.cat((self.kv_cache, new_qkv), dim=1)
    else:
      qkv = self.attn_qkv(x)
    # store kv cache in a sliding window (can't exceed context len)
    if store_kv:
      self.kv_cache = qkv[:, -(self.max_seqlen-1):].clone()
      
    qkv = einops.rearrange(
      qkv,
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads)
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = rotary_cos_sin
      if self.attn_backend == 'flash_attn':
        qkv = apply_rotary_pos_emb(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
      else:
        qkv = apply_rotary_pos_emb_torchscript(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
          
    return qkv

  def cross_attn(self, qkv, mask=None):
    scale = qkv.shape[-1]
    qkv = qkv.transpose(1, 3)
    mask = mask.bool() if mask is not None else None
    x = F.scaled_dot_product_attention(
      query=qkv[:, :, 0],
      key=qkv[:, :, 1],
      value=qkv[:, :, 2],
      attn_mask=mask,
      is_causal=True,
      scale=1 / math.sqrt(scale))
    x = x.transpose(1, 2)
    x = rearrange(x, 'b s h d -> b s (h d)')
    return x

  def forward(self,
              x,
              rotary_cos_sin,
              c=None,
              causal=True,
              mask=None,
              store_kv=False,
              **kwargs):
    del kwargs
    batch_size, seq_len = x.shape[0], x.shape[1]
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = None, None, None, None, None, None
    bias_dropout_scale_fn = self._get_bias_dropout_scale()
    if c is not None and c.shape[0] == batch_size:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
      scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
    elif c is not None:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
      scale_mlp, gate_mlp) = rearrange(
        self.adaLN_modulation(c), '(b h) d -> b h d', b=batch_size
        ).chunk(6, dim=-1)

    # attention operation
    x_skip = x
    if c is not None:
      x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
    else:
      x = self.norm1(x)
    
    qkv = self.get_qkv(x, rotary_cos_sin, store_kv=store_kv)
    if self.attn_backend == 'flash_attn':
      qkv = einops.rearrange(qkv, 'b s ... -> (b s) ...')
      cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len,
        step=seq_len, dtype=torch.int32, device=qkv.device)
      x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, seq_len, 0.0, causal=True)
      x = einops.rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
    else:
      x = self.cross_attn(qkv, c)
      
    if c is not None:
      x = bias_dropout_scale_fn(self.attn_out(x),
        None,
        gate_msa,
        x_skip,
        self.dropout)
      # mlp operation
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


class DDiTBlock(nn.Module):
  def __init__(self, n, dim, n_heads, adaLN,
               latent_dim=None, cond_dim=None,
               latent_conditioning=-1, mlp_ratio=4,
               dropout=0.1, block_size=1,
               max_batch_size=64, max_seqlen=1024, attn_backend='flash_attn',
               step_memory_enabled=False, dc_spatial_rope_dim=None,
               dc_temporal_rope_dim=None,
               step_memory_gate_enabled=False,
               step_memory_gate_init=0.1, attention_mode='separate',
               current_only_merged=False, merged_policy='legacy'):
    super().__init__()
    self.max_seqlen = max_seqlen
    self.n = n
    self.n_heads = n_heads
    self.adaLN = adaLN
    self.latent_conditioning = latent_conditioning
    self.block_size = block_size

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    # The denoising branch is a complete second attention sublayer. Its
    # projections and normalization are independent of normal BD3 attention.
    self.step_memory_enabled = step_memory_enabled
    if attention_mode not in {'separate', 'merged'}:
      raise ValueError('Unknown step-memory attention_mode')
    if attention_mode == 'merged' and not step_memory_enabled and not current_only_merged:
      raise ValueError('Merged attention requires step memory enabled')
    self.attention_mode = attention_mode
    if merged_policy not in {'legacy', 'current_preserving'}:
      raise ValueError('Unknown step-memory merged_policy')
    if merged_policy == 'current_preserving' and (
        attention_mode != 'merged' or step_memory_gate_enabled):
      raise ValueError(
        'current_preserving requires merged attention with the previous-V gate disabled')
    self.merged_policy = merged_policy
    self.dc_norm = None
    self.dc_qkv = None
    self.dc_attn_out = None
    self.dc_dropout = None
    self.dc_spatial_rope_dim = None
    self.dc_temporal_rope_dim = None
    self.step_memory_gate = None
    if self.step_memory_enabled or (attention_mode == 'merged' and current_only_merged):
      if self.attention_mode == 'separate':
        self.dc_norm = LayerNorm(dim)
        self.dc_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.dc_attn_out = nn.Linear(dim, dim, bias=False)
        self.dc_dropout = nn.Dropout(dropout)
      head_dim = dim // n_heads
      if dc_temporal_rope_dim is None:
        dc_temporal_rope_dim = max(2, head_dim // 4)
      if dc_spatial_rope_dim is None:
        dc_spatial_rope_dim = head_dim - dc_temporal_rope_dim
      if dc_spatial_rope_dim + dc_temporal_rope_dim != head_dim:
        raise ValueError(
          'Denoising spatial and temporal RoPE dimensions must sum to head_dim')
      if dc_spatial_rope_dim % 2 or dc_temporal_rope_dim % 2:
        raise ValueError('Denoising RoPE dimensions must both be even')
      self.dc_spatial_rope_dim = dc_spatial_rope_dim
      self.dc_temporal_rope_dim = dc_temporal_rope_dim
      if step_memory_gate_enabled and self.step_memory_enabled:
        if not -1.0 < step_memory_gate_init < 1.0:
          raise ValueError('Step-memory gate initialization must be in (-1, 1)')
        raw_gate = math.atanh(float(step_memory_gate_init))
        self.step_memory_gate = nn.Parameter(torch.tensor(raw_gate))

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout
    self.kv_cache = None
    self.cache_idx = 0

    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()
    self.attn_backend = attn_backend

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def get_qkv(self, x, rotary_cos_sin, store_kv=False):
    # compute qkv (potentially use cache)
    if self.kv_cache is not None:
      new_qkv = self.attn_qkv(x)
      self.kv_cache[:, self.cache_idx:self.cache_idx+self.block_size] = new_qkv
      qkv = self.kv_cache[:, :self.cache_idx+self.block_size].clone()
    else:
      qkv = self.attn_qkv(x)
    # store kv cache in a sliding window (can't exceed context len)
    if store_kv:
      self.cache_idx += self.block_size
      if self.cache_idx >= self.max_seqlen:
        # left-shift the cache
        self.cache_idx = self.max_seqlen - self.block_size
        self.kv_cache[:, :-self.block_size] = self.kv_cache[:, self.block_size:].clone()

    qkv = einops.rearrange(
      qkv,
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads)
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = rotary_cos_sin
      if self.attn_backend == 'flash_attn':
        qkv = apply_rotary_pos_emb(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
      else:
        qkv = apply_rotary_pos_emb_torchscript(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
    return qkv
  
  def attention_residual(self, x, c, gate_msa, x_skip):
    bias_dropout_scale_fn = self._get_bias_dropout_scale()
    if c is not None:
      x = bias_dropout_scale_fn(self.attn_out(x),
        None,
        gate_msa,
        x_skip,
        self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, scale, x_skip, self.dropout)
    return x

  def mlp_residual(self, x, c, gate_mlp, shift_mlp, scale_mlp):
    bias_dropout_scale_fn = self._get_bias_dropout_scale()
    if c is not None:
      x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(
          self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x, self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(
        self.mlp(self.norm2(x)), None, scale, x, self.dropout)
    return x

  def cross_attn(self, qkv, mask=None):
    scale = qkv.shape[-1]
    qkv = qkv.transpose(1, 3)
    mask = mask.bool() if mask is not None else None
    x = F.scaled_dot_product_attention(
      query=qkv[:, :, 0],
      key=qkv[:, :, 1],
      value=qkv[:, :, 2],
      attn_mask=mask,
      is_causal=False,
      scale=1 / math.sqrt(scale))
    x = x.transpose(1, 2)
    x = rearrange(x, 'b s h d -> b s (h d)')
    return x

  def cross_attn_flex(self, qkv, mask=None):
    qkv = rearrange(qkv, 'b s three h d -> b h three s d', h=self.n_heads)
    x = fused_flex_attention(
      qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], mask=mask)
    x = rearrange(x, 'b h s d -> b s (h d)')
    return x

  def _project_denoising_qkv(self, hidden):
    """Project block-group hidden states into raw denoising Q/K/V."""
    qkv = self.dc_qkv(self.dc_norm(hidden))
    return rearrange(
      qkv, 'b g s (three h d) -> (b g) h three s d',
      three=3, h=self.n_heads)

  def merged_attention(self, hidden, previous_step_kv, detach_cache_backbone,
                       source_mask, shift=None, scale=None,
                       key_valid=None, previous_key_valid=None):
    """One joint softmax; raw shifted cache uses the normal QKV projection.

    Temporal positions are previous=0/current=1, as in the legacy DC branch;
    they encode relative iteration age, NOT the continuous noise level.
    In the legacy policy the optional gate scales previous V only, never the
    full residual. current_preserving has no such gate and forbids cache-only
    queries; current-only queries still remove previous keys before softmax.
    """
    batch, length, dim = hidden.shape
    def project(value):
      value = self.norm1(value)
      if shift is not None:
        value = modulate_fused(value, shift, scale)
      return rearrange(self.attn_qkv(value),
                       'b s (three h d) -> b h three s d',
                       three=3, h=self.n_heads)

    qkv = project(hidden)
    raw = project(hidden.detach()) if detach_cache_backbone else qkv
    entry = torch.stack((raw[:, :, 1], raw[:, :, 2]), dim=2)
    entry = rearrange(entry, 'b h two s d -> b s two h d')
    positions = torch.arange(length, device=hidden.device)
    q = apply_denoising_rope_2d(qkv[:, :, 0], positions, 1, self.dc_spatial_rope_dim)
    k = apply_denoising_rope_2d(qkv[:, :, 1], positions, 1, self.dc_spatial_rope_dim)
    v = qkv[:, :, 2]
    attention_mask = None
    if previous_step_kv is not None:
      if previous_step_kv.shape != entry.shape:
        raise ValueError('Merged previous K/V must match [batch, sequence, 2, heads, head_dim]')
      pk = apply_denoising_rope_2d(previous_step_kv[:, :, 0].transpose(1, 2),
                                 positions, 0, self.dc_spatial_rope_dim)
      pv = previous_step_kv[:, :, 1].transpose(1, 2)
      if self.step_memory_gate is not None:
        pv = torch.tanh(self.step_memory_gate) * pv
      k, v = torch.cat((pk, k), dim=-2), torch.cat((pv, v), dim=-2)
      if source_mask is not None:
        if (source_mask.shape != (batch, length)
            or source_mask.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}):
          raise ValueError('Merged source mask requires integer [batch, sequence] modes 0/1/2')
        invalid_source = (source_mask < 0) | (source_mask > 2)
        if self.merged_policy == 'current_preserving':
          invalid_source = invalid_source | source_mask.eq(1)
        # One validation synchronization, including the new policy constraint.
        if invalid_source.any():
          if self.merged_policy == 'current_preserving':
            raise ValueError('current_preserving forbids cache-only queries; use modes 0/2')
          raise ValueError('Merged source mask requires integer [batch, sequence] modes 0/1/2')
        attention_mask = torch.cat((
          source_mask.ne(2)[:, :, None].expand(-1, -1, length),
          source_mask.ne(1)[:, :, None].expand(-1, -1, length)), dim=-1)[:, None]
    elif source_mask is not None:
      raise ValueError('Source dropout requires a previous denoising cache')
    if key_valid is not None:
      valid = key_valid.bool()
      if previous_step_kv is not None:
        previous_valid = valid if previous_key_valid is None else previous_key_valid.bool()
        valid = torch.cat((previous_valid, valid), dim=-1)
      valid = valid[:, None, None, :]
      attention_mask = valid if attention_mask is None else attention_mask & valid
    attended = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask,
                                             dropout_p=0.0, is_causal=False)
    return rearrange(attended, 'b h s d -> b s (h d)'), entry

  def _write_denoising_kv(self, hidden, detach_backbone=False):
    """Write raw K/V while optionally truncating gradients at the hidden state.

    Re-projecting a detached hidden state keeps the lightweight normalization
    and QKV projection trainable from the next denoising loss without retaining
    a cross-forward graph through the preceding transformer backbone.
    """
    if detach_backbone:
      hidden = hidden.detach()
    qkv = self._project_denoising_qkv(hidden)
    raw_kv = torch.stack((qkv[:, :, 1], qkv[:, :, 2]), dim=2)
    batch_size, groups = hidden.shape[:2]
    raw_kv = rearrange(
      raw_kv,
      '(b g) h two s d -> b g s two h d',
      b=batch_size, g=groups)
    return raw_kv[:, -1]

  def denoising_attn(
      self, hidden, previous_step_kv=None, detach_cache_backbone=False,
      source_mask=None):
    """Jointly attend to previous and current denoising K/V with 2D RoPE.

    Args:
      hidden: `[batch, groups, block, dim]`, where groups are independent
        active blocks during parallel base training.
      previous_step_kv: optional raw `[batch, block, 2, heads, head_dim]`.
      source_mask: optional integer `[batch, block]` query mask. Zero uses
        joint cached/current attention, one permits cached K/V only, and two
        permits current K/V only. It is applied before softmax.

    Returns:
      Projected denoising-attention output and the last group's raw entry K/V.
    """
    batch_size, groups, block_len, dim = hidden.shape
    qkv = self._project_denoising_qkv(hidden)
    current_q = qkv[:, :, 0]
    current_k = qkv[:, :, 1]
    current_v = qkv[:, :, 2]

    spatial_positions = torch.arange(block_len, device=hidden.device)
    current_q = apply_denoising_rope_2d(
      current_q, spatial_positions, temporal_position=1,
      spatial_dim=self.dc_spatial_rope_dim)
    current_k_rotated = apply_denoising_rope_2d(
      current_k, spatial_positions, temporal_position=1,
      spatial_dim=self.dc_spatial_rope_dim)

    if previous_step_kv is not None:
      if groups != 1:
        raise ValueError('Previous denoising K/V is only valid for one active block')
      if previous_step_kv.shape[1] != block_len:
        raise ValueError(
          'Previous denoising K/V length must match the active block length')
      previous_k = previous_step_kv[:, :, 0].transpose(1, 2)
      previous_v = previous_step_kv[:, :, 1].transpose(1, 2)
      previous_k = apply_denoising_rope_2d(
        previous_k, spatial_positions, temporal_position=0,
        spatial_dim=self.dc_spatial_rope_dim)
      attended_k = torch.cat((previous_k, current_k_rotated), dim=-2)
      attended_v = torch.cat((previous_v, current_v), dim=-2)
    else:
      if source_mask is not None:
        raise ValueError('Source dropout requires a previous denoising cache')
      attended_k = current_k_rotated
      attended_v = current_v

    attention_mask = None
    if source_mask is not None:
      if groups != 1:
        raise ValueError('Source dropout is only valid for one active block')
      if source_mask.shape != (batch_size, block_len):
        raise ValueError(
          'Source mask must have shape [batch, active block length]')
      if source_mask.dtype not in {
          torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise ValueError('Source mask must contain integer source modes')
      if ((source_mask < 0) | (source_mask > 2)).any():
        raise ValueError('Source mask modes must be joint=0, cache=1, current=2')
      cache_allowed = source_mask.ne(2)
      current_allowed = source_mask.ne(1)
      cache_allowed = cache_allowed[:, :, None].expand(
        batch_size, block_len, block_len)
      current_allowed = current_allowed[:, :, None].expand(
        batch_size, block_len, block_len)
      attention_mask = torch.cat(
        (cache_allowed, current_allowed), dim=-1)[:, None]

    output = F.scaled_dot_product_attention(
      current_q, attended_k, attended_v,
      attn_mask=attention_mask,
      is_causal=False,
      scale=1 / math.sqrt(current_q.shape[-1]))
    output = rearrange(
      output, '(b g) h s d -> b g s (h d)', b=batch_size, g=groups)
    output = self.dc_attn_out(output)
    output = self.dc_dropout(output)
    if self.step_memory_gate is not None:
      output = torch.tanh(self.step_memory_gate) * output

    if detach_cache_backbone:
      raw_entry_kv = self._write_denoising_kv(
        hidden, detach_backbone=True)
    else:
      raw_entry_kv = torch.stack((current_k, current_v), dim=2)
      raw_entry_kv = rearrange(
        raw_entry_kv,
        '(b g) h two s d -> b g s two h d',
        b=batch_size, g=groups)[:, -1]
    return output, raw_entry_kv

  def denoising_residual(
      self, hidden, previous_step_kv, sample_mode, has_training_mask,
      detach_cache_backbone=False, source_mask=None):
    """Apply denoising attention to active tokens, preserving other tokens."""
    if sample_mode:
      active = hidden[:, -self.block_size:]
      groups = active[:, None]
      output, next_step_kv = self.denoising_attn(
        groups,
        previous_step_kv=previous_step_kv,
        detach_cache_backbone=detach_cache_backbone,
        source_mask=source_mask)
      updated_active = active + output[:, 0]
      hidden = torch.cat((hidden[:, :-self.block_size], updated_active), dim=1)
      return hidden, next_step_kv

    if has_training_mask:
      # The first n tokens are x_t. Apply independent current-only denoising
      # attention to every BD3 block so the base loss trains the new sublayer.
      noisy = hidden[:, :self.n]
      groups = rearrange(
        noisy, 'b (g s) d -> b g s d', s=self.block_size)
      output, _ = self.denoising_attn(
        groups, previous_step_kv=None,
        detach_cache_backbone=detach_cache_backbone)
      noisy = noisy + rearrange(output, 'b g s d -> b (g s) d')
      hidden = torch.cat((noisy, hidden[:, self.n:]), dim=1)
      return hidden, None

    # Fallback for a single non-cross-attention sequence.
    groups = hidden[:, None]
    output, next_step_kv = self.denoising_attn(
      groups,
      previous_step_kv=previous_step_kv,
      detach_cache_backbone=detach_cache_backbone,
      source_mask=source_mask)
    return hidden + output[:, 0], next_step_kv

  def forward(self,
              x,
              rotary_cos_sin,
              c,
              causal=False,
              mask=None,
              sample_mode=False,
              store_kv=False,
              previous_step_kv=None,
              return_step_kv=False,
              detach_cache_backbone=False,
              step_memory_source_mask=None, key_valid=None,
              previous_key_valid=None):
    batch_size, seq_len = x.shape[0], x.shape[1]

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = None, None, None, None, None, None
    if c is not None and c.shape[0] == batch_size:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
      scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
    elif c is not None:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
      scale_mlp, gate_mlp) = rearrange(
        self.adaLN_modulation(c), '(b h) d -> b h d', b=batch_size
        ).chunk(6, dim=-1)

    if self.attention_mode == 'merged':
      if (causal or mask is not None or store_kv or self.kv_cache is not None
          or (seq_len != self.block_size and key_valid is None)
          or self.attn_backend != 'sdpa'):
        raise ValueError('Merged attention currently requires full-sequence noncausal SDPA without prefix caching')
      attended, entry_step_kv = self.merged_attention(
        x, previous_step_kv, detach_cache_backbone,
        step_memory_source_mask, shift_msa, scale_msa,
        key_valid=key_valid, previous_key_valid=previous_key_valid)
      x = self.attention_residual(attended, c, gate_msa, x)
      x = self.mlp_residual(x, c, gate_mlp, shift_mlp, scale_mlp)
      return (x, entry_step_kv) if return_step_kv else x

    # Diagonal recurrent memory is the first sublayer. At layer l it reads
    # M_{l+1} from the previous denoising forward, then ordinary BD3 attention
    # and the MLP complete H_l. The next layer's entry projection consequently
    # writes M_{l+1}=dc_qkv_{l+1}(H_l) without an extra per-layer writer.
    entry_step_kv = None
    if self.step_memory_enabled:
      x, entry_step_kv = self.denoising_residual(
        x,
        previous_step_kv=previous_step_kv,
        sample_mode=sample_mode,
        has_training_mask=mask is not None,
        detach_cache_backbone=detach_cache_backbone,
        source_mask=step_memory_source_mask)

    x_skip = x
    if c is not None:
      x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
    else:
      x = self.norm1(x)

    # get qkvs
    if mask is not None and not sample_mode:
      qkv_x = self.get_qkv(x[:,:self.n], rotary_cos_sin)
      qkv_x0 = self.get_qkv(x[:,self.n:], rotary_cos_sin)
      qkv = torch.cat((qkv_x, qkv_x0), dim=1)
    else:
      qkv = self.get_qkv(x, rotary_cos_sin, store_kv=store_kv)

    # attention
    if self.attn_backend == 'flash_attn' and mask is None:
      qkv = einops.rearrange(qkv, 'b s ... -> (b s) ...')
      cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len, step=seq_len,
        dtype=torch.int32, device=qkv.device)
      x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, seq_len, 0., causal=causal)
      x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)     
    elif self.attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
      x = self.cross_attn_flex(qkv, mask=mask)
    elif self.attn_backend == 'sdpa':
      sdpa_mask = mask
      if key_valid is not None:
        valid = key_valid[:, None, None, :].bool()
        sdpa_mask = valid if mask is None else mask.bool() & valid
      x = self.cross_attn(qkv, mask=sdpa_mask)
    else:
      raise ValueError('Unknown attention backend')
    if self.kv_cache is not None:
      x = x[:, -self.block_size:]
    x = self.attention_residual(x, c, gate_msa, x_skip)

    x = self.mlp_residual(x, c, gate_mlp, shift_mlp, scale_mlp)
    if return_step_kv:
      if not self.step_memory_enabled:
        raise ValueError('Cannot return denoising K/V while step memory is disabled')
      return x, entry_step_kv
    return x
   
class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x):
    return self.embedding[x]


class DDiTFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim, 
               adaLN, tie_word_embeddings=False):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()
    self.adaLN = adaLN
    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim,
                                        2 * hidden_size,
                                        bias=True)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()
    self.tie_word_embeddings = tie_word_embeddings

  def forward(self, x, c):
    x = self.norm_final(x)
    if c is not None:
      if c.shape[0] == x.shape[0]:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
      else:
        shift, scale = rearrange(
          self.adaLN_modulation(c), '(b h) d -> b h d', b=x.shape[0]).chunk(2, dim=-1)
      x = modulate_fused(x, shift, scale)
    x = self.linear(x)
    return x


class DenoisingCacheFinalWriter(nn.Module):
  """Write M_{L+1} from the completed final-layer hidden state H_L."""
  def __init__(self, hidden_size, n_heads):
    super().__init__()
    self.n_heads = n_heads
    self.norm = LayerNorm(hidden_size)
    self.kv = nn.Linear(hidden_size, 2 * hidden_size, bias=False)

  def forward(self, hidden, detach_backbone=False):
    if detach_backbone:
      hidden = hidden.detach()
    kv = self.kv(self.norm(hidden))
    return rearrange(
      kv, 'b s (two h d) -> b s two h d',
      two=2, h=self.n_heads)


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)
    self.causal = getattr(config.model, 'causal_attention', config.algo.parameterization == 'ar')
    self.n = config.model.length
    self.no_time_conditioning = bool(getattr(config.model, 'no_time_conditioning', False))
    self.adaLN = (not self.no_time_conditioning and
                  (not self.causal or getattr(config.model, 'adaln', False)))
    self.config = config
    self.vocab_size = vocab_size
    self.block_size = config.block_size
    dim = config.model.hidden_size
    cond_dim = config.model.cond_dim
    self.n_heads = config.model.n_heads
    self.vocab_embed = EmbeddingLayer(dim, vocab_size)
    if self.adaLN == True:
      self.sigma_map = TimestepEmbedder(cond_dim)
    if not self.causal and not self.no_time_conditioning:
      self.sigma_map = TimestepEmbedder(cond_dim)
    self.rotary_emb = Rotary(dim // config.model.n_heads)
    self.attn_backend = getattr(config.model, 'attn_backend', 'flash_attn')
    self.max_seqlen = 1024
    step_memory_config = getattr(config, 'step_memory', {})
    step_memory_enabled = bool(
      getattr(step_memory_config, 'enabled', False))
    attention_mode = getattr(step_memory_config, 'attention_mode', 'separate')
    merged_policy = getattr(step_memory_config, 'merged_policy', 'legacy')
    current_only_merged = bool(getattr(step_memory_config, 'current_only_merged', False))
    if attention_mode not in {'separate', 'merged'}:
      raise ValueError('Unknown step-memory attention_mode')
    if attention_mode == 'merged' and (
        (not step_memory_enabled and not current_only_merged) or self.causal or config.algo.cross_attn
        or self.block_size != self.n or self.attn_backend != 'sdpa'
        or config.sampling.kv_cache):
      raise ValueError('Merged attention requires full-sequence MDLM, enabled step memory, SDPA and no prefix cache')
    dc_spatial_rope_dim = int(getattr(
      step_memory_config, 'spatial_rope_dim',
      (dim // self.n_heads) * 3 // 4))
    dc_temporal_rope_dim = int(getattr(
      step_memory_config, 'temporal_rope_dim',
      (dim // self.n_heads) - dc_spatial_rope_dim))
    head_dim = dim // self.n_heads
    if dc_spatial_rope_dim + dc_temporal_rope_dim != head_dim:
      # Preserve the intended 3:1 split for model variants whose head size is
      # not 64 (for example the repository's tiny test model).
      dc_temporal_rope_dim = max(2, head_dim // 4)
      dc_temporal_rope_dim -= dc_temporal_rope_dim % 2
      dc_spatial_rope_dim = head_dim - dc_temporal_rope_dim
    gate_config = getattr(step_memory_config, 'gate', {})
    step_memory_gate_enabled = bool(getattr(gate_config, 'enabled', False))
    step_memory_gate_init = float(getattr(gate_config, 'init', 0.1))
    if merged_policy not in {'legacy', 'current_preserving'}:
      raise ValueError('Unknown step-memory merged_policy')
    if merged_policy == 'current_preserving':
      if attention_mode != 'merged' or step_memory_gate_enabled:
        raise ValueError(
          'current_preserving requires merged attention with the previous-V gate disabled')
      source_config = getattr(
        getattr(step_memory_config, 'pretrain', {}), 'source_dropout', {})
      if (bool(getattr(source_config, 'enabled', False))
          and float(getattr(source_config, 'cache_only_probability', 0.0)) != 0.0):
        raise ValueError('current_preserving requires cache_only_probability=0')
    dcachehooping_config = getattr(config, 'dcachehooping', {})
    self.dcachehooping_enabled = bool(getattr(
      dcachehooping_config, 'enabled', False))
    two_forward_config = getattr(dcachehooping_config, 'two_forward', {})
    self.dcachehooping_two_forward_enabled = bool(getattr(
      two_forward_config, 'enabled', False))
    status_config = getattr(dcachehooping_config, 'status_embedding', {})
    confidence_config = getattr(dcachehooping_config, 'confidence', {})
    self.dcachehooping_status_enabled = bool(getattr(
      status_config, 'enabled', True))
    self.dcachehooping_confidence_enabled = bool(getattr(
      confidence_config, 'enabled', True))
    self.dcachehooping_latent_norm = None
    self.dcachehooping_status_embed = None
    self.dcachehooping_confidence_head = None
    if self.dcachehooping_enabled:
      # A zero scale makes an adapted DCache-v2 checkpoint logit-identical
      # before the new recurrent latent path has learned to contribute.
      if self.dcachehooping_two_forward_enabled:
        # Match Loopholing exactly for the new two-forward mode: the same
        # affine LayerNorm processes an explicit zero latent on the first
        # pass and the previous final hidden on the second pass. Both affine
        # parameters start at zero, so enabling the path is initially neutral.
        self.dcachehooping_latent_norm = nn.LayerNorm(dim)
        self.dcachehooping_latent_norm.weight.data.zero_()
        self.dcachehooping_latent_norm.bias.data.zero_()
      else:
        # Preserve the parameterization and checkpoint schema of every legacy
        # DCachehooping/final-state run.
        self.dcachehooping_latent_norm = LayerNorm(dim)
        self.dcachehooping_latent_norm.weight.data.zero_()
      if self.dcachehooping_status_enabled:
        self.dcachehooping_status_embed = nn.Embedding(3, dim)
        self.dcachehooping_status_embed.weight.data.zero_()
      if self.dcachehooping_confidence_enabled:
        self.dcachehooping_confidence_head = nn.Linear(dim, 1)
        self.dcachehooping_confidence_head.weight.data.zero_()
        self.dcachehooping_confidence_head.bias.data.zero_()

    blocks = []
    for _ in range(config.model.n_blocks):
      if self.causal:
        block = DDiTBlockCausal(
          n=config.model.length,
          dim=dim,
          n_heads=config.model.n_heads,
          dropout=config.model.dropout,
          max_batch_size=config.loader.eval_batch_size,
          adaLN=self.adaLN,
          cond_dim=cond_dim,
          attn_backend=self.attn_backend)
      else:
        block = DDiTBlock(
          n=config.model.length,
          dim=dim,
          n_heads=config.model.n_heads,
          cond_dim=cond_dim,
          adaLN=self.adaLN,
          dropout=config.model.dropout,
          block_size=self.block_size,
          attn_backend=self.attn_backend,
          step_memory_enabled=step_memory_enabled,
          dc_spatial_rope_dim=dc_spatial_rope_dim,
          dc_temporal_rope_dim=dc_temporal_rope_dim,
          step_memory_gate_enabled=step_memory_gate_enabled,
          step_memory_gate_init=step_memory_gate_init,
          attention_mode=attention_mode,
          merged_policy=merged_policy,
          current_only_merged=current_only_merged,
          max_seqlen=self.max_seqlen)
      blocks.append(block)
    self.blocks = nn.ModuleList(blocks)
    self.step_memory_enabled = step_memory_enabled
    self.dc_final_writer = None
    if self.step_memory_enabled and not self.causal:
      self.dc_final_writer = DenoisingCacheFinalWriter(
        hidden_size=dim, n_heads=self.n_heads)
    self.output_layer = DDiTFinalLayer(
      hidden_size=dim,
      out_channels=vocab_size,
      cond_dim=cond_dim,
      adaLN=self.adaLN,
      tie_word_embeddings=config.model.tie_word_embeddings)
    if config.algo.cross_attn:
      self.gen_mask(config.model.length, self.block_size, self.attn_backend)
    self.neighbor_heads = None
    neighbor_config = getattr(config, 'neighbor_prediction', {})
    if bool(getattr(neighbor_config, 'enabled', False)):
      if attention_mode != 'merged':
        raise ValueError('Neighbor prediction currently requires merged attention')
      from neighbor_prediction import NeighborPredictionHeads
      # Adding auxiliary heads must not shift initialization/training RNG for
      # shared parameters or the trajectory. Disabled adds no state_dict keys.
      with torch.random.fork_rng(devices=[]):
        self.neighbor_heads = NeighborPredictionHeads(dim, vocab_size)

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference
    
  def gen_mask(self, seqlen, block_size, attn_backend='sdpa'):
    """Genererates attention mask"""
    if attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
      self.block_diff_mask = create_block_mask(
        partial(block_diff_mask, block_size=block_size, n=seqlen),
        B=None, H=None, Q_LEN=seqlen*2, KV_LEN=seqlen*2)
    elif attn_backend == 'sdpa':
      self.block_diff_mask = block_diff_mask(
        b=None, h=None, q_idx=torch.arange(seqlen*2)[:, None], 
        kv_idx=torch.arange(seqlen*2)[None, :], block_size=block_size, n=seqlen)
    else:
      raise ValueError('Unknown attention backend')
    
  def reset_kv_cache(self, eval_batch_size=None):
    if eval_batch_size is None:
      eval_batch_size = self.config.loader.eval_batch_size
    parameter = next(self.parameters())
    for block in self.blocks:
      block.kv_cache = torch.zeros(
        eval_batch_size,
        self.max_seqlen,
        self.config.model.hidden_size * 3,
        device=parameter.device,
        dtype=parameter.dtype)
      block.cache_idx = 0

  def forward(self, indices, sigma, sample_mode=False, store_kv=False,
              previous_step_kv=None, return_step_kv=False,
              detach_cache_backbone=False,
              step_memory_source_mask=None,
              previous_final_hidden=None, token_status=None,
              return_dcachehooping=False,
              return_confidence_logits=False, return_hidden=False,
              attention_mask=None, previous_attention_mask=None):
    # Opt-in variable-length full-sequence SDPA path for symbolic reasoning.
    # Existing OWT callers pass None and keep the original attention exactly.
    if attention_mask is not None:
      if (self.attn_backend != 'sdpa' or self.causal or self.config.algo.cross_attn
          or store_kv or self.config.sampling.kv_cache):
        raise ValueError('Padding masks currently require full-sequence noncausal SDPA')
      if self.step_memory_enabled and self.blocks[0].attention_mode != 'merged':
        raise ValueError('Padding masks with recurrent memory require merged attention')
      if attention_mask.shape != indices.shape or not attention_mask.bool().any(-1).all():
        raise ValueError('attention_mask must match tokens and contain a valid key per example')
      if previous_attention_mask is not None and previous_attention_mask.shape != indices.shape:
        raise ValueError('Previous attention mask must match current token shape')
    x = self.vocab_embed(indices)
    if (self.dcachehooping_two_forward_enabled
        and previous_final_hidden is None):
      # The public caller uses None for "no previous state"; Loopholing turns
      # that sentinel into an explicit zero latent inside the backbone.
      previous_final_hidden = torch.zeros_like(x)
    if previous_final_hidden is not None:
      if not self.dcachehooping_enabled:
        raise ValueError(
          'Previous final hidden requires dcachehooping.enabled=true')
      if previous_final_hidden.shape != x.shape:
        raise ValueError(
          'Previous final hidden must match token embedding shape')
      x = x + self.dcachehooping_latent_norm(previous_final_hidden)
    if token_status is not None:
      if not self.dcachehooping_enabled:
        raise ValueError('Token status requires dcachehooping.enabled=true')
      if self.dcachehooping_status_embed is None:
        raise ValueError(
          'Token status requires dcachehooping.status_embedding.enabled=true')
      if token_status.shape != indices.shape:
        raise ValueError('Token status must match input token shape')
      if ((token_status < 0) | (token_status > 2)).any():
        raise ValueError(
          'Token status values must be mask=0, committed=1, tentative=2')
      x = x + self.dcachehooping_status_embed(token_status.long())
    if return_dcachehooping and not self.dcachehooping_enabled:
      raise ValueError(
        'Rich workspace output requires dcachehooping.enabled=true')
    if return_confidence_logits:
      if not return_dcachehooping:
        raise ValueError(
          'Confidence output requires return_dcachehooping=true')
      if self.dcachehooping_confidence_head is None:
        raise ValueError(
          'Confidence output requires dcachehooping.confidence.enabled=true')
    if sigma is None:
      t_cond = None
    else:
      if self.no_time_conditioning:
        raise ValueError('This backbone was constructed without time conditioning')
      t_cond = F.silu(self.sigma_map(sigma))

    cross_attn = hasattr(self, 'block_diff_mask')
    if cross_attn:
      mask = self.block_diff_mask
      # special cases for sampling
      if sample_mode:
        if self.config.sampling.kv_cache:
          # full cross-attention to kv cache
          mask = None
          accum_length = self.blocks[0].cache_idx + self.block_size
          # positional encodings for cache
          x_full = torch.zeros((
            x.shape[0], accum_length, x.shape[2]), device=x.device)
          rotary_cos_sin = self.rotary_emb(x_full)
        else:
          # index block-causal mask only during sampling
          if self.attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
            mask = create_block_mask(
              partial(
                sample_block_causal_mask,
                block_size=self.block_size),
              B=None,
              H=None,
              Q_LEN=x.shape[1],
              KV_LEN=x.shape[1],
              device=x.device)
          else:
            mask = mask[
              self.n:self.n+x.shape[1], self.n:self.n+x.shape[1]]
          rotary_cos_sin = self.rotary_emb(x)

      else:
        rotary_cos_sin = self.rotary_emb(x[:, :self.n])

    else:
      rotary_cos_sin = self.rotary_emb(x)
      mask = None

    with (nullcontext() if getattr(self.config.model, 'external_autocast', False)
          else torch.amp.autocast('cuda', dtype=torch.bfloat16)):
      if return_step_kv and not self.step_memory_enabled:
        raise ValueError(
          'Cannot return denoising K/V while step memory is disabled')
      if previous_step_kv is not None:
        if not self.step_memory_enabled:
          raise ValueError(
            'Cannot consume denoising K/V while step memory is disabled')
        if len(previous_step_kv) != len(self.blocks):
          raise ValueError(
            'Previous denoising cache must contain one entry per reader layer')
      elif step_memory_source_mask is not None:
        raise ValueError('Source dropout requires a previous denoising cache')

      # The returned list is [M2, M3, ..., M_{L+1}]. Reader layer i consumes
      # entry i on the next denoising forward. M2...M_L are written by the
      # entry projections of blocks 2...L; only M_{L+1} needs a final writer.
      next_step_kv = [] if return_step_kv else None
      for i in range(len(self.blocks)):
        block_output = self.blocks[i](
          x,
          rotary_cos_sin,
          c=t_cond,
          causal=self.causal,
          sample_mode=sample_mode,
          mask=mask,
          store_kv=store_kv,
          previous_step_kv=(
            previous_step_kv[i] if previous_step_kv is not None else None),
          return_step_kv=return_step_kv,
          detach_cache_backbone=detach_cache_backbone,
          step_memory_source_mask=step_memory_source_mask,
          **({'key_valid': attention_mask,
              'previous_key_valid': previous_attention_mask}
             if attention_mask is not None else {}))
        if return_step_kv:
          x, entry_step_kv = block_output
          if i > 0:
            next_step_kv.append(entry_step_kv)
        else:
          x = block_output
      if return_step_kv:
        active_hidden = x[:, -self.block_size:]
        next_step_kv.append(self.dc_final_writer(
          active_hidden,
          detach_backbone=detach_cache_backbone))
        if len(next_step_kv) != len(self.blocks):
          raise RuntimeError('Shifted denoising cache mapping is incomplete')
        if attention_mask is not None:
          valid = attention_mask[:, :, None, None, None].bool()
          next_step_kv = [entry.masked_fill(~valid, 0) for entry in next_step_kv]
      final_hidden = x
      if attention_mask is not None:
        final_hidden = final_hidden.masked_fill(~attention_mask.bool()[:, :, None], 0)
      confidence_logits = None
      if return_confidence_logits:
        confidence_logits = self.dcachehooping_confidence_head(
          final_hidden).squeeze(-1)
      x = self.output_layer(final_hidden, t_cond)
    if cross_attn and not sample_mode:
      x = x[:, :self.n]
      final_hidden = final_hidden[:, :self.n]
      if confidence_logits is not None:
        confidence_logits = confidence_logits[:, :self.n]
    if return_dcachehooping or return_hidden:
      return DcachehoopingBackboneOutput(
        logits=x,
        step_kv=next_step_kv,
        final_hidden=final_hidden,
        confidence_logits=confidence_logits)
    if return_step_kv:
      return x, next_step_kv
    return x
