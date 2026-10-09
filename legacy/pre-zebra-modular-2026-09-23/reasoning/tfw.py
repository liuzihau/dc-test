"""Paper-informed GPT-2 MDM control, isolated from every historical DiT run.

Training/alignment and remasking adapted from HKUNLP/diffusion-vs-ar,
commit 6743981a4ba42062c95279e590f3991de3985581 (Apache-2.0).
See legacy/upstream/diffusion-vs-ar/LICENSE. This is NOT an author-verified
reproduction: the ICML Zebra configuration and processed splits are unreleased.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel

UPSTREAM_COMMIT = '6743981a4ba42062c95279e590f3991de3985581'


class PaperMDM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        if config['family'] != 'tfw_gpt2_v1':
            raise ValueError('Not a TFW GPT-2 checkpoint')
        # Missing key preserves existing shifted checkpoints/configs exactly.
        self.logit_shift = config.get('logit_shift', 1)
        if type(self.logit_shift) is not int or self.logit_shift not in (0, 1):
            raise ValueError('logit_shift must be 0 (same position) or 1 (upstream)')
        self.mask_id, self.pad_id = config['mask_id'], config['pad_id']
        self.target_region = config.get('target_region', 'padded_tail')
        self.padding_attention = config.get('padding_attention', 'visible')
        if self.target_region not in ('padded_tail', 'answer'):
            raise ValueError('Invalid target_region')
        if self.padding_attention not in ('visible', 'masked'):
            raise ValueError('Invalid padding_attention')
        if self.padding_attention == 'masked' and self.target_region != 'answer':
            raise ValueError('Masked padding attention requires answer-only targets')
        gpt = GPT2Config(
            vocab_size=config['vocab_size'], n_positions=config['n_positions'],
            n_ctx=config['n_positions'], n_embd=config['hidden_size'],
            n_layer=config['n_layers'], n_head=config['n_heads'],
            activation_function='gelu_new', resid_pdrop=config['dropout'],
            embd_pdrop=config['dropout'], attn_pdrop=config['dropout'],
            layer_norm_epsilon=1e-5, initializer_range=0.02,
            bos_token_id=config['bos_id'], eos_token_id=config['eos_id'],
            pad_token_id=self.pad_id, use_cache=False, tie_word_embeddings=True,
            attn_implementation='eager')
        self.gpt = GPT2LMHeadModel(gpt)
        # Eager is mandatory: merely filling bias does NOT disable causal SDPA.
        if self.gpt.config._attn_implementation != 'eager':
            raise ValueError('Bidirectional GPT-2 requires explicit eager attention')
        for block in self.gpt.transformer.h:
            block.attn.bias.fill_(True)
        self.zebra_encoding = config.get('zebra_encoding', 'absolute')
        if self.zebra_encoding not in ('absolute', 'answer_relative', 'typed_coordinates'):
            raise ValueError('Invalid Zebra encoding')
        if self.zebra_encoding != 'absolute':
            if self.target_region != 'answer' or self.padding_attention != 'masked' or self.logit_shift:
                raise ValueError('Zebra encoding requires repaired no-shift answer-only mode')
            if config['n_positions'] < 421:
                raise ValueError('Zebra encoding requires at least 421 position embeddings')
        if self.zebra_encoding == 'typed_coordinates':
            from .zebra_encoding import TypedCoordinates
            self.coordinates = TypedCoordinates(config['hidden_size'])

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if any(kwargs.get(k) is not None for k in ('previous_step_kv', 'previous_final_hidden')):
            raise ValueError('TFW baseline has no recurrent memory')
        # Preserve historical checkpoints exactly. The repaired mode requires
        # the public layout mask, NEVER infers the boundary from token values.
        if self.padding_attention == 'masked':
            if attention_mask is None or attention_mask.shape != input_ids.shape:
                raise ValueError('Answer-only masked attention needs the public attention_mask')
        else:
            attention_mask = torch.ones_like(input_ids)
        embedding_args = dict(input_ids=input_ids)
        if self.zebra_encoding != 'absolute':
            from .zebra_encoding import public_features
            features = public_features(input_ids, attention_mask, **self.config['zebra_public_tokens'])
            embedding_args['position_ids'] = features['position_ids']
            if self.zebra_encoding == 'typed_coordinates':
                embedding_args.pop('input_ids')
                embedding_args['inputs_embeds'] = self.gpt.transformer.wte(input_ids) + self.coordinates(features)
        hidden = self.gpt.transformer(
            **embedding_args, attention_mask=attention_mask,
            use_cache=False, return_dict=True).last_hidden_state
        raw = self.gpt.lm_head(hidden)
        logits = torch.cat((raw[:, :1], raw[:, :-1]), dim=1) if self.logit_shift else raw
        return dict(logits=logits, final_hidden=None, step_kv=None)


def tail_mask(batch):
    """All positions after the public source/SEP, including padded output slots."""
    target = batch['target_mask'].bool()
    if not bool(target.any(-1).all()):
        raise ValueError('Every example needs a public answer region')
    start = target.long().argmax(-1)
    return torch.arange(target.shape[1], device=target.device)[None, :] >= start[:, None]


def prediction_mask(batch, target_region='padded_tail'):
    if target_region == 'padded_tail':
        return tail_mask(batch)
    if target_region != 'answer':
        raise ValueError('Invalid target_region')
    # The data loader builds this from public houses*attributes + EOS, not
    # clean answer token IDs. Keep revealed clues and outer PAD untouched.
    target = batch['target_mask'].bool()
    if not bool(target.any(-1).all()):
        raise ValueError('Every example needs public answer slots')
    if bool((target & ~batch['attention_mask'].bool()).any()):
        raise ValueError('Answer targets must be attention-visible')
    return target


def corrupt(batch, mask_id, timesteps, generator, target_region='padded_tail'):
    if timesteps < 1:
        raise ValueError('Positive discrete timestep count required')
    clean = batch['input_ids']
    j = torch.randint(1, timesteps + 1, (len(clean),), device=clean.device, generator=generator)
    eligible = prediction_mask(batch, target_region)
    masked = (torch.rand(clean.shape, device=clean.device, generator=generator)
              < j[:, None] / timesteps) & eligible
    return clean.masked_fill(masked, mask_id), masked, j, eligible


def full_mask_mixture(inputs, masked, j, eligible, mask_id, generator, probability, timesteps=64):
    """Override a random fraction of examples with the fully masked answer.

    Draw this AFTER the ordinary corruption, so the remaining examples keep
    exactly the control's masks. Probability zero consumes no extra randomness.
    Callers retain the PRE-mixture loss denominator for a matched stepwise
    gradient scale. This deliberately changes the objective, not an unbiased
    importance-sampled estimate of the old objective. No clean tokens are read.
    """
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('full_mask_probability must be between 0 and 1')
    forced = torch.zeros(len(inputs), dtype=torch.bool, device=inputs.device)
    if probability:
        forced = torch.rand(len(inputs), device=inputs.device, generator=generator) < probability
        masked = torch.where(forced[:, None], eligible, masked)
        inputs = inputs.masked_fill(masked, mask_id)
        j = torch.where(forced, torch.full_like(j, timesteps), j)
    return inputs, masked, j, forced


def loss_terms(logits, clean, masked, j, grid_targets, pad_id=0):
    """Unnormalized sums, so accumulation divides ONCE by global masked count.

    Upstream DiffusionArguments defaults: no focal loss; time weight 1/(t+1).
    Raw objective and unweighted answer-only CE must not be conflated.
    """
    ce = F.cross_entropy(logits.float().transpose(1, 2), clean, reduction='none')
    weighted_sum = (ce * masked / j[:, None]).sum()
    answer_mask = masked & grid_targets.bool()
    return weighted_sum, dict(
        weighted_sum=weighted_sum.detach(), masked_count=masked.sum(),
        ce_sum=(ce * masked).sum().detach(),
        answer_ce_sum=(ce * answer_mask).sum().detach(), answer_count=answer_mask.sum(),
        answer_correct=((logits.argmax(-1) == clean) & answer_mask).sum(),
        answer_predicted_pad=((logits.argmax(-1) == pad_id) & answer_mask).sum(),
        padding_count=(masked & ~grid_targets.bool()).sum())


def learning_rate(step, peak, total_steps):
    """HF default linear decay/no warmup, over the planned 300-epoch horizon."""
    return peak * max(0.0, 1.0 - step / total_steps)


def decode(model, batch, generator, *, steps=50, policy='upstream_remask', noise=0.5):
    """No gold answer token is used. Public source boundary defines the canvas.

    upstream_remask follows referenced stochastic0.5-linear: greedy token
    proposals, log-probability ranking, noise=.5*(t/T), and revisable output.
    paper_monotonic follows the paper's no-remask formulation: max-probability
    ranking + .5 Gumbel, floor(remaining/(t+1)) newly revealed tokens. Integer
    rounding/noise placement are reconstruction choices, not verified settings.
    """
    if steps < 1 or noise < 0 or policy not in ('upstream_remask', 'paper_monotonic'):
        raise ValueError('Invalid decoder settings')
    config = getattr(model, 'config', {})
    eligible = prediction_mask(batch, config.get('target_region', 'padded_tail'))
    current = batch['input_ids'].masked_fill(eligible, model.mask_id)
    counts = eligible.sum(-1)
    for t in range(steps - 1, -1, -1):
        kwargs = ({'attention_mask': batch['attention_mask']}
                  if config.get('padding_attention') == 'masked' else {})
        logits = model(current, **kwargs)['logits'].float()
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('Nonfinite generation logits')
        confidence, predicted = logits.log_softmax(-1).max(-1)
        u = torch.rand(confidence.shape, generator=generator, device=current.device)
        gumbel = -torch.log(-torch.log(u + 1e-8) + 1e-8)
        if policy == 'upstream_remask':
            proposed = torch.where(eligible, predicted, current)
            if t:
                ranking = confidence.masked_fill(~eligible, 1000) + noise * (t / steps) * gumbel
                cutoff = (counts * (t / steps)).long()
                threshold = ranking.sort(-1).values.gather(1, cutoff[:, None])
                remask = (ranking < threshold) & eligible
                current = proposed.masked_fill(remask, model.mask_id)
            else:
                current = proposed
        else:
            remaining = eligible & current.eq(model.mask_id)
            # Do not permit MASK as a committed prediction in this policy.
            logits[..., model.mask_id] = -torch.inf
            probability, predicted = logits.softmax(-1).max(-1)
            ranking = (probability + noise * gumbel).masked_fill(~remaining, -torch.inf)
            count = remaining.sum(-1) // (t + 1)
            order = ranking.argsort(-1, descending=True)
            chosen = torch.zeros_like(remaining).scatter(
                1, order, torch.arange(current.shape[1], device=current.device)[None, :] < count[:, None])
            current = torch.where(chosen & remaining, predicted, current)
    return current
