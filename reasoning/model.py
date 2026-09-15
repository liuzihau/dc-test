"""Small conditional MDM experiments, using the existing BD3 DIT blocks.

This is a controlled reproduction, not the unreleased Latent Tokens model.
The primary objective is *per-example conditional masked-answer CE*, averaged
over examples, then over states with normalized trajectory weights. It is not
an unconditional sequence NLL or a claimed diffusion ELBO. Prompts are never
corrupted. Neighbor heads predict clean masked targets, not joint token plans.
"""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from models.dit import DIT
from neighbor_prediction import neighbor_prediction_loss
from recurrent_gradients import AdjacentCacheGradients


class CacheState(list):
    """Layer caches plus their key validity; remains compatible with list APIs."""

    def __init__(self, entries=(), attention_mask=None):
        super().__init__(entries)
        self.attention_mask = attention_mask

    def index_select_batch(self, indices):
        return CacheState([entry.index_select(0, indices) for entry in self],
                          None if self.attention_mask is None else
                          self.attention_mask.index_select(0, indices))

    def detached(self):
        return CacheState([entry.detach() for entry in self], self.attention_mask)


DEFAULTS = dict(
    pad_id=0, mask_id=1, special_ids=(0, 1), hidden_size=512, n_heads=8,
    n_layers=6, max_length=384, memory_mode="none", attention_mode="merged",
    neighbors=False, neighbor_weight=0.5, gradient_mode="adjacent",
    trajectory="five", weights=(0.05, 0.1, 0.2, 1.0, 0.7),
    kmin=0.025, kmax=0.10, max_mask_ratio=0.9975,
    cache_only_probability=0.20, current_only_probability=0.05,
    source_dropout_warmup_steps=1000, final_dropout=0.1,
    identity_probability=0.25, identity_margin=0.05,
    identity_weight=0.1, identity_final_probability=0.5,
    gate_enabled=True, gate_init=0.1, dropout=0.0,
)


class ReasoningModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        unknown = set(config) - (set(DEFAULTS) | {"vocab_size", "merged_policy"})
        if unknown:
            raise ValueError("Unknown reasoning model config keys: " + ", ".join(sorted(unknown)))
        self.config = {**copy.deepcopy(DEFAULTS), **copy.deepcopy(dict(config))}
        c = self.config
        # Keep old default contracts exactly unchanged: a missing policy means
        # legacy, including when an old checkpoint is loaded for evaluation.
        merged_policy = c.pop("merged_policy", "legacy")
        if merged_policy not in {"legacy", "current_preserving"}:
            raise ValueError("merged_policy must be legacy or current_preserving")
        if merged_policy != "legacy":
            c["merged_policy"] = merged_policy
        if merged_policy == "current_preserving" and (
                c["attention_mode"] != "merged" or c["gate_enabled"]
                or c["cache_only_probability"] != 0):
            raise ValueError("current_preserving requires merged attention, no previous-V gate, "
                             "and cache_only_probability=0")
        self.mask_id, self.pad_id = int(c["mask_id"]), int(c["pad_id"])
        if (c["vocab_size"] < 2 or not 0 <= self.mask_id < c["vocab_size"]
                or not 0 <= self.pad_id < c["vocab_size"] or self.mask_id == self.pad_id):
            raise ValueError("Distinct in-vocabulary MASK and PAD IDs required")
        if c["n_layers"] < 1 or c["max_length"] < 1:
            raise ValueError("Positive n_layers and max_length required")
        self.has_dcache = c["memory_mode"] in {"dcache", "both"}
        self.has_final = c["memory_mode"] in {"final", "both"}
        if c["memory_mode"] not in {"none", "dcache", "final", "both"}:
            raise ValueError("memory_mode must be none, dcache, final, or both")
        if c["attention_mode"] not in {"merged", "vanilla"}:
            raise ValueError("attention_mode must be merged or vanilla")
        if self.has_dcache and c["attention_mode"] != "merged":
            raise ValueError("Reasoning DCache requires merged attention")
        if c["neighbors"] and c["attention_mode"] != "merged":
            raise ValueError("Neighbor-controlled trials require merged attention")
        if c["trajectory"] not in {"single", "five"}:
            raise ValueError("trajectory must be single or five")
        if c["gradient_mode"] not in {"adjacent", "detached"}:
            raise ValueError("gradient_mode must be adjacent or detached")
        if not 0 < c["kmin"] <= c["kmax"] < c["max_mask_ratio"] / 3 <= 1 / 3:
            raise ValueError("Invalid local trajectory interval")
        if len(c["weights"]) != 5 or min(c["weights"]) < 0 or sum(c["weights"]) <= 0:
            raise ValueError("Five nonnegative weights with positive sum required")
        for key in ("cache_only_probability", "current_only_probability", "final_dropout",
                    "identity_probability", "identity_final_probability"):
            if not 0 <= c[key] <= 1:
                raise ValueError(f"Invalid probability {key}")
        if c["cache_only_probability"] + c["current_only_probability"] > 1:
            raise ValueError("Source probabilities exceed one")
        if c["hidden_size"] % c["n_heads"]:
            raise ValueError("hidden_size must be divisible by n_heads")
        head_dim = c["hidden_size"] // c["n_heads"]
        if head_dim < 4 or head_dim % 2:
            raise ValueError("Even head dimension >=4 required")
        temporal = max(2, head_dim // 4)
        temporal -= temporal % 2
        dit_config = dict(
            model=dict(length=c["max_length"], hidden_size=c["hidden_size"],
                       cond_dim=c["hidden_size"], n_heads=c["n_heads"],
                       n_blocks=c["n_layers"], dropout=c["dropout"],
                       attn_backend="sdpa", tie_word_embeddings=False,
                       external_autocast=True, no_time_conditioning=True),
            algo=dict(parameterization="subs", cross_attn=False),
            sampling=dict(kv_cache=False), block_size=c["max_length"],
            loader=dict(eval_batch_size=1),
            step_memory=dict(enabled=self.has_dcache,
                             attention_mode="merged" if c["attention_mode"] == "merged" else "separate",
                             merged_policy=merged_policy,
                             current_only_merged=c["attention_mode"] == "merged",
                             spatial_rope_dim=head_dim-temporal, temporal_rope_dim=temporal,
                             gate=dict(enabled=c["gate_enabled"], init=c["gate_init"])),
            dcachehooping=dict(enabled=self.has_final,
                              status_embedding=dict(enabled=False),
                              confidence=dict(enabled=False)),
            neighbor_prediction=dict(enabled=c["neighbors"]),
        )
        self.backbone = DIT(dit_config, int(c["vocab_size"]))

    def forward(self, input_ids, attention_mask, previous_step_kv=None,
                previous_final_hidden=None, return_memory=True,
                detach_cache_backbone=False, source_mask=None):
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise ValueError("Expected matching [batch, sequence] input_ids/attention_mask")
        if input_ids.shape[1] > self.config["max_length"]:
            raise ValueError("Sequence exceeds configured max_length")
        if previous_step_kv is not None and len(previous_step_kv) == 0:
            previous_step_kv = None
        if previous_step_kv is not None and not self.has_dcache:
            raise ValueError("This control has no DCache")
        if previous_final_hidden is not None and not self.has_final:
            raise ValueError("This control has no final-state feedback")
        # Stop the final feedback independently of DCache gradient policy.
        final = None if previous_final_hidden is None else previous_final_hidden.detach()
        previous_valid = getattr(previous_step_kv, "attention_mask", None)
        out = self.backbone(
            input_ids, sigma=None, sample_mode=True,
            previous_step_kv=previous_step_kv,
            previous_final_hidden=final,
            return_step_kv=bool(return_memory and self.has_dcache),
            return_hidden=True, attention_mask=attention_mask,
            previous_attention_mask=previous_valid,
            detach_cache_backbone=detach_cache_backbone,
            step_memory_source_mask=source_mask)
        logits = out.logits.float()
        forbidden = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
        forbidden[self.mask_id] = True
        logits = logits.masked_fill(forbidden, -torch.inf)
        return dict(logits=logits, final_hidden=out.final_hidden,
                    step_kv=CacheState(out.step_kv or (), attention_mask.bool()))

    def sample_trajectory(self, batch, generator=None):
        """RNG-isolated corruption, sampled completely before model execution.

        Five-state masks are full -> centered t0..t3. With >=5 answer tokens,
        each transition reveals >=1 and retains >=1 mask. Shorter answers
        exhaust their available reveals and repeat the last one-mask state;
        they are never silently dropped or allowed to reveal every target.
        """
        clean = batch["input_ids"]
        eligible = batch["target_mask"].bool() & batch["attention_mask"].bool()
        counts = eligible.sum(-1)
        if not (counts > 0).all():
            raise ValueError("Every training example needs at least one answer target")
        if (clean[eligible] == self.mask_id).any():
            raise ValueError("Clean answer targets cannot contain MASK")
        batch_size = clean.shape[0]
        rand = lambda shape: torch.rand(shape, generator=generator, device=clean.device)
        if self.config["trajectory"] == "single":
            sampled = rand((batch_size, 1))
            k = torch.zeros((batch_size, 1), device=clean.device)
        else:
            k = self.config["kmin"] + rand((batch_size, 1)) * (self.config["kmax"] - self.config["kmin"])
            center = 1.5*k + rand((batch_size, 1)) * (self.config["max_mask_ratio"] - 3*k)
            sampled = torch.cat((torch.ones_like(k), center + k *
                                 clean.new_tensor([1.5, .5, -.5, -1.5], dtype=torch.float)), dim=1)
        masks = [torch.zeros_like(eligible) for _ in range(sampled.shape[1])]
        for b in range(batch_size):
            candidates = eligible[b].nonzero(as_tuple=False).flatten()
            n = candidates.numel()
            order = candidates[torch.randperm(n, generator=generator, device=clean.device)]
            previous = n + 1
            for index in range(sampled.shape[1]):
                count = max(1, min(n, int(torch.round(sampled[b, index]*n))))
                if self.config["trajectory"] == "five":
                    if index == 0:
                        count = n
                    elif n >= 5:
                        count = max(5-index, min(previous-1, count))
                    else:
                        count = max(1, min(previous-1, count))
                masks[index][b, order[:count]] = True
                previous = count
        states = [clean.masked_fill(mask, self.mask_id) for mask in masks]
        ratios = torch.stack([m.sum(-1) / counts for m in masks], dim=-1)
        return dict(states=states, masks=masks, ratios=ratios,
                    sampled_ratios=sampled, eligible=eligible, step_size=k)

    @staticmethod
    def _masked_ce(logits, clean, mask):
        # Gather only supervised positions. Fixed answer slots may include
        # EOS/PAD targets (preventing gold answer-length leakage); outer PAD
        # keys have attention_mask=false and are never supervised.
        losses = torch.zeros_like(clean, dtype=torch.float32)
        losses[mask] = F.cross_entropy(logits[mask], clean[mask], reduction="none")
        return losses.sum(-1) / mask.sum(-1).clamp_min(1)

    def compute_loss(self, batch, step=0, generator=None, training=True):
        c = self.config
        trajectory = self.sample_trajectory(batch, generator)
        clean, valid = batch["input_ids"], batch["attention_mask"].bool()
        # A fixed extra draw seeds a separate stream. Aux/memory choices never
        # change next batch's corruption stream or another control's inputs.
        local_seed = int(torch.randint(0, 2**31-1, (), generator=generator, device=clean.device))
        local = torch.Generator(device=clean.device).manual_seed(local_seed)
        rand = lambda shape=(): torch.rand(shape, generator=local, device=clean.device)
        adjacent = AdjacentCacheGradients(
            enabled=training and self.has_dcache and c["gradient_mode"] == "adjacent")
        detached = c["gradient_mode"] == "detached"
        weights = c["weights"] if c["trajectory"] == "five" else [1.0]
        names = ["full", "t0", "t1", "t2", "t3"] if len(weights) == 5 else ["single"]
        drop_final = training and self.has_final and bool(rand() < c["final_dropout"])
        previous_cache, previous_final = None, None
        losses, neighbors, metrics = [], [], {}
        last_input_cache, last_input_final, last_source = None, None, None
        for index, (state, mask, name) in enumerate(zip(
                trajectory["states"], trajectory["masks"], names)):
            source = None
            if training and previous_cache is not None and index >= 3:
                warmup = min(1.0, max(0, step) / max(1, c["source_dropout_warmup_steps"]))
                draw = rand(state.shape)
                source = torch.zeros_like(state)
                source[mask & (draw < c["cache_only_probability"]*warmup)] = 1
                source[mask & (draw >= c["cache_only_probability"]*warmup) &
                       (draw < c["cache_only_probability"]*warmup+c["current_only_probability"])] = 2
            supplied_final = None if drop_final else previous_final
            out = self(state, valid, previous_cache, supplied_final,
                       return_memory=index < len(weights)-1,
                       detach_cache_backbone=detached, source_mask=source)
            per_example = self._masked_ce(out["logits"], clean, mask)
            losses.append(per_example.mean())
            metrics["loss_"+name] = per_example.mean().detach()
            metrics["mask_ratio_"+name] = trajectory["ratios"][:, index].mean()
            metrics["accuracy_"+name] = out["logits"].detach().argmax(-1)[mask].eq(clean[mask]).float().mean()
            metrics["masked_tokens_"+name] = mask.sum().float()
            metrics["cache_only_fraction_"+name] = (source.eq(1).sum() / mask.sum()
                if source is not None else mask.new_zeros((), dtype=torch.float32))
            metrics["current_only_fraction_"+name] = (source.eq(2).sum() / mask.sum()
                if source is not None else mask.new_zeros((), dtype=torch.float32))
            content_mask = mask.clone()
            for special in c["special_ids"]:
                content_mask &= clean.ne(special)
            content_examples = content_mask.any(-1)
            content_nll = self._masked_ce(out["logits"], clean, content_mask)
            metrics["content_nll_"+name] = (content_nll[content_examples].mean().detach()
                                           if content_examples.any() else losses[-1].detach()*0)
            metrics["content_accuracy_"+name] = (out["logits"].detach().argmax(-1)[content_mask]
                .eq(clean[content_mask]).float().mean() if content_mask.any() else losses[-1].detach()*0)
            if c["neighbors"] and training:
                neighbor = neighbor_prediction_loss(
                    self.backbone.neighbor_heads, out["final_hidden"], clean, state,
                    valid, self.mask_id, excluded_token_ids=c["special_ids"],
                    checkpoint_chunks=False)
                neighbors.append(neighbor["loss"])
            last_input_cache, last_input_final, last_source = previous_cache, supplied_final, source
            previous_cache = (CacheState(adjacent.consume(out["step_kv"]), valid)
                              if self.has_dcache and index < len(weights)-1 else None)
            previous_final = out["final_hidden"].detach() if self.has_final else None
        base = sum(w*loss for w, loss in zip(weights, losses)) / sum(weights)
        zero = base.new_zeros(())
        neighbor_loss = sum(w*loss for w, loss in zip(weights, neighbors)) / sum(weights) if neighbors else zero
        identity, identity_final, identity_forwards = zero, False, 0
        if (training and len(weights) > 1 and clean.shape[0] > 1
                and (self.has_dcache or (self.has_final and not drop_final))
                and bool(rand() < c["identity_probability"])):
            shuffle = torch.arange(clean.shape[0], device=clean.device).roll(1)
            identity_final = self.has_final and not drop_final and (
                not self.has_dcache or bool(rand() < c["identity_final_probability"]))
            wrong_cache, wrong_final = last_input_cache, last_input_final
            if identity_final:
                wrong_final = last_input_final.index_select(0, shuffle)
            else:
                wrong_cache = last_input_cache.index_select_batch(shuffle)
            # Corrupted reference cannot learn to make itself deliberately bad.
            identity_forwards = 1
            with torch.no_grad():
                wrong = self(trajectory["states"][-1], valid, wrong_cache, wrong_final,
                             return_memory=False, source_mask=last_source)
            identity_mask = trajectory["masks"][-1]
            if last_source is not None and not identity_final:
                identity_mask = identity_mask & last_source.ne(2)
            eligible_examples = identity_mask.any(-1)
            if eligible_examples.any():
                correct_ce = self._masked_ce(out["logits"], clean, identity_mask)
                wrong_ce = self._masked_ce(wrong["logits"], clean, identity_mask)
                identity = F.relu(c["identity_margin"]+correct_ce-wrong_ce.detach())[eligible_examples].mean()
        total = base + c["neighbor_weight"]*neighbor_loss + c["identity_weight"]*identity
        if training:
            total = adjacent.attach(total)
        metrics.update(loss=total.detach(), objective=total.detach(), nll=losses[-1].detach(),
                       conditional_nll=losses[-1].detach(), base_loss=base.detach(),
                       content_only_nll=metrics["content_nll_"+names[-1]],
                       content_only_accuracy=metrics["content_accuracy_"+names[-1]],
                       neighbor_loss=neighbor_loss.detach(), identity_loss=identity.detach(),
                       identity_final=zero+float(identity_final),
                       final_dropout=zero+float(drop_final),
                       adjacent_edges=zero+adjacent.num_edges,
                       loss_weight_sum=zero+sum(weights),
                       trajectory_forwards=zero+len(weights),
                       identity_forwards=zero+identity_forwards,
                       num_forwards=zero+len(weights)+identity_forwards)
        return total, metrics
