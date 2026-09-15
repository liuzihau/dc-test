"""Task evaluation, isolated from OpenWebText evaluation.

Generation never reads answer lengths or answer tokens to construct its inputs:
``target_mask`` must designate task-fixed answer slots, including EOS/PAD slots.
Top-prob is the positional candidate-k rule of He et al., Appendix E.2, not
nucleus sampling. See https://arxiv.org/html/2602.03769v1#A5.SS2 .
"""

from __future__ import annotations

import hashlib
import math
import time
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F


MEMORY_CONDITIONS = (
    "correct", "none", "shuffle_dcache", "shuffle_final", "shuffle_both"
)


def _generator(seed: int, record_index: int, stream: str) -> torch.Generator:
    # Stable across Python processes, batching, and models; no Python hash().
    payload = f"{seed}:{record_index}:{stream}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return torch.Generator(device="cpu").manual_seed(value % (2**63 - 1))


def _components(dataloader, tokenizer, records):
    dataset = getattr(dataloader, "dataset", None)
    tokenizer = tokenizer if tokenizer is not None else getattr(dataset, "tokenizer", None)
    records = records if records is not None else getattr(dataset, "records", None)
    if tokenizer is None or records is None:
        raise ValueError("Provide tokenizer and records, or a dataset exposing both.")
    return tokenizer, records


def _batch(batch, device):
    ids = batch["input_ids"].to(device=device, dtype=torch.long)
    attention = batch["attention_mask"].to(device=device, dtype=torch.bool)
    target = batch["target_mask"].to(device=device, dtype=torch.bool)
    indices = torch.as_tensor(batch["record_index"]).reshape(-1).tolist()
    if ids.ndim != 2 or attention.shape != ids.shape or target.shape != ids.shape:
        raise ValueError("input_ids, attention_mask, and target_mask must share [B,L] shape.")
    if len(indices) != ids.shape[0] or bool((target & ~attention).any()):
        raise ValueError("Answer slots must be attention-visible and have one record index per row.")
    if bool((target.sum(-1) == 0).any()):
        raise ValueError("Every evaluated example must contain answer slots.")
    return ids, attention, target, [int(index) for index in indices]


def _check_condition(condition, batch_size):
    if condition not in MEMORY_CONDITIONS:
        raise ValueError(f"Unknown memory condition {condition!r}; use {MEMORY_CONDITIONS}.")
    if condition.startswith("shuffle_") and batch_size < 2:
        raise ValueError("Memory shuffling requires batch size >= 2, including the final batch; use drop_last or regroup it.")


def _index_cache(cache, indices):
    if cache is None:
        return None
    if hasattr(cache, "index_select_batch"):
        return cache.index_select_batch(indices)
    if torch.is_tensor(cache):
        return cache.index_select(0, indices.to(cache.device))
    if isinstance(cache, dict):
        return {key: _index_cache(value, indices) for key, value in cache.items()}
    if type(cache) in (list, tuple):
        return type(cache)(_index_cache(value, indices) for value in cache)
    raise TypeError("Cache containers with metadata must implement index_select_batch().")


def _memory_inputs(cache, final_hidden, condition, batch_size, device):
    if condition == "none":
        return None, None
    if not condition.startswith("shuffle_"):
        return cache, final_hidden
    # A deterministic cyclic derangement; every donor differs from its recipient.
    indices = torch.arange(batch_size, device=device).roll(1)
    if condition in ("shuffle_dcache", "shuffle_both"):
        cache = _index_cache(cache, indices)
    if condition in ("shuffle_final", "shuffle_both") and final_hidden is not None:
        final_hidden = final_hidden.index_select(0, indices.to(final_hidden.device))
    return cache, final_hidden


@contextmanager
def _evaluating(model):
    training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            yield
    finally:
        model.train(training)


def _synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _forward(model, ids, attention, cache, final_hidden, condition):
    cache, final_hidden = _memory_inputs(
        cache, final_hidden, condition, ids.shape[0], ids.device
    )
    result = model(
        ids, attention_mask=attention, previous_step_kv=cache,
        previous_final_hidden=final_hidden, return_memory=True,
        detach_cache_backbone=False, source_mask=None,
    )
    if result["logits"].shape[:2] != ids.shape:
        raise ValueError("Model logits must have shape [B,L,V].")
    return result


def _returned_memory(model, result):
    config = getattr(model, "config", {})
    mode = config.get("memory_mode", "both") if isinstance(config, dict) else getattr(config, "memory_mode", "both")
    cache = result.get("step_kv") if mode in ("dcache", "both") else None
    final_hidden = result.get("final_hidden") if mode in ("final", "both") else None
    return cache, final_hidden


def _probabilities(logits, mask_id):
    logits = logits.detach().float().cpu().clone()
    # PAD and EOS are learned target symbols. Do not forbid them or repair syntax.
    logits[..., mask_id] = -torch.inf
    probabilities = logits.softmax(-1)
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("Model produced invalid probability distributions.")
    return probabilities


def _answer_tokens(tokenizer, slot_ids):
    ids = [int(value) for value in slot_ids]
    has_eos = tokenizer.eos_id in ids
    prefix = ids[:ids.index(tokenizer.eos_id)] if has_eos else ids
    # Tokenizer decoding must preserve malformed specials before EOS.
    return tokenizer.decode(prefix), has_eos


def evaluate_generation(
    model, dataloader, device="cpu", *, tokenizer=None, records=None,
    policy="top_prob", candidate_k=8, seed=0, memory_condition="correct",
    tokens_per_step=1, max_steps=None, token_selection="paper", scorer=None,
):
    """Return ``(metrics, per_example_records)`` for closed-loop generation.

    ``paper`` samples vocabulary tokens except for Countdown, where it uses
    argmax (Appendix G.4). ``sample`` and ``argmax`` are explicit overrides.
    All fixed answer slots are decoded; only then is output truncated at the
    first predicted EOS. Out-of-order prediction of EOS cannot skip its prefix.
    ``max_steps`` is a per-batch forward-call budget, independent of gold length.
    """
    if policy not in ("uniform", "top_prob"):
        raise ValueError("policy must be 'uniform' or 'top_prob', not nucleus top-p.")
    if token_selection not in ("paper", "sample", "argmax"):
        raise ValueError("token_selection must be paper, sample, or argmax.")
    if candidate_k < 1 or tokens_per_step < 1 or (max_steps is not None and max_steps < 1):
        raise ValueError("candidate_k, tokens_per_step, and max_steps must be positive.")
    tokenizer, records = _components(dataloader, tokenizer, records)
    if scorer is None:
        from .tasks import score_prediction
        scorer = score_prediction
    details = []
    nfe = sample_nfe = tokens_processed = attention_tokens_processed = 0
    latency = 0.0
    with _evaluating(model):
        for batch in dataloader:
            gold, attention, target, record_indices = _batch(batch, device)
            _check_condition(memory_condition, len(record_indices))
            # Never inspect gold answer IDs or use their EOS/PAD positions here.
            current = gold.masked_fill(target, tokenizer.mask_id)
            slots = [torch.where(row)[0].cpu().tolist() for row in target]
            schedules, sampling_generators = [], []
            for index, positions in zip(record_indices, slots):
                order = torch.randperm(len(positions), generator=_generator(seed, index, "order"))
                schedules.append([positions[offset] for offset in order.tolist()])
                sampling_generators.append(_generator(seed, index, "tokens"))
            selected_orders = [[] for _ in record_indices]
            selections = [
                "argmax" if token_selection == "paper" and records[index]["task"] == "countdown"
                else "sample" if token_selection == "paper" else token_selection
                for index in record_indices
            ]
            # Memory belongs to this batch/trajectory, never the prior batch.
            cache = final_hidden = None
            batch_nfe = 0
            budget = max_steps or math.ceil(max(map(len, slots)) / tokens_per_step)
            _synchronize(device)
            started = time.perf_counter()
            while any(schedules) and batch_nfe < budget:
                result = _forward(model, current, attention, cache, final_hidden, memory_condition)
                probabilities = _probabilities(result["logits"], tokenizer.mask_id)
                for row, schedule in enumerate(schedules):
                    for _ in range(min(tokens_per_step, len(schedule))):
                        candidates = schedule[:candidate_k] if policy == "top_prob" else schedule[:1]
                        offset = int(probabilities[row, candidates].amax(-1).argmax()) if policy == "top_prob" else 0
                        position = candidates[offset]
                        distribution = probabilities[row, position]
                        token = int(distribution.argmax()) if selections[row] == "argmax" else int(
                            torch.multinomial(distribution, 1, generator=sampling_generators[row])
                        )
                        current[row, position] = token
                        schedule.remove(position)
                        selected_orders[row].append(position)
                cache, final_hidden = _returned_memory(model, result)
                batch_nfe += 1
            _synchronize(device)
            batch_latency = time.perf_counter() - started
            latency += batch_latency
            nfe += batch_nfe
            sample_nfe += batch_nfe * len(record_indices)
            tokens_processed += batch_nfe * current.numel()
            attention_tokens_processed += batch_nfe * int(attention.sum())
            for row, index in enumerate(record_indices):
                ids = current[row, slots[row]].cpu().tolist()
                tokens, has_eos = _answer_tokens(tokenizer, ids)
                raw_tokens = tokenizer.decode(ids)
                scores = scorer(records[index], raw_tokens)
                details.append({
                    "id": records[index]["id"], "record_index": index,
                    "task": records[index]["task"], "predicted_answer": tokens,
                    "predicted_answer_slots": raw_tokens,
                    "predicted_answer_ids": ids, "has_eos": has_eos,
                    "remaining_masked_slots": len(schedules[row]),
                    "all_slots_completed": not schedules[row],
                    "decode_order": selected_orders[row], "nfe": batch_nfe,
                    "token_selection": selections[row], "scores": scores,
                })
    count = len(details)
    numeric_keys = {key for item in details for key, value in item["scores"].items()
                    if isinstance(value, (bool, int, float))}
    task_metrics = {key: sum(float(item["scores"].get(key, 0)) for item in details) / count
                    for key in sorted(numeric_keys)} if count else {}
    metrics = {
        "evaluation": "closed_loop_generation", "num_examples": count,
        "policy": policy, "candidate_k": candidate_k if policy == "top_prob" else None,
        "token_selection": token_selection, "seed": seed,
        "memory_condition": memory_condition, "tokens_per_step": tokens_per_step,
        "max_steps": max_steps, "nfe": nfe, "sample_nfe": sample_nfe,
        "mean_nfe_per_example": sample_nfe / count if count else None,
        "latency_seconds": latency,
        "latency_seconds_per_example": latency / count if count else None,
        "tokens_processed": tokens_processed,
        "attention_visible_tokens_processed": attention_tokens_processed,
        "compute_unit": "dense input token positions; excludes extra cached KV attention, not FLOPs",
        "batch_completion_rate": sum(item["all_slots_completed"] for item in details) / count if count else None,
        **task_metrics,
    }
    return metrics, details


def evaluate_corruption(
    model, dataloader, device="cpu", *, tokenizer=None, records=None,
    ratios=(0.1, 0.3, 0.5, 0.7), seed=0, memory_condition="correct",
    reset_each_ratio=True,
):
    """Evaluate aligned fixed corruptions, returning metrics and row records.

    Default ``cold_independent`` resets memory at every ratio. With
    ``reset_each_ratio=False``, ``teacher_forced_nested`` processes descending
    mask ratios and carries memory: later states reveal additional gold tokens.
    Neither protocol reports these teacher-forced predictions as solve rates.
    Mask permutations are seeded by record index, shared across ratios/models.
    All-target metrics include EOS/PAD; content metrics exclude special IDs.
    """
    ratios = tuple(sorted(set(float(ratio) for ratio in ratios), reverse=True))
    if not ratios or any(not 0 < ratio <= 1 for ratio in ratios):
        raise ValueError("ratios must be nonempty and lie in (0, 1].")
    tokenizer, records = _components(dataloader, tokenizer, records)
    special_ids = getattr(tokenizer, "special_ids", ())
    if isinstance(special_ids, dict):
        special_ids = special_ids.values()
    special_ids = set(int(value) for value in special_ids)
    details = []
    accumulators = {ratio: {"nll_sum": 0.0, "correct": 0, "masked_tokens": 0,
                            "content_nll_sum": 0.0, "content_correct": 0, "content_tokens": 0}
                    for ratio in ratios}
    nfe = tokens_processed = 0
    latency = 0.0
    with _evaluating(model):
        for batch in dataloader:
            gold, attention, target, record_indices = _batch(batch, device)
            _check_condition(memory_condition, len(record_indices))
            orders = []
            for row, index in enumerate(record_indices):
                positions = torch.where(target[row])[0].cpu()
                permutation = torch.randperm(len(positions), generator=_generator(seed, index, "corruption"))
                orders.append(positions[permutation])
            cache = final_hidden = None
            for ratio in ratios:
                if reset_each_ratio:
                    cache = final_hidden = None
                masked = torch.zeros_like(target)
                for row, order in enumerate(orders):
                    number = max(1, math.ceil(len(order) * ratio))
                    masked[row, order[:number].to(gold.device)] = True
                current = gold.masked_fill(masked, tokenizer.mask_id)
                _synchronize(device)
                started = time.perf_counter()
                result = _forward(model, current, attention, cache, final_hidden, memory_condition)
                _synchronize(device)
                latency += time.perf_counter() - started
                logits = result["logits"].float().clone()
                logits[..., tokenizer.mask_id] = -torch.inf
                selected_logits = logits[masked]
                selected_targets = gold[masked]
                losses = F.cross_entropy(selected_logits, selected_targets, reduction="none")
                predictions = logits.argmax(-1)
                if not bool(torch.isfinite(losses).all()):
                    raise ValueError("Model produced non-finite conditional NLL.")
                loss_grid = torch.zeros_like(gold, dtype=torch.float)
                loss_grid[masked] = losses
                content = masked.clone()
                for token_id in special_ids:
                    content &= gold != token_id
                correct = predictions == gold
                aggregate = accumulators[ratio]
                aggregate["nll_sum"] += float(losses.sum())
                aggregate["correct"] += int((correct & masked).sum())
                aggregate["masked_tokens"] += int(masked.sum())
                aggregate["content_nll_sum"] += float(loss_grid[content].sum())
                aggregate["content_correct"] += int((correct & content).sum())
                aggregate["content_tokens"] += int(content.sum())
                for row, index in enumerate(record_indices):
                    total, content_total = int(masked[row].sum()), int(content[row].sum())
                    details.append({
                        "id": records[index]["id"], "record_index": index,
                        "task": records[index]["task"], "mask_ratio": ratio,
                        "masked_positions": torch.where(masked[row])[0].cpu().tolist(),
                        "masked_tokens": total,
                        "conditional_nll": float(loss_grid[row, masked[row]].sum()) / total,
                        "masked_token_accuracy": int((correct[row] & masked[row]).sum()) / total,
                        "content_tokens": content_total,
                        "content_conditional_nll": float(loss_grid[row, content[row]].sum()) / content_total if content_total else None,
                        "content_masked_token_accuracy": int((correct[row] & content[row]).sum()) / content_total if content_total else None,
                    })
                cache, final_hidden = _returned_memory(model, result)
                nfe += 1
                tokens_processed += gold.numel()
    per_ratio = {}
    for ratio, aggregate in accumulators.items():
        total, content_total = aggregate["masked_tokens"], aggregate["content_tokens"]
        per_ratio[str(ratio)] = {
            "masked_tokens": total,
            "conditional_nll": aggregate["nll_sum"] / total if total else None,
            "masked_token_accuracy": aggregate["correct"] / total if total else None,
            "content_tokens": content_total,
            "content_conditional_nll": aggregate["content_nll_sum"] / content_total if content_total else None,
            "content_masked_token_accuracy": aggregate["content_correct"] / content_total if content_total else None,
        }
    metrics = {
        "evaluation": "fixed_corruption",
        "protocol": "cold_independent" if reset_each_ratio else "teacher_forced_nested",
        "reset_each_ratio": reset_each_ratio, "memory_condition": memory_condition,
        "seed": seed, "ratios": per_ratio, "nfe": nfe,
        "num_examples": len(details) // len(ratios),
        "tokens_processed": tokens_processed, "latency_seconds": latency,
        "compute_unit": "dense input token positions; excludes extra cached KV attention, not FLOPs",
        "metric_note": "Conditional masked-token prediction, not closed-loop solve accuracy; content excludes special tokens.",
    }
    return metrics, details
