#!/usr/bin/env python3
"""CPU-only numerical wiring check, not a trained-model performance benchmark.

Uses finite differences so stop-gradient cannot masquerade as absent numerical
dependence. Does not load checkpoints, train, or change model implementation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
LOCAL_TEMP = ROOT / ".cache/research-structure/tmp"
LOCAL_TEMP.mkdir(parents=True, exist_ok=True)
for name in ("TMPDIR", "TMP", "TEMP"):
    os.environ[name] = str(LOCAL_TEMP)
sys.path.insert(0, str(ROOT))

import torch
from models.dit import DIT


def check():
    torch.set_num_threads(1)
    torch.manual_seed(20260908)
    layers, tokens, width, heads = 4, 8, 8, 2
    config = NS(
        block_size=tokens,
        model=NS(causal_attention=False, length=tokens, hidden_size=width,
                 cond_dim=4, n_heads=heads, n_blocks=layers, dropout=0.0,
                 tie_word_embeddings=True, attn_backend="sdpa"),
        algo=NS(parameterization="subs", cross_attn=False),
        loader=NS(eval_batch_size=1), sampling=NS(kv_cache=False),
        step_memory=NS(enabled=True, spatial_rope_dim=2, temporal_rope_dim=2,
                       gate=NS(enabled=True, init=0.1)),
        dcachehooping=NS(enabled=True, two_forward=NS(enabled=False),
                        status_embedding=NS(enabled=False),
                        confidence=NS(enabled=False)))
    model = DIT(config, vocab_size=11).eval()
    # Nondegenerate random weights for a wiring test: do not confuse exact-zero
    # initialization gates/output weights with architectural independence.
    with torch.no_grad():
        for block in model.blocks:
            torch.nn.init.normal_(block.adaLN_modulation.weight, std=0.02)
            block.adaLN_modulation.bias.zero_()
            block.adaLN_modulation.bias[2*width:3*width].fill_(0.2)
            block.adaLN_modulation.bias[5*width:6*width].fill_(0.2)
        torch.nn.init.normal_(model.output_layer.linear.weight, std=0.02)
        model.dcachehooping_latent_norm.weight.fill_(0.5)
    x = torch.randint(0, 11, (1, tokens))
    sigma = torch.tensor([0.3])
    banks = [torch.randn(1, tokens, 2, heads, width // heads)
             for _ in range(layers)]
    directions = [torch.randn_like(bank) for bank in banks]
    directions = [direction / direction.norm() for direction in directions]
    final = torch.randn(1, tokens, width)
    epsilon = 1e-3

    def forward(memory, feedback, detach=False):
        return model(x, sigma, sample_mode=True, previous_step_kv=memory,
                     previous_final_hidden=feedback, return_step_kv=True,
                     return_dcachehooping=True, detach_cache_backbone=detach)

    def perturbed(index):
        plus = [bank.clone() for bank in banks]
        minus = [bank.clone() for bank in banks]
        plus[index] += epsilon * directions[index]
        minus[index] -= epsilon * directions[index]
        return plus, minus

    def sensitivity(plus, minus):
        return float(((plus - minus) / (2 * epsilon)).abs().max())

    report = {
        "scope": "random tiny real-backbone numerical wiring, not trained-model stability",
        "cpu_only": True, "seed": 20260908, "dtype": "float32",
        "torch_version": torch.__version__, "layers": layers, "tokens": tokens,
        "width": width, "heads": heads, "vocab": 11, "epsilon": epsilon,
        "initialization": {"adaLN_weight_std": 0.02, "msa_mlp_gate_bias": 0.2,
                           "dcache_effective_gate": 0.1, "final_norm_scale": 0.5,
                           "output_weight_std": 0.02},
        "config_tie_word_embeddings": True,
        "weights_are_tied": model.vocab_embed.embedding is model.output_layer.linear.weight,
        "model_source_sha256": hashlib.sha256((ROOT / "models/dit.py").read_bytes()).hexdigest(),
    }
    assert next(model.parameters()).device.type == "cpu"
    with torch.no_grad():
        for name, feedback in (("absent_final", None), ("fixed_final", final)):
            matrix = torch.zeros(layers, layers)
            final_row = []
            for column in range(layers):
                plus, minus = perturbed(column)
                p, m = forward(plus, feedback), forward(minus, feedback)
                for row in range(layers):
                    matrix[row, column] = sensitivity(p.step_kv[row], m.step_kv[row])
                final_row.append(sensitivity(p.final_hidden, m.final_hidden))
            a, b = forward(banks, feedback, False), forward(banks, feedback, True)
            assert torch.triu(matrix, diagonal=1).max() == 0
            assert bool(torch.all(matrix.diag() > 0))
            detach_difference = max(float((u-v).abs().max())
                                    for u, v in zip(a.step_kv, b.step_kv))
            assert detach_difference == 0
            report[name] = {
                "max_abs_directional_derivative_by_output_row_input_column": matrix.tolist(),
                "max_strict_upper": float(torch.triu(matrix, diagonal=1).max()),
                "diagonal": matrix.diag().tolist(), "last_hidden_sensitivity": final_row,
                "detach_flag_max_value_difference": detach_difference}
        final_direction = torch.randn_like(final)
        final_direction /= final_direction.norm()
        p = forward(banks, final + epsilon * final_direction)
        m = forward(banks, final - epsilon * final_direction)
        report["final_input_to_first_bank"] = sensitivity(p.step_kv[0], m.step_kv[0])
        plus, minus = perturbed(layers - 1)
        p1, m1 = forward(plus, final), forward(minus, final)
        p2 = forward(p1.step_kv, p1.final_hidden.detach())
        m2 = forward(m1.step_kv, m1.final_hidden.detach())
        joint = sensitivity(p2.step_kv[0], m2.step_kv[0])
        report["two_iterations_joint_feedback_last_bank_to_first_bank"] = joint
        p2, m2 = forward(p1.step_kv, final), forward(m1.step_kv, final)
        frozen = sensitivity(p2.step_kv[0], m2.step_kv[0])
        report["two_iterations_frozen_feedback_last_bank_to_first_bank"] = frozen
        assert joint > 0 and frozen == 0
    report["status"] = "passed"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional local JSON report path.")
    args = parser.parse_args()
    result = json.dumps(check(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    print(result, end="")
