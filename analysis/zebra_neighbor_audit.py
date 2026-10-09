"""Read-only Zebra NP audit: target coverage, matched losses, and label routing.

Runs frozen CPU forwards; never attaches a trainer or saves model checkpoints.
Uses the vendored dense attention path, full sequence length, FP32, no dropout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / "third_party/reasoning_with_latent_tokens"
os.environ.setdefault("ESOLM_FORCE_NAIVE_ATTENTION", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/zebra-neighbor-audit-mpl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTHOR))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from zebra.runtime import install_no_cudagraph_compile
install_no_cudagraph_compile()
from zebra.model import ZebraMDM
from zebra.neighbor import neighbor_terms
from synthetic_data.zebra.data import ZebraTokenizer


def prefix_array(path, key, rows):
    """Read only a prefix of a compressed NPY member, avoiding the full train set."""
    with zipfile.ZipFile(path) as archive, archive.open(key + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
        if fortran or len(shape) != 2:
            raise ValueError("Expected a row-major two-dimensional array")
        count = min(rows, shape[0]) * shape[1]
        return np.frombuffer(stream.read(count * dtype.itemsize), dtype=dtype).reshape(-1, shape[1]).copy()


def pair_masks(x0, xt, boundaries, mask_id):
    content = ~torch.isin(x0, torch.tensor(boundaries))
    adjacency = content[:, :-1] & content[:, 1:]
    result = {}
    # Each mask is indexed by target, including zero padding at the edges.
    prev = torch.zeros_like(content)
    nxt = torch.zeros_like(content)
    prev[:, :-1] = adjacency & xt[:, :-1].eq(mask_id)
    nxt[:, 1:] = adjacency & xt[:, 1:].eq(mask_id)
    result[-1], result[1] = prev, nxt
    return content, result


def survey(x, boundaries):
    content = ~np.isin(x, boundaries)
    adjacent = content[:, :-1] & content[:, 1:]
    return dict(rows=len(x), sequence_length=x.shape[1],
                padding_fraction=float((x == 0).mean()),
                boundary_nonpadding_fraction=float(np.isin(x, [i for i in boundaries if i != 0]).mean()),
                content_fraction=float(content.mean()),
                directional_pair_fraction=float(adjacent.sum() / x.size),
                uniform_logits_expected_ratio=float(0.5 * adjacent.sum() / x.size))


def evaluate(model, x0, solution, t, uniforms, batch_size, stage):
    model.eval()
    total = {"main_sum": 0., "aux_sum": 0., "matched_main_sum": 0.,
             "weight_main": 0., "weight_pairs": 0., "tokens": x0.numel()}
    regions, directions, token_rows, source_rows = {}, {}, {}, {}
    assertions = dict(targets_masked=True, canonical_positions=True,
                      backbone_receives_no_clean_labels=True, exact_loss_decomposition=True,
                      masked_label_change_preserves_pair_selection=True,
                      masked_label_change_preserves_model_predictions=True)
    label_probe_done = False
    with torch.inference_mode():
        for start in range(0, len(x0), batch_size):
            clean = x0[start:start + batch_size]
            sol = solution[start:start + batch_size]
            ts = t[start:start + batch_size]
            dalpha, alpha = model.noise(ts)
            alpha = alpha[:, None]
            masked = uniforms[start:start + batch_size] < (1-alpha)
            xt = clean.masked_fill(masked, model.mask_index)
            w = torch.broadcast_to(-dalpha/(1-alpha), clean.shape)
            hidden = []
            inputs = []
            h1 = model.backbone.output_layer.linear.register_forward_pre_hook(
                lambda mod, inp: hidden.append(inp[0]))
            h2 = model.backbone.register_forward_pre_hook(
                lambda mod, inp, kwargs: inputs.append((inp, kwargs)), with_kwargs=True)
            identity = torch.arange(clean.shape[1]).expand_as(clean)
            lp = model.forward(xt, model._sigma_from_alphat(alpha), sort_idx=identity)
            h1.remove(); h2.remove()
            h = hidden.pop()
            args, kwargs = inputs.pop()
            assert torch.equal(args[0], xt)
            assert torch.equal(args[2], identity)
            assert len(args) >= 4 and args[3] is None and kwargs.get("x0") is None
            main_ce = -lp.gather(-1, clean[..., None]).squeeze(-1)
            main = main_ce*w*masked
            content, pairs = pair_masks(clean, xt, model.boundary_ids, model.mask_index)
            prod_terms, prod_counts = neighbor_terms(
                model.backbone.neighbor_heads, h, clean, xt,
                torch.ones_like(clean), torch.ones_like(clean), model.mask_index,
                model.boundary_ids, -dalpha/(1-alpha), torch.tensor(clean.numel()))
            main_prod = model.nll_per_token(lp, xt, clean, alpha, dalpha, train_mode=True)
            torch.testing.assert_close(main_prod, main)
            region_masks = {"padding": clean.eq(0), "boundary": ~content & clean.ne(0),
                            "answer": content & sol,
                            "problem": content & ~sol}
            total["main_sum"] += main.sum().item()
            total["weight_main"] += (w*masked).sum().item()
            for name, region in region_masks.items():
                row = regions.setdefault(name, dict(main_sum=0., aux_sum=0., matched_main_sum=0., weight_main=0., weight_pairs=0.))
                row["main_sum"] += main[region].sum().item()
                row["weight_main"] += (w*masked*region).sum().item()
            for tid in clean.unique().tolist():
                row = token_rows.setdefault(tid, dict(main_sum=0., aux_sum=0., matched_main_sum=0.))
                row["main_sum"] += main[clean.eq(tid)].sum().item()
            for offset, head in zip(model.np_config.offsets, model.backbone.neighbor_heads.heads):
                src = slice(1,None) if offset < 0 else slice(None,-1)
                tgt = slice(None,-1) if offset < 0 else slice(1,None)
                logits = head(h[:, src]).float()
                logits[..., model.mask_index] = -1000000.
                ce = F.cross_entropy(logits.transpose(1,2), clean[:,tgt], reduction="none")
                pair = pairs[offset][:,tgt]
                assert bool(xt[:,tgt][pair].eq(model.mask_index).all())
                aux = ce*w[:,tgt]*pair
                matched = main[:,tgt]*pair
                weights = w[:,tgt]*pair
                torch.testing.assert_close(aux.sum()/clean.numel(), prod_terms[offset])
                assert prod_counts[offset].item() == pair.sum().item()
                direction = directions.setdefault(offset, dict(main_sum=0., aux_sum=0., weight_pairs=0.))
                for key, value in [("main_sum", matched.sum()), ("aux_sum",aux.sum()), ("weight_pairs",weights.sum())]:
                    direction[key] += value.item()
                total["aux_sum"] += aux.sum().item()
                total["matched_main_sum"] += matched.sum().item()
                total["weight_pairs"] += weights.sum().item()
                for name, region in region_masks.items():
                    r = region[:,tgt]
                    regions[name]["aux_sum"] += aux[r].sum().item()
                    regions[name]["matched_main_sum"] += matched[r].sum().item()
                    regions[name]["weight_pairs"] += weights[r].sum().item()
                for source_state, condition in [("visible",~masked[:,src]),("masked",masked[:,src])]:
                    row = source_rows.setdefault(source_state, dict(aux_sum=0., matched_main_sum=0., weight_pairs=0.))
                    row["aux_sum"] += aux[condition].sum().item()
                    row["matched_main_sum"] += matched[condition].sum().item()
                    row["weight_pairs"] += weights[condition].sum().item()
                for tid in clean.unique().tolist():
                    r = clean[:,tgt].eq(tid)
                    token_rows[tid]["aux_sum"] += aux[r].sum().item()
                    token_rows[tid]["matched_main_sum"] += matched[r].sum().item()
            if not label_probe_done:
                altered = clean.clone()
                # Change only masked numeric labels; same xt, validity and boundaries.
                change = masked & clean.ge(4) & clean.le(9)
                altered[change] = 4 + (clean[change]-4+1) % 6
                _, new_pairs = pair_masks(altered, xt, model.boundary_ids, model.mask_index)
                assert all(torch.equal(new_pairs[o], pairs[o]) for o in pairs)
                new_lp = model.forward(xt, model._sigma_from_alphat(alpha), sort_idx=identity)
                torch.testing.assert_close(lp,new_lp,rtol=0,atol=0)
                assert int(change.sum()) > 0
                label_probe_done = True
    total["reported_ratio"] = .25*total["aux_sum"]/total["main_sum"]
    total["same_targets_main_ratio"] = .25*total["matched_main_sum"]/total["main_sum"]
    total["matched_aux_main_ce_ratio"] = total["aux_sum"]/total["matched_main_sum"]
    total["uniform_logits_realized_ratio"] = .25*total["weight_pairs"]/total["weight_main"]
    for d in (regions, directions, source_rows):
        for row in d.values():
            matched = row.get("matched_main_sum",row.get("main_sum",0.))
            row["matched_aux_main_ce_ratio"] = row["aux_sum"]/matched if matched else None
    total["main_elbo"] = total["main_sum"]/total["tokens"]
    total["gap_from_50pct_due_to_selection"] = .5-total["same_targets_main_ratio"]
    total["gap_due_to_actual_ce_difference"] = total["same_targets_main_ratio"]-total["reported_ratio"]
    excluded = regions["padding"]["main_sum"]+regions["boundary"]["main_sum"]
    total["gap_due_to_excluded_target_tokens"] = .5*excluded/total["main_sum"]
    total["gap_due_to_excluded_sources_for_content_targets"] = (
        total["gap_from_50pct_due_to_selection"]-total["gap_due_to_excluded_target_tokens"])
    print(stage, json.dumps(total), flush=True)
    return dict(stage=stage, totals=total, regions=regions, directions=directions,
                token_ids=token_rows, source_states=source_rows, assertions=assertions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples",type=int,default=256)
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs/analysis/zebra-neighbor-audit")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    run = ROOT/"outputs/zebra/mdm-np-40ep/mdm_np"
    config = OmegaConf.load(run/"resolved_config.yaml")
    config.mode = "completions"
    config.training.ema = 0
    tokenizer = ZebraTokenizer()
    torch.manual_seed(int(config.seed))
    # Unused text-generative-perplexity metrics otherwise fetch GPT-2's tokenizer.
    # They contain no model parameters and are never called by this audit.
    with patch("trainer_base.metrics.Metrics",return_value=torch.nn.Module()):
        model = ZebraMDM(config,tokenizer)
    cache = ROOT/".cache/author-zebra"
    train = prefix_array(cache/"zebra_zebra-train-data_ds1499933_sl384_vs23.npz","input_ids",4096)
    data = np.load(cache/"zebra_zebra-test-data_ds100000_sl384_vs23.npz")
    all_x = data["input_ids"]
    all_sol = data["loss_mask"]
    survey_result = {"train_first4096":survey(train,model.boundary_ids),"test_all100000":survey(all_x,model.boundary_ids)}
    print("SURVEY",json.dumps(survey_result),flush=True)
    rng = np.random.default_rng(20260930)
    indices = rng.choice(len(all_x),size=args.samples,replace=False)
    x = torch.from_numpy(all_x[indices].copy()).long()
    sol = torch.from_numpy(all_sol[indices].copy()).bool()
    gen = torch.Generator().manual_seed(20260930)
    t = .001+.999*((torch.arange(args.samples)+torch.rand(args.samples,generator=gen))/args.samples)
    uniforms = torch.rand(x.shape,generator=gen)
    del all_x,all_sol,data,train
    results=[]
    results.append(evaluate(model,x,sol,t,uniforms,args.batch_size,"initialization"))
    for checkpoint in ["2-8790.ckpt","78-172870.ckpt"]:
        payload = torch.load(run/"checkpoints"/checkpoint,map_location="cpu",weights_only=False)
        model.load_state_dict(payload["state_dict"],strict=True)
        del payload
        results.append(evaluate(model,x,sol,t,uniforms,args.batch_size,checkpoint))
    history=pd.read_csv(run/"local_metrics/train.csv")
    windows=[]
    for begin,end in [(1,1),(1,128),(2373,2500),(172743,172870)]:
        f=history[history.optimizer_step.between(begin,end)]
        windows.append(dict(begin=begin,end=end,main=float(f.main_elbo.mean()),
                            prev=float(f.np_prev.mean()),next=float(f.np_next.mean()),
                            ratio=float(.25*(f.np_prev.mean()+f.np_next.mean())/f.main_elbo.mean())))
    output=dict(samples=args.samples,seed=20260930,indices=indices.tolist(),
                inference="CPU FP32 dense full attention, dropout off, raw training weights (not EMA)",
                corruption="training regime: all 384 slots eligible, continuous stratified t, independent Bernoulli masks",
                boundaries=model.boundary_ids,vocab=tokenizer.get_vocab(),survey=survey_result,
                logged_windows=windows,probes=results)
    (args.output_dir/"audit.json").write_text(json.dumps(output,indent=2)+"\n")
    rows=[]
    for result in results:
        for name,row in result["regions"].items():
            rows.append(dict(stage=result["stage"],region=name,**row))
    pd.DataFrame(rows).to_csv(args.output_dir/"region_losses.csv",index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    labels=["Initial","Update 8,790","Update 172,870"]
    xx=np.arange(len(labels))
    for shift,key,label in [(-.22,"reported_ratio","Actual weighted NP / main"),(0,"same_targets_main_ratio","Main loss on NP targets / main × 0.25"),(.22,"uniform_logits_realized_ratio","Equal CE on all targets")]:
        axes[0].bar(xx+shift,[r["totals"][key] for r in results],width=.22,label=label)
    axes[0].axhline(.5,color="gray",ls="--",lw=1)
    axes[0].set_xticks(xx,labels);axes[0].set_ylabel("Loss ratio");axes[0].set_title("Target selection changes the reference ratio")
    axes[0].legend(fontsize=8)
    axes[1].bar(labels,[r["totals"]["matched_aux_main_ce_ratio"] for r in results],color="#0f8b76")
    axes[1].axhline(1,color="gray",ls="--",lw=1)
    axes[1].set_title("Auxiliary / main CE on identical masked targets")
    axes[1].set_ylabel("Ratio (1 = equally difficult for these heads)")
    fig.suptitle(f"Zebra NP audit · {args.samples} fixed held-out examples · training corruption")
    fig.tight_layout();fig.savefig(args.output_dir/"decomposition.png",dpi=180);plt.close(fig)
    print("Saved",args.output_dir,flush=True)


if __name__ == "__main__":
    main()
