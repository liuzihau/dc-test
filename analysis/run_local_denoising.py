"""Frozen BD3 main-head diagnostic; GPU runs serialize with existing training."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types
import warnings

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'

import numpy as np
from analysis.download_bd3_reference import sha256
from analysis.local_denoising_metrics import (make_canvas, visibility_statistics,
                                             error_statistics, summarize, ERROR_FIELDS)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / 'outputs/analysis/owt-local-denoising-20261006'
LOCK = ROOT / 'outputs/owt/mdm-np-5k/.queue.lock'


def write_json(path, value):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def load_reference():
    """Load inspected, revision-pinned local code and safe tensor weights."""
    import torch
    from safetensors.torch import load_file
    receipt = json.loads((ROOT / 'outputs/pretrained/bd3lm-owt-block_size1024-pretrain/download_receipt.json').read_text())
    directory = Path(receipt['directory'])
    for name, record in receipt['files'].items():
        if sha256(directory / name) != record['sha256']:
            raise ValueError('Reference artifact changed: '+name)
    package_name = 'inspected_bd3_reference'
    package = types.ModuleType(package_name)
    package.__path__ = [str(directory)]
    sys.modules[package_name] = package
    modules = {}
    for name in ('configuration_bd3lm', 'modeling_bd3lm'):
        qualified = package_name+'.'+name
        spec = importlib.util.spec_from_file_location(qualified, directory / (name+'.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        modules[name] = module
    config = modules['configuration_bd3lm'].BD3LMConfig(**json.loads((directory / 'config.json').read_text()))
    if (config.block_size, config.model_length, config.vocab_size, config.cross_attn,
            config.causal, config.time_conditioning, config.attn_backend) != (1024, 1024, 50258, False, False, False, 'sdpa'):
        raise ValueError('Reference inference geometry differs from inspected checkpoint')
    model = modules['modeling_bd3lm'].BD3LM(config)
    model.load_state_dict(load_file(str(directory / 'model.safetensors')), strict=True, assign=True)
    model.float().eval()
    model.requires_grad_(False)
    receipt = dict(receipt, parameter_state='published Hugging Face export; no EMA swap',
                   time_conditioning=False, inference='one full-sequence noncausal SDPA pass; no sampling or KV cache')
    return model, receipt


def score_reference(model, canvas, clean, probability, device):
    import torch
    x = torch.as_tensor(canvas, dtype=torch.long, device=device)
    y = torch.as_tensor(clean, dtype=torch.long, device=device)
    sigma = torch.full((len(x),), -np.log1p(-probability), dtype=torch.float32, device=device)
    with torch.inference_mode(), warnings.catch_warnings():
        # Official code requests CUDA autocast with float32; torch disables it.
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', message='.*CUDA is not available.*')
        warnings.filterwarnings('ignore', message='.*target dtype is not supported.*')
        logits = model(x, timesteps=sigma, sample_mode=False, store_kv=False, return_dict=False)
        if logits.dtype != torch.float32 or logits.shape != (*x.shape, 50258):
            raise ValueError('Unexpected official logits dtype or shape')
        logits[..., 50257] = -torch.inf
        prediction = logits.argmax(-1)
        nll = torch.logsumexp(logits, -1) - logits.gather(-1, y[..., None]).squeeze(-1)
        if not torch.isfinite(nll).all():
            raise ValueError('Nonfinite reference NLL')
    return prediction.cpu().numpy(), nll.cpu().numpy()


def run(args, output):
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = np.load(args.data / 'inputs.npz', allow_pickle=False)
    clean = dataset['clean'][:args.samples]
    ids = dataset['document_ids'][:args.samples]
    if len(clean) != args.samples or clean.shape[1] != 1024 or len(np.unique(ids)) != len(ids):
        raise ValueError('Invalid independent document selection')
    data_receipt = json.loads((args.data / 'data_receipt.json').read_text())
    if sha256(args.data / 'inputs.npz') != data_receipt['inputs_sha256']:
        raise ValueError('Saved diagnostic inputs changed')
    model, reference = load_reference()
    model.to(args.device)
    def check_block(module, inputs, value):
        if inputs[0].dtype != torch.float32 or value.dtype != torch.float32:
            raise ValueError('Reference block entered lower precision')
    handles = [block.register_forward_hook(check_block) for block in model.backbone.blocks]
    script_files = [Path(__file__), ROOT/'analysis/local_denoising_metrics.py']
    protocol = dict(model='official_bd3_owt_full_block', reference=reference,
        data_receipt_sha256=sha256(args.data/'data_receipt.json'), inputs_sha256=data_receipt['inputs_sha256'],
        samples=args.samples, corruption_seeds=list(range(args.seeds)), mask_probability=args.mask_probability,
        document_ids=ids.tolist(), main_heads_only=True, precision='FP32; TF32 disabled',
        eligible='masked centers; all neighborhood positions content within same document/block',
        local_offsets=[-2,-1,1,2], error_offsets=[-1,1], error_fields=list(ERROR_FIELDS),
        mask_id=50257, special_ids=[50256], time_conditioning=False,
        device=args.device, microbatch=args.microbatch, torch_version=torch.__version__,
        numpy_version=np.__version__, cpu_threads=torch.get_num_threads(),
        script_sha256={str(p.relative_to(ROOT)):sha256(p) for p in script_files},
        full_requested_experiment=args.samples==100 and args.seeds==10 and args.mask_probability==.5)
    write_json(output/'protocol.json', protocol)
    started = time.monotonic()
    all_vis, all_error = [], []
    for seed in range(args.seeds):
        canvas, masked = make_canvas(clean, ids, seed, args.mask_probability)
        predictions, losses = [], []
        for start in range(0, len(clean), args.microbatch):
            end = min(start+args.microbatch, len(clean))
            prediction, nll = score_reference(model, canvas[start:end], clean[start:end], args.mask_probability, args.device)
            predictions.append(prediction); losses.append(nll)
            write_json(output/'progress.json', {'stage':'evaluating', 'seed':seed,
                'finished_documents_this_seed':end, 'total_document_forwards':seed*len(clean)+end,
                'requested_document_forwards':args.seeds*len(clean), 'elapsed_seconds':time.monotonic()-started})
        prediction, nll = np.concatenate(predictions), np.concatenate(losses)
        visibility = visibility_statistics(clean, masked, prediction, nll)
        errors = error_statistics(clean, masked, prediction)
        np.savez_compressed(output/f'seed-{seed:02d}.npz', document_ids=ids, clean=clean, canvas=canvas,
            masked=masked, prediction=prediction, nll=nll, visibility=visibility, errors=errors)
        all_vis.append(visibility); all_error.append(errors)
        partial_summary = summarize(np.stack(all_vis), np.stack(all_error), bootstraps=args.bootstraps)
        partial_summary.update(model=protocol['model'], samples=args.samples,
            completed_corruption_seeds=seed+1, requested_corruption_seeds=args.seeds,
            mask_probability=args.mask_probability, complete=False)
        write_json(output/'summary.json',partial_summary)
        from analysis.plot_local_denoising import refresh as plot_saved
        plot_saved(output)
        print('Finished corruption seed',seed,'documents',len(clean),'elapsed',round(time.monotonic()-started,1),flush=True)
    for handle in handles:
        handle.remove()
    summary = summarize(np.stack(all_vis), np.stack(all_error), bootstraps=args.bootstraps)
    summary.update(model=protocol['model'], samples=args.samples, corruption_seeds=args.seeds,
                   completed_corruption_seeds=args.seeds, requested_corruption_seeds=args.seeds, complete=True,
                   mask_probability=args.mask_probability, full_requested_experiment=protocol['full_requested_experiment'],
                   parameter_state=reference['parameter_state'], protocol_sha256=sha256(output/'protocol.json'),
                   source_revision=reference['revision'], elapsed_seconds=time.monotonic()-started,
                   dataset_scope=data_receipt['sampling_population'])
    write_json(output/'summary.json', summary)
    plot_saved(output)
    write_json(output/'progress.json', {'stage':'complete','document_forwards':args.seeds*len(clean),
        'elapsed_seconds':time.monotonic()-started})
    print('Completed',output,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--run-name', default='reference')
    parser.add_argument('--device', choices=['cpu','cuda:0'], default='cpu')
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--mask-probability', type=float, default=.5)
    parser.add_argument('--microbatch', type=int, default=1)
    parser.add_argument('--bootstraps', type=int, default=2000)
    parser.add_argument('--wait-for-gpu', action='store_true')
    args = parser.parse_args()
    if min(args.samples,args.seeds,args.microbatch,args.bootstraps)<1 or not 0<args.mask_probability<1:
        parser.error('Positive counts and mask probability in (0,1) required')
    if '/' in args.run_name or args.run_name in ('.','..'):
        parser.error('Run name must be a single directory name')
    if args.wait_for_gpu and args.device=='cpu':
        parser.error('--wait-for-gpu requires a CUDA device')
    output=args.data/args.run_name
    if output.exists():
        raise FileExistsError('Refuse to overwrite a previous/partial collection: '+str(output))
    output.mkdir(parents=True)
    write_json(output/'progress.json',{'stage':'starting','pid':os.getpid()})
    try:
        if args.device.startswith('cuda'):
            if os.environ.get('CUDA_VISIBLE_DEVICES')!='2':
                raise ValueError('Use only physical GPU2: CUDA_VISIBLE_DEVICES=2')
            with LOCK.open('a') as lock:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    if not args.wait_for_gpu:
                        raise RuntimeError('Existing training owns GPU lock; use --wait-for-gpu')
                    write_json(output/'progress.json',{'stage':'waiting_for_training_lock','pid':os.getpid()})
                    print('Waiting for existing training to release the GPU lock',flush=True)
                    fcntl.flock(lock,fcntl.LOCK_EX)
                run(args,output)
        else:
            run(args,output)
    except BaseException as error:
        write_json(output/'progress.json',{'stage':'failed','error':str(error),'pid':os.getpid()})
        raise


if __name__=='__main__':
    main()
