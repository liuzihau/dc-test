"""Prepare document-level OWT validation inputs for local denoising analysis."""
import argparse
import hashlib
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_REVISION = '79d93d786212f7344586290adb811d4ae6a1762c'
SHARD = 'plain_text/train-00079-of-00080.parquet'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/analysis/owt-local-denoising-20261006')
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    cache = ROOT / 'outputs/datasets/owt-local-reference'
    cache.mkdir(parents=True, exist_ok=True)
    shard = cache / 'train-00079-of-00080.parquet'
    if not shard.exists():
        if not args.download:
            raise FileNotFoundError('Use --download to fetch the final OWT shard')
        import requests
        url = f'https://huggingface.co/datasets/openwebtext/resolve/{DATA_REVISION}/{SHARD}'
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            partial = shard.with_suffix('.partial')
            size = 0
            with partial.open('wb') as stream:
                for chunk in response.iter_content(8 * 1024 * 1024):
                    stream.write(chunk)
                    size += len(chunk)
                    if size // (64 * 1024 * 1024) != (size-len(chunk)) // (64 * 1024 * 1024):
                        print('OWT downloaded', size // (1024 * 1024), 'MiB', flush=True)
            if size != 302753464:
                raise ValueError('Pinned OWT shard size mismatch')
            partial.replace(shard)
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    table = pq.read_table(shard, columns=['text'])
    source_documents = table.num_rows
    tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir=str(ROOT / '.cache/huggingface/hub'),
                                              local_files_only=True)
    if tokenizer.vocab_size != 50257 or tokenizer.bos_token_id != 50256 or tokenizer.eos_token_id != 50256:
        raise ValueError('Expected GPT-2 token IDs')
    selected, document_ids, raw_digests, token_lengths = [], [], [], []
    # Arrow cache chunk lengths are not the source Parquet shard lengths.
    # Select only the final 100k documents, even if the final shard is larger.
    holdout_start = max(0, source_documents - 100000)
    order = np.random.default_rng(20261006).permutation(
        np.arange(holdout_start, source_documents))
    for index in order:
        text = table['text'][int(index)].as_py()
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 1022:
            continue
        selected.append([50256, *ids[:1022], 50256])
        document_ids.append(8013769 - source_documents + int(index))
        raw_digests.append(hashlib.sha256(text.encode()).hexdigest())
        token_lengths.append(len(ids))
        if len(selected) == args.samples:
            break
    if len(selected) != args.samples:
        raise ValueError('Insufficient length-eligible independent documents')
    if (args.output / 'inputs.npz').exists():
        raise FileExistsError('Refuse to replace selected diagnostic inputs')
    np.savez_compressed(args.output / 'inputs.npz', clean=np.asarray(selected, dtype=np.int64),
                        document_ids=np.asarray(document_ids, dtype=np.int64))
    from analysis.download_bd3_reference import sha256
    receipt = dict(dataset='openwebtext', revision=DATA_REVISION, shard=SHARD,
        shard_sha256=sha256(shard), split_rule='train[-100000:]',
        total_documents=8013769, source_shard_documents=source_documents,
        selection_seed=20261006, samples=args.samples, length=1024,
        sampling_population='documents with at least 1022 GPT-2 tokens; first 1022 content tokens',
        packing='one independent document per row; GPT-2 BOS + 1022 content + EOS',
        document_ids=document_ids, raw_text_sha256=raw_digests, document_token_lengths=token_lengths,
        tokenizer='gpt2', vocab_size=50257, bos_id=50256, eos_id=50256, mask_id=50257,
        inputs_sha256=sha256(args.output / 'inputs.npz'),
        pretrained_training_revision_recorded=False,
        split_note='Uses the official last-100k convention on the pinned OWT corpus; original checkpoint corpus revision is not recorded in its model config.',
        preprocessing_note='Document-level prefixes prevent cross-document windows; this differs from corpus-concatenated production validation.')
    (args.output / 'data_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('Prepared', args.samples, 'independent OWT held-out documents', flush=True)


if __name__ == '__main__':
    main()
