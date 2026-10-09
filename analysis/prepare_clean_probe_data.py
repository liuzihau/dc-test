"""Disjoint fit/development documents; preserve the existing100-document test set."""
import json
from pathlib import Path
import numpy as np
from owt.continuation import ROOT,digest
from owt.research import atomic_write,timestamp


def prepare(output,train_count=200,development_count=50,test_count=100):
    import hashlib
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    prior=ROOT/'outputs/analysis/owt-local-denoising-20261006'
    receipt=json.loads((prior/'data_receipt.json').read_text())
    if digest(prior/'inputs.npz')!=receipt['inputs_sha256']:raise ValueError('Evaluation data changed')
    data=np.load(prior/'inputs.npz',allow_pickle=False)
    test=data['clean'][:test_count];test_ids=data['document_ids'][:test_count]
    if len(test)!=test_count:raise ValueError('Evaluation count is unavailable')
    shard=ROOT/'outputs/datasets/owt-local-reference/train-00079-of-00080.parquet'
    if digest(shard)!=receipt['shard_sha256']:raise ValueError('Raw data shard changed')
    table=pq.read_table(shard,columns=['text'])
    tokenizer=AutoTokenizer.from_pretrained('gpt2',cache_dir=str(ROOT/'.cache/huggingface/hub'),local_files_only=True)
    exclude_ids=set(data['document_ids'].tolist())
    exclude_hash=set(receipt['raw_text_sha256'])
    prefixes={hashlib.sha256(row.tobytes()).hexdigest() for row in data['clean']}
    order=np.random.default_rng(20261007).permutation(np.arange(max(0,len(table)-100000),len(table)))
    selected=[];ids=[];raw_hash=[]
    for index in order:
        document_id=receipt['total_documents']-len(table)+int(index)
        if document_id in exclude_ids:continue
        text=table['text'][int(index)].as_py();sha=hashlib.sha256(text.encode()).hexdigest()
        if sha in exclude_hash:continue
        tokens=tokenizer.encode(text,add_special_tokens=False)
        if len(tokens)<1022:continue
        row=np.asarray([50256,*tokens[:1022],50256],dtype=np.int64)
        prefix=hashlib.sha256(row.tobytes()).hexdigest()
        if prefix in prefixes or (row>=50257).any():continue
        prefixes.add(prefix);exclude_hash.add(sha)
        selected.append(row);ids.append(document_id);raw_hash.append(sha)
        if len(selected)==train_count+development_count:break
    if len(selected)!=train_count+development_count:raise ValueError('Insufficient independent documents')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'data.npz').exists():raise FileExistsError('Preserve previously selected probe documents')
    np.savez_compressed(output/'data.npz',train=np.stack(selected[:train_count]),
        development=np.stack(selected[train_count:]),evaluation=test,
        train_ids=np.asarray(ids[:train_count]),development_ids=np.asarray(ids[train_count:]),evaluation_ids=test_ids)
    record=dict(created_at=timestamp(),train_documents=train_count,development_documents=development_count,
        evaluation_documents=test_count,length=1024,selection_seed=20261007,
        train_ids=ids[:train_count],development_ids=ids[train_count:],evaluation_ids=test_ids.tolist(),
        additional_raw_sha256=raw_hash,raw_shard_sha256=receipt['shard_sha256'],
        evaluation_source_sha256=receipt['inputs_sha256'],data_sha256=digest(output/'data.npz'),
        evaluation_documents_excluded_from_fitting=True,duplicate_raw_text_and_prefixes_excluded=True,
        tokenizer='GPT2, original diagnostic token IDs',backbone_holdout_rule='final100000 OWT documents')
    atomic_write(output/'receipt.json',json.dumps(record,indent=2)+'\n')
    return record
