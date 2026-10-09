import copy
import json
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from reasoning.benchmark import (PROTOCOL, convert_sudoku, convert_zebra,
                                 import_sudoku, migrate_zebra)
from reasoning.data import ReasoningDataset, write_prepared_dataset
from reasoning.tasks import (TaskTokenizer, generate_record, record_answer_slots,
                             score_prediction, validate_record)
from test_reasoning_official_zebra import safe_raw
from test_reasoning_official_zebra import cpu_pipeline
from reasoning.zebra_official import convert_source_record


def sudoku_row(seed=1):
    r = generate_record('sudoku', random.Random(seed))
    given = [i for i, t in enumerate(r['prompt']) if t != '0']
    other = [i for i, t in enumerate(r['prompt']) if t == '0']
    row = [len(given)]
    for i in given + other:
        row.extend([i // 9, i % 9, int(r['answer'][i]), 99])
    return np.array(row, dtype=np.int16)


def zebra_record(index=0, h=3, a=3):
    return validate_record(convert_source_record(safe_raw(index, h, a), 'train', index))


@pytest.mark.parametrize('task,vocab,length', [('sudoku-benchmark', 14, 192), ('zebra-benchmark', 23, 384)])
def test_benchmark_layout(tmp_path, task, vocab, length):
    records = ([convert_sudoku(sudoku_row(i), 'train', i) for i in (1, 2, 3)]
               if task == 'sudoku-benchmark' else [convert_zebra(zebra_record(i)) for i in range(3)])
    write_prepared_dataset(tmp_path, task, dict(train=[records[0]], validation=[records[1]], test=[records[2]]), 17, {})
    data = ReasoningDataset(tmp_path)
    batch = data[0]
    assert data.tokenizer.vocab_size == vocab
    assert len(batch['input_ids']) == length
    assert not (batch['target_mask'] & (batch['input_ids'] == data.tokenizer.pad_id)).any()
    assert not (batch['attention_mask'] & (batch['input_ids'] == data.tokenizer.pad_id)).any()
    assert batch['target_mask'].sum() == record_answer_slots(records[0])
    assert data.manifest['benchmark_protocol'] == PROTOCOL
    assert not PROTOCOL['author_verified_exact_reproduction']
    if task == 'sudoku-benchmark':
        assert batch['input_ids'][0] == data.tokenizer.bos_id
    else:
        assert batch['input_ids'][0] != data.tokenizer.bos_id
        assert 'HOUSES' not in records[0]['prompt']
        assert 'ATTRS' not in records[0]['prompt']


@pytest.mark.parametrize('h,a', [(3,3),(3,6),(6,3),(6,6)])
def test_public_width_no_gold_length(h, a):
    record = convert_zebra(zebra_record(h=h, a=a))
    assert record_answer_slots(record) == h*a+1
    bad = dict(record, answer=['[EOS]'])
    assert record_answer_slots(bad) == h*a+1
    with pytest.raises(ValueError):
        validate_record(bad)
    with pytest.raises(ValueError):
        record_answer_slots(dict(record, metadata={}))


def test_sudoku_conversion_and_strategies_discarded():
    row = sudoku_row()
    record = convert_sudoku(row, 'test', 42)
    changed = row.copy()
    changed[4::4] = -1
    assert convert_sudoku(changed, 'test', 42) == record
    assert record['metadata']['source_index'] == 42
    assert record['prompt'].count('[MASK]') == 81-int(row[0])
    assert score_prediction(record, record['answer']+['[EOS]'])['strict_sequence_success']
    with pytest.raises(ValueError, match='Repeated'):
        changed[5:7] = changed[1:3]
        convert_sudoku(changed, 'test', 42)


@pytest.mark.parametrize('task', ['sudoku-benchmark', 'zebra-benchmark'])
def test_content_and_format_scoring(task):
    r = convert_sudoku(sudoku_row(), 'train', 0) if task.startswith('sudoku') else convert_zebra(zebra_record())
    gold = r['answer']
    good = score_prediction(r, gold+['[EOS]'])
    assert good['valid_solution'] and good['strict_sequence_success']
    no_eos = score_prediction(r, gold+['[PAD]'])
    assert no_eos['valid_solution'] and not no_eos['format_success']
    assert not no_eos['strict_sequence_success']
    assert not score_prediction(r, ['[EOS]']+gold[1:]+['[EOS]'])['valid_solution']
    assert not score_prediction(r, gold[:-1])['valid_solution']
    wrong = gold.copy(); wrong[0] = wrong[1]
    assert not score_prediction(r, wrong+['[EOS]'])['valid_solution']


def test_zebra_migration_preserves_ids_order_and_old_layout(tmp_path):
    splits = dict(train=[zebra_record(0), zebra_record(1)], validation=[zebra_record(2)], test=[zebra_record(3)])
    old, new = tmp_path/'old', tmp_path/'new'
    write_prepared_dataset(old, 'zebra-official', splits, 17, {})
    before = (old/'manifest.json').read_bytes()
    migrate_zebra(old, new)
    assert (old/'manifest.json').read_bytes() == before
    for s in splits:
        a, b = ReasoningDataset(old,s), ReasoningDataset(new,s)
        assert [r['id'] for r in a.records] == [r['id'] for r in b.records]
        assert a[0]['target_mask'].sum() == 37
        assert b[0]['target_mask'].sum() == 10
    with pytest.raises(FileExistsError):
        migrate_zebra(old, new)


def test_sudoku_source_split_leakage_and_determinism(tmp_path):
    rows = [sudoku_row(i) for i in range(1, 9)]
    # A duplicate test puzzle in training must never be selected for train/valid.
    np.save(tmp_path/'train.npy', np.stack(rows[:6]+[rows[6]]))
    np.save(tmp_path/'test.npy', np.stack(rows[6:]))
    for name in ('a','b'):
        import_sudoku(tmp_path/'train.npy', tmp_path/'test.npy', tmp_path/name, 3, 2, 2)
    assert (tmp_path/'a/manifest.json').read_bytes() == (tmp_path/'b/manifest.json').read_bytes()
    chosen = json.loads((tmp_path/'a/manifest.json').read_text())['source']['source_indices']
    assert 6 not in chosen['train']+chosen['validation']


def test_benchmark_oracle_generation_mixed_public_lengths(tmp_path):
    from reasoning.evaluation import evaluate_generation
    records = [convert_zebra(zebra_record(0,3,3)), convert_zebra(zebra_record(1,4,3))]
    write_prepared_dataset(tmp_path,'zebra-benchmark',dict(train=[convert_zebra(zebra_record(2))],
                           validation=[convert_zebra(zebra_record(3))],test=records),17,{})
    data=ReasoningDataset(tmp_path,'test'); loader=DataLoader(data,batch_size=2)
    batch=next(iter(loader)); gold=batch['input_ids']; seen=[]
    class Oracle(torch.nn.Module):
        config={'memory_mode':'none'}
        def forward(self, input_ids, **kw):
            seen.append(input_ids.clone())
            logits=torch.full((*input_ids.shape,data.tokenizer.vocab_size),-100.)
            logits.scatter_(-1,gold.unsqueeze(-1),100.)
            return dict(logits=logits)
    metrics, details=evaluate_generation(Oracle(),loader,device='cpu',seed=2026)
    assert metrics['valid_solution']==1 and metrics['strict_sequence_success']==1
    assert (seen[0][batch['target_mask']]==data.tokenizer.mask_id).all()
    assert torch.equal(seen[0][~batch['target_mask']],gold[~batch['target_mask']])
    assert [len(r['decode_order']) for r in details]==[10,13]
    assert all(r['token_selection']=='sample' for r in details)
    # Contract tampering cannot silently change padding conventions.
    path=tmp_path/'manifest.json'; manifest=json.loads(path.read_text())
    manifest['benchmark_protocol']['version']='changed'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='protocol'):
        ReasoningDataset(tmp_path)


@pytest.mark.parametrize('task', ['sudoku-benchmark', 'zebra-benchmark'])
@pytest.mark.parametrize('variant', ['vanilla', 'mdm', 'mdm_aux', 'both', 'both_aux'])
def test_benchmark_fit_validate_generate(tmp_path, cpu_pipeline, task, variant):
    from reasoning import runner
    records = ([convert_sudoku(sudoku_row(i), 'train', i) for i in range(1,7)]
               if task.startswith('sudoku') else [convert_zebra(zebra_record(i)) for i in range(6)])
    data = tmp_path/'data'; run = tmp_path/'run'
    write_prepared_dataset(data,task,dict(train=records[:2],validation=records[2:4],test=records[4:]),17,{})
    runner.train(runner.parser().parse_args([
        'train','--task',task,'--variant',variant,'--data-dir',str(data),'--run-dir',str(run),
        '--size','debug','--device','cpu','--precision','fp32','--global-batch','2',
        '--micro-batch','2','--max-steps','1','--val-every','1','--save-every','1',
        '--validation-examples','2','--eval-batch-size','2','--cpu-threads','1',
        '--merged-policy','legacy' if variant=='vanilla' else 'current_preserving']))
    out=tmp_path/'eval.json'
    runner.evaluate(runner.parser().parse_args([
        'evaluate','--checkpoint',str(run/'checkpoints/last.pt'),'--data-dir',str(data),
        '--output',str(out),'--examples','2','--batch-size','2','--device','cpu','--cpu-threads','1']))
    result=json.loads(out.read_text())
    assert result['benchmark_protocol']==PROTOCOL
    assert result['metrics']['nfe']==(82 if task.startswith('sudoku') else 10)
    assert all(e['all_slots_completed'] for e in result['examples'])


def test_queue_commands_and_report(tmp_path, monkeypatch):
    from reasoning.benchmark_queue import training_command, report
    for variant in ('vanilla','mdm','mdm_aux','both','both_aux'):
        cmd=training_command('sudoku-benchmark',variant,'data','run')
        assert cmd[cmd.index('--global-batch')+1]=='128'
        assert cmd[cmd.index('--max-steps')+1]=='5000'
        assert ('--merged-policy' in cmd)==(variant!='vanilla')
    result=dict(benchmark_protocol=copy.deepcopy(PROTOCOL), step=5000,
                contract=dict(task='zebra-benchmark',variant='vanilla',data_sha256='same'),
                metrics=dict(policy='top_prob',candidate_k=8,token_selection='paper',
                             tokens_per_step=1,memory_condition='correct',seed=2026),
                examples=[dict(id='one',scores=dict(valid_solution=True,strict_sequence_success=True))])
    path=tmp_path/'zebra-benchmark/vanilla/generation.json'; path.parent.mkdir(parents=True)
    path.write_text(json.dumps(result)); report(tmp_path)
    assert (tmp_path/'report/accuracy.png').exists()
    result['contract']['variant']='both'; result['examples'][0]['id']='different'
    path=tmp_path/'zebra-benchmark/both/generation.json'; path.parent.mkdir(parents=True)
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='same data'):
        report(tmp_path)
