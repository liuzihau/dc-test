import copy
import json
from types import SimpleNamespace as NS

import pytest
from zebra.batch_policy import migrate_batch_counters, apply_scheduled_batch


def checkpoint():
    return {'hyper_parameters': {'config': NS(loader=NS(batch_size=128, global_batch_size=512),
        trainer=NS(devices=2))}, 'global_step': 8790, 'loops': {'fit_loop': {
        'epoch_loop.batch_progress': {'is_last_batch': True,
            'current': dict(ready=5860, started=5860, processed=5860, completed=5860),
            'total': dict(ready=17580, started=17580, processed=17580, completed=17580)},
        'epoch_loop.state_dict': {'_batches_that_stepped': 8790}}}}


def test_counter_conversion_preserves_samples_and_updates():
    c = checkpoint()
    migrate_batch_counters(c, NS(loader=NS(batch_size=256, global_batch_size=512), trainer=NS(devices=2)))
    progress = c['loops']['fit_loop']['epoch_loop.batch_progress']
    assert progress['current']['completed'] == 2930
    assert progress['current']['completed'] * 256 == 5860 * 128
    assert progress['total']['completed'] * 256 == 17580 * 128
    assert c['global_step'] == 8790
    assert c['loops']['fit_loop']['epoch_loop.state_dict']['_batches_that_stepped'] == 8790


def test_no_change_and_mid_epoch_rejected():
    c = checkpoint()
    old = copy.deepcopy(c)
    migrate_batch_counters(c, c['hyper_parameters']['config'])
    assert c == old
    c['loops']['fit_loop']['epoch_loop.batch_progress']['is_last_batch'] = False
    with pytest.raises(ValueError, match='epoch boundary'):
        migrate_batch_counters(c, NS(loader=NS(batch_size=256, global_batch_size=512), trainer=NS(devices=2)))


def test_keep_global_batch():
    with pytest.raises(ValueError, match='unchanged'):
        migrate_batch_counters(checkpoint(), NS(loader=NS(batch_size=256, global_batch_size=1024), trainer=NS(devices=2)))


def test_policy_waits_for_current_chunk(tmp_path, monkeypatch):
    monkeypatch.delenv('LOCAL_RANK', raising=False)
    (tmp_path/'microbatch_policy.json').write_text(json.dumps({'after_step': 8790, 'after_epoch': 3}))
    args = NS(stage='train', smoke=False, run=tmp_path/'mdm_np', target_steps=8790, microbatch=128)
    apply_scheduled_batch(args, tmp_path)
    assert args.microbatch == 128
    assert not (tmp_path/'microbatch_benchmark.json').exists()


def test_policy_requires_both_evaluations_and_uses_decision(tmp_path, monkeypatch):
    monkeypatch.delenv('LOCAL_RANK', raising=False)
    monkeypatch.setattr('sys.argv', ['entrypoint.py', '--microbatch', '128'])
    (tmp_path/'microbatch_policy.json').write_text(json.dumps({'after_step':8790, 'after_epoch':3}))
    args = NS(stage='train', smoke=False, run=tmp_path/'mdm', target_steps=17580,
              resume=tmp_path/'source.ckpt', microbatch=128)
    with pytest.raises(RuntimeError, match='Both'):
        apply_scheduled_batch(args, tmp_path)
    for variant in ('mdm', 'mdm_np'):
        directory = tmp_path/variant/'generation/epoch-003'
        directory.mkdir(parents=True)
        (directory/'complete.json').write_text(json.dumps({'step':8790}))
    (tmp_path/'microbatch_benchmark.json').write_text(json.dumps({'selected_microbatch':256}))
    apply_scheduled_batch(args, tmp_path)
    assert args.microbatch == 256
    import sys
    assert sys.argv[-1] == '256'


def test_failed_benchmark_falls_back(tmp_path, monkeypatch):
    monkeypatch.delenv('LOCAL_RANK', raising=False)
    monkeypatch.setattr('sys.argv', ['entrypoint.py'])
    (tmp_path/'microbatch_policy.json').write_text(json.dumps({'after_step':8790, 'after_epoch':3}))
    for variant in ('mdm', 'mdm_np'):
        directory = tmp_path/variant/'generation/epoch-003'
        directory.mkdir(parents=True)
        (directory/'complete.json').write_text(json.dumps({'step':8790}))
    def fail(*args, **kwargs):
        raise OSError('test launch failure')
    monkeypatch.setattr('subprocess.Popen', fail)
    args = NS(stage='train', smoke=False, run=tmp_path/'mdm', target_steps=17580,
              resume=tmp_path/'source.ckpt', microbatch=128)
    apply_scheduled_batch(args, tmp_path)
    assert args.microbatch == 128
    assert json.loads((tmp_path/'microbatch_benchmark.json').read_text())['selected_microbatch'] == 128
