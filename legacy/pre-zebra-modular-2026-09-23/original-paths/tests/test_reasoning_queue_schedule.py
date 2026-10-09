import importlib.util
import json
from pathlib import Path
import signal

import pytest
from reasoning.second_epoch_queue import select_schedule


def test_explicit_subset_reverse_order_keeps_contracts():
    variants = ['mdm','tt','tt_ea','tt_ea_np','tt_ea_rm','tt_ea_rm_np']
    plan = [dict(task=t,variant=v,target_step=42,run=t+'/'+v)
            for t in ('sudoku-benchmark','zebra-benchmark') for v in variants]
    jobs = [dict(task='sudoku-benchmark',variant='tt_ea_np')]
    jobs += [dict(task='zebra-benchmark',variant=v) for v in reversed(variants)]
    spec = dict(version=1,reason='User requests finish active Sudoku then Zebra only',jobs=jobs)
    selected = select_schedule(plan,spec)
    assert len(selected)==7
    assert [j['variant'] for j in selected[1:]]==list(reversed(variants))
    assert all(any(j is old for old in plan) for j in selected)
    for invalid in ([],jobs+jobs[:1],[dict(task='unknown',variant='tt')],
                    [dict(task='zebra-benchmark',variant='tt',target_step=999)]):
        with pytest.raises(ValueError):
            select_schedule(plan,dict(spec,jobs=invalid))


def test_handoff_never_signals_worker_and_checks_exit(monkeypatch,tmp_path):
    path = Path(__file__).resolve().parents[1]/'scripts/reasoning/handoff_active_epoch_queue.py'
    spec = importlib.util.spec_from_file_location('handoff',path)
    mod = importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    run=tmp_path/'sudoku-benchmark/tt_ea_np';run.mkdir(parents=True)
    argv=['python','-m','torch.distributed.run','--run-dir',str(run)]
    q=dict(pid=101,parent=1,birth='1',state='S',argv=['python','-m','reasoning.second_epoch_queue',str(tmp_path)])
    w=dict(pid=102,parent=101,birth='2',state='S',argv=argv)
    for name,data in {
        'status.json':dict(pid=101,command=argv),
        'requested_schedule.json':dict(jobs=[dict(task='sudoku-benchmark',variant='tt_ea_np')]),
        'plan.json':[dict(task='sudoku-benchmark',variant='tt_ea_np',run=str(run),target_step=6,examples=7)]
    }.items():
        (tmp_path/name).write_text(json.dumps(data))
    (run/'status.json').write_text(json.dumps(dict(status='finished',max_steps=6,examples_seen=21)))
    monkeypatch.setattr(mod.sys,'argv',['handoff','--queue-pid','101','--worker-pid','102','--output',str(tmp_path)])
    monkeypatch.setattr(mod,'process',lambda pid:q if pid==101 else w)
    signals=[];polls=[]
    def same(p):
        polls.append(p['pid'])
        if p['pid']==102:
            return dict(w,state='Z')
        return None if (101,signal.SIGCONT) in signals else q
    monkeypatch.setattr(mod,'same_process',same)
    monkeypatch.setattr(mod.os,'kill',lambda pid,s:signals.append((pid,s)))
    monkeypatch.setattr(mod.signal,'signal',lambda *a:None)
    monkeypatch.setattr(mod.os,'chdir',lambda *a:None)
    class Executed(BaseException):pass
    def execute(*args):raise Executed()
    monkeypatch.setattr(mod.os,'execvpe',execute)
    with pytest.raises(Executed):mod.main()
    assert signals==[(101,signal.SIGSTOP),(101,signal.SIGTERM),(101,signal.SIGCONT)]
    assert 102 in polls


def test_birth_time_rejects_reused_pid(monkeypatch):
    from scripts.reasoning.handoff_active_epoch_queue import same_process
    monkeypatch.setattr('scripts.reasoning.handoff_active_epoch_queue.process',
                        lambda _:dict(pid=101,birth='changed'))
    assert same_process(dict(pid=101,birth='original')) is None
