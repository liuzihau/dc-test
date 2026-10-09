"""Explicit, epoch-boundary microbatch migration; never changes effective batch."""
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def migrate_batch_counters(checkpoint, new_config):
    old = checkpoint['hyper_parameters']['config']
    before, after = int(old.loader.batch_size), int(new_config.loader.batch_size)
    if before == after:
        return
    if (int(old.loader.global_batch_size) != int(new_config.loader.global_batch_size)
            or int(old.trainer.devices) != int(new_config.trainer.devices)):
        raise ValueError('Microbatch migration requires unchanged world/global batch size')
    loop = checkpoint['loops']['fit_loop']
    progress = loop['epoch_loop.batch_progress']
    if not progress['is_last_batch']:
        raise ValueError('Microbatch migration is allowed only at a data-epoch boundary')
    # Convert batch counts, NOT optimizer steps. The sampler multiplies the
    # resulting count by the NEW microbatch; its consumed-row offset is preserved.
    converted = {}
    for scope in ('current', 'total'):
        converted[scope] = {}
        for key, value in progress[scope].items():
            if value * before % after:
                raise ValueError('Non-integral microbatch counter migration')
            converted[scope][key] = value * before // after
    progress.update(converted)
    print(f'Microbatch resume migration {before}->{after}: optimizer/EMA/LR unchanged', flush=True)


def apply_scheduled_batch(args, repo):
    """Called only by the parent entrypoint; old running queue can stay alive."""
    if args.stage != 'train' or args.smoke or 'LOCAL_RANK' in os.environ:
        return
    root = args.run.parent
    policy_path = root/'microbatch_policy.json'
    if not policy_path.exists():
        return
    policy = json.loads(policy_path.read_text())
    boundary = int(policy['after_step'])
    if args.target_steps <= boundary:
        return
    if args.resume is None:
        raise ValueError('Scheduled microbatch migration requires full-state resume')
    for variant in ('mdm', 'mdm_np'):
        done = root/variant/'generation'/f"epoch-{policy['after_epoch']:03d}"/'complete.json'
        if not done.exists() or json.loads(done.read_text())['step'] != boundary:
            raise RuntimeError('Both epoch-three generation stages must finish before batch migration')
    report = root/'microbatch_benchmark.json'
    if not report.exists():
        log = root/'microbatch_benchmark.log'
        print(f'Benchmarking scheduled microbatch change; log: {log}', flush=True)
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node=2', '-m', 'zebra.benchmark_batch', '--report', str(report)]
        try:
            with log.open('w') as stream:
                with subprocess.Popen(command, cwd=repo, stdout=stream, stderr=subprocess.STDOUT,
                                      start_new_session=True) as process:
                    try:
                        code = process.wait(timeout=1200)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise
                    if code:
                        raise subprocess.CalledProcessError(code, command)
        except (subprocess.SubprocessError, OSError) as exc:
            # Failed benchmark cannot alter or invalidate training checkpoints.
            result = {'selected_microbatch': 128, 'reason': f'Benchmark failed: {exc}'}
            temporary = report.with_suffix('.partial')
            temporary.write_text(json.dumps(result, indent=2)+'\n')
            temporary.replace(report)
    result = json.loads(report.read_text())
    selected = int(result['selected_microbatch'])
    if selected not in (128, 256):
        raise ValueError('Unexpected benchmark decision')
    args.microbatch = selected
    # Lightning rank 1 must receive the same effective microbatch.
    if '--microbatch' in sys.argv:
        sys.argv[sys.argv.index('--microbatch')+1] = str(selected)
    else:
        sys.argv.extend(['--microbatch', str(selected)])
    print(f'Scheduled microbatch: {selected}; global batch stays 512. {result.get("reason", "")}', flush=True)
