"""CPU-only serialized follower for the reviewed low-weight training trial."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from owt.research import ROOT, atomic_write, read_json, record_event
from owt.low_weight_diagnostics import VARIANT, verify_completed_trial, verify_protocol, sha256


def require_terminal_training(root):
    queue=read_json(root/'low_weight_queue.json') or {}
    if queue.get('stage')!='complete' or queue.get('variants')!=[VARIANT] or queue.get('target_steps')!=5000:
        raise RuntimeError('Low-weight controller did not complete the selected5000-step trial')
    verify_completed_trial(root)


def require_final_collection(path, protocol_path, protocol):
    artifact=read_json(path/'summary.json')
    if (not artifact or artifact.get('preflight') or artifact.get('evaluated_variant')!=VARIANT
            or artifact.get('protocol_sha256')!=sha256(protocol_path)
            or artifact.get('row_ids')!=protocol['row_ids'] or len(artifact.get('cells',[]))!=5):
        raise RuntimeError('Collection completion evidence differs from the selected trial')


def require_final_report(path,protocol_path,protocol):
    artifact=read_json(path/'summary.json')
    if (not artifact or artifact.get('preflight') or artifact.get('fitting_performed') is not False
            or artifact.get('protocol_sha256')!=sha256(protocol_path)
            or artifact.get('row_ids')!=protocol['row_ids'] or len(artifact.get('cells',[]))!=5
            or artifact.get('scoring_row_ids')!=protocol['fixed_fusion']['scoring_row_ids']
            or artifact.get('fixed_lambda')!=protocol['fixed_fusion']['lambda_weight']
            or not artifact.get('training_evidence')):
        raise RuntimeError('Final paired report or primary validation evidence is incomplete')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'outputs/research-notes/low_weight_diagnostic_protocol_20261001.json')
    args=parser.parse_args(); protocol=read_json(args.protocol)
    if not protocol or protocol['variant']!=VARIANT: parser.error('Registered low-weight protocol required')
    verify_protocol(protocol)
    root=ROOT/protocol['run_root']; collection=ROOT/protocol['collection_output']; report=ROOT/protocol['report_output']
    state=root/'low_weight_diagnostics_queue.json'
    with (root/'.low-weight-diagnostic-follower.lock').open('a') as controller:
        fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def status(stage,**extra):
            atomic_write(state,json.dumps(dict(stage=stage,queue_pid=os.getpid(),protocol=str(args.protocol),
                collection=str(collection),report=str(report),next_training='scientific_review_required',**extra),indent=2)+'\n')
        status('waiting_for_existing_low_weight_training')
        record_event('low_weight_diagnostic_follower_armed_20261001','Low-weight final diagnostics armed',
            'A CPU-only follower waits on the existing serialized training lock. After successful5000-step completion it '
            'evaluates only the new EMA checkpoint on five registered correct-context conditions, reuses verified saved '
            'MDM/standard-zero evidence, and reports primary validation, native errors/rescue and the fixed original fusion. No new training is launched.',
            dict(protocol=str(args.protocol),collection=str(collection),report=str(report)))
        try:
            # Parent takes no tensor/CUDA context while the current trainer owns this lock.
            with (root/'.queue.lock').open('a') as serial:
                fcntl.flock(serial,fcntl.LOCK_EX)
                require_terminal_training(root)
                verify_protocol(protocol)
            commands=[]
            if not (collection/'summary.json').exists():
                commands.append(('frozen_collection',[sys.executable,'-u','-m','owt.low_weight_diagnostics','--protocol',str(args.protocol)]))
            if not (report/'summary.json').exists():
                commands.append(('paired_reports',[sys.executable,'-u','-m','analysis.owt_low_weight_report','--protocol',str(args.protocol)]))
            log=root/'low_weight_final_diagnostics.log'
            for stage,command in commands:
                with log.open('a') as stream:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                    status(stage,worker_pid=child.pid)
                    if child.wait()!=0: raise RuntimeError(stage+' failed; inspect '+str(log))
            require_final_collection(collection,args.protocol,protocol)
            require_final_report(report,args.protocol,protocol)
            status('complete_waiting_for_scientific_review')
            record_event('low_weight_final_diagnostics_ready_20261001','Low-weight outcomes ready for review',
                'Selected training, five matched frozen conditions and fixed-fusion/error reports are complete. '
                'Scientific review and archived theory revision remain required before any additional recipe.',dict(report=str(report)))
        except Exception as exc:
            status('failed',error=str(exc)); record_event('low_weight_diagnostic_follower_failed_20261001',
                'Low-weight diagnostic follower needs inspection',str(exc)); raise


if __name__ == '__main__':
    main()
