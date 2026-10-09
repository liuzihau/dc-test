"""CPU-only diagnostic follower for one already selected source-policy arm."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

from owt.research import ROOT, atomic_write, read_json, record_event
from owt.source_pairing_diagnostics import VARIANTS, verify_protocol, verify_trial_completed, sha256
from analysis.owt_source_pairing_report import require_collection


def require_terminal_training(root, variant):
    queue=read_json(root/'source_pairing_queue.json') or {}
    if queue.get('variant')!=variant or queue.get('stage')!='complete_waiting_for_scientific_review':
        raise RuntimeError('Selected source controller has not completed the registered arm')
    verify_trial_completed(root,variant)


def require_report(folder,collection,protocol_path,protocol,variant):
    report=read_json(folder/'summary.json') or {}
    if (report.get('preflight') is not False
            or report.get('variant')!=variant
            or report.get('protocol_sha256')!=sha256(protocol_path)
            or report.get('collection_sha256')!=sha256(collection/'summary.json')
            or report.get('row_ids')!=protocol['row_ids']
            or report.get('scoring_row_ids')!=protocol['fixed_fusion']['scoring_row_ids']
            or report.get('fixed_lambda')!=protocol['fixed_fusion']['lambda_weight']
            or report.get('fitting_performed') is not False
            or report.get('model_forward_passes')!=0
            or [c.get('cell') for c in report.get('cells',[])]!=protocol['conditions']
            or not report.get('training_evidence')):
        raise ValueError('Real final source report is incomplete or refitted')
    if variant==VARIANTS[1] and (not report.get('paired_source_arm_available')
            or not report['training_evidence'].get('matched_two_arm_supervision_audit')):
        raise ValueError('Count-control report lacks the required masked-source contrast')
    for cell in report['cells']:
        required={'trial_union',*[name+'_vs_trial_main_all' for name in
                                  ('mdm','mdm_np_zero_init','mdm_np_zero_init_low_weight')]}
        if variant==VARIANTS[1]:required.add('count_vs_masked_main_all')
        if not required.issubset(cell.get('comparisons',{})):
            raise ValueError('Source report is missing required main/error comparisons')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--variant',choices=VARIANTS,required=True)
    args=parser.parse_args();protocol=read_json(args.protocol);verify_protocol(protocol)
    root=ROOT/protocol['run_root'];settings=protocol['outputs'][args.variant]
    collection=ROOT/settings['collection'];report=ROOT/settings['report']
    state=ROOT/settings['queue'];log=root/(args.variant+'_diagnostics.log')
    with (root/('.'+args.variant+'-diagnostic-follower.lock')).open('a') as controller:
        fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def status(stage,**extra):
            atomic_write(state,json.dumps(dict(stage=stage,variant=args.variant,queue_pid=os.getpid(),
                protocol=str(args.protocol),collection=str(collection),report=str(report),
                next_training='scientific_review_and_serial_order_required',**extra),indent=2)+'\n')
        status('waiting_for_existing_source_training')
        record_event('source_diagnostic_follower_armed_'+args.variant+'_20261002',
            'Source-policy final diagnostics armed',
            'A CPU-only follower waits on the shared serialization lock for the existing selected source-policy trainer. '
            'After verified 5,000-step completion it collects five fixed EMA/FP32 conditions and compares saved '
            'MDM/standard-zero/lower-weight evidence. It launches no training arm.',dict(variant=args.variant,protocol=str(args.protocol)))
        try:
            with (root/'.queue.lock').open('a') as serial:
                fcntl.flock(serial,fcntl.LOCK_EX)
                require_terminal_training(root,args.variant)
                verify_protocol(protocol)
            commands=[]
            if not (collection/'summary.json').exists():
                commands.append(('frozen_collection',[sys.executable,'-u','-m','owt.source_pairing_diagnostics',
                    '--protocol',str(args.protocol),'--variant',args.variant]))
            if not (report/'summary.json').exists():
                commands.append(('paired_reports',[sys.executable,'-u','-m','analysis.owt_source_pairing_report',
                    '--protocol',str(args.protocol),'--variant',args.variant]))
            for stage,command in commands:
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3',OMP_NUM_THREADS='2',
                         OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
                with log.open('a') as stream:
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                    status(stage,worker_pid=child.pid)
                    if child.wait()!=0:raise RuntimeError(stage+' failed; inspect '+str(log))
            require_collection(collection,args.protocol,protocol,args.variant)
            require_report(report,collection,args.protocol,protocol,args.variant)
            status('complete_waiting_for_scientific_review')
            record_event('source_final_diagnostics_ready_'+args.variant+'_20261002',
                'Source-policy final outcomes ready',
                'The selected arm and all five final frozen/error/fixed-fusion conditions are complete. '
                'Scientific review is required; source placement needs the serial matched-count contrast.',
                dict(variant=args.variant,report=str(report)))
        except Exception as exc:
            status('failed',error=str(exc))
            record_event('source_diagnostic_failure_'+args.variant+'_20261002',
                         'Source diagnostic follower needs inspection',str(exc))
            raise


if __name__=='__main__':
    main()
