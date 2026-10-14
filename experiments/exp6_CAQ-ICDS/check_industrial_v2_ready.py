"""Enforce real data/preflight/short-run gates before starting long runs."""
import argparse
import json
from pathlib import Path
from industrial_adapter import ICDS_ROOT
from icds.storage import checksum
from train_industrial import DEFAULT_CHECKPOINT


def check(phase):
    root=Path(__file__).resolve().parent
    pre=json.loads((root/'checkpoints/industrial-v2-preflight.json').read_text())
    assert pre['status']=='passed' and pre['cpu']['worlds']==512
    assert pre['manifest_sha256']==checksum(ICDS_ROOT/'data/industrial-v2/manifest.json')
    assert pre['base_checkpoint_sha256']==checksum(DEFAULT_CHECKPOINT)
    assert pre['context']==960 and pre['query']==256
    assert all(v['status']=='passed' and v['frozen_icl_unchanged'] for v in pre['gpu'].values())
    if phase=='formal':
        runs=[]
        for kind in ['caq','tabicl']:
            path=root/f'checkpoints/industrial-v2/smoke-{kind}-seed42'
            summary=json.loads((path/'summary.json').read_text())
            args=json.loads((path/'args.json').read_text())
            assert summary['status']=='completed' and summary['steps']==200
            assert summary['successful_updates']==200
            assert summary['skipped_steps']==0 and summary['rolled_back']==0
            assert summary['manifest_sha256']==pre['manifest_sha256']
            assert summary['base_checkpoint_sha256']==pre['base_checkpoint_sha256']
            assert args['input_scale_profile']=='industrial_v2' and args['schedule_steps']==30000
            assert args['cuda_memory_fraction']==.45
            assert args['history_ratio']==0 and args['context_size']==960 and args['query_size']==256
            assert len(summary['scenario_validation_fixed_scale_mse_after'])==8
            for source,digest in json.loads((path/'source-hashes.json').read_text()).items():
                assert checksum(source)==digest, f'Source changed after smoke: {source}'
            runs.append(summary)
        assert runs[0]['industrial_task_sequence_sha256']==runs[1]['industrial_task_sequence_sha256']
    print(f'READY: {phase}; industrial-v2, full-state hr0, seed42, context960/query256',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['smoke','formal'],required=True)
    check(parser.parse_args().phase)
