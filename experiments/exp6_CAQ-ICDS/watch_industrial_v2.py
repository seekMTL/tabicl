"""Watch exact training processes and analyze only after both truly finish."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from industrial_adapter import ICDS_ROOT
from icds.storage import write_json

EXP=Path(__file__).resolve().parent


def process_start(pid):
    text=(Path('/proc')/str(pid)/'stat').read_text()
    return text[text.rfind(')')+2:].split()[19]


def alive(pid,start,out):
    try:
        if process_start(pid)!=start:return False
        command=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
        return b'train_industrial.py' in b' '.join(command) and os.fsencode(str(out)) in command
    except (FileNotFoundError,ProcessLookupError):
        return False


def last_row(path):
    if not path.exists():return None
    with path.open('rb') as stream:
        stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-8192))
        rows=stream.read().splitlines()
    for row in reversed(rows):
        try:return json.loads(row)
        except json.JSONDecodeError:continue
    return None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--caq-pid',type=int,required=True)
    parser.add_argument('--ft-pid',type=int,required=True);args=parser.parse_args()
    root=EXP/'checkpoints/industrial-v2'
    processes={'caq':args.caq_pid,'tabicl':args.ft_pid}
    starts={k:process_start(pid) for k,pid in processes.items()}
    paths={k:root/f'{k}-A4-seed42-hr0' for k in processes}
    assert all(alive(processes[k],starts[k],paths[k]) for k in processes)
    prior=None
    while True:
        state={}
        for kind in processes:
            path=paths[kind]
            row=last_row(path/'history.jsonl')
            summary=json.loads((path/'summary.json').read_text()) if (path/'summary.json').exists() else None
            running=alive(processes[kind],starts[kind],path)
            state[kind]=dict(pid=processes[kind],running=running,
                             step=row['step'] if row else 0,
                             completed=summary is not None and summary.get('status')=='completed')
        write_json(root/'progress.json',dict(updated_unix=time.time(),models=state))
        milestone=tuple((k,v['step']//1000,v['running'],v['completed']) for k,v in state.items())
        if milestone!=prior:
            print(json.dumps(state),flush=True);prior=milestone
        failed=[k for k,v in state.items() if not v['running'] and not v['completed']]
        if failed:
            write_json(root/'completion-audit.json',dict(status='failed',models=state,failed_models=failed))
            raise RuntimeError(f'Training stopped without completed summary: {failed}')
        if all(not v['running'] and v['completed'] for v in state.values()):
            result=subprocess.run([sys.executable,str(EXP/'summarize_industrial_v2.py'),'--phase','formal'])
            write_json(root/'completion-audit.json',dict(status='passed' if result.returncode==0 else 'analysis_failed',models=state))
            if result.returncode:raise RuntimeError('Completed-run verification failed')
            print('BOTH_RUNS_COMPLETED_AND_VERIFIED',root/'formal-results.md',flush=True)
            return
        time.sleep(30)


if __name__=='__main__':main()
