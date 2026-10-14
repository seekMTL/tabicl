"""Frozen external validation/test. No simulator truth is added to model features."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from industrial_adapter import predict_task
from train_industrial import build_model, DEFAULT_CHECKPOINT
from icds.benchmarks.external import make_external_task, pc_models, rk4
from icds.storage import checksum, write_json
from scipy.integrate import solve_ivp


def paired_task(task, shard, manifest):
    """Deterministic PC step restore; IB has no validated fork implementation."""
    if shard['domain']=='industrial_benchmark':
        return None
    spec=shard['spec'];modules=pc_models()
    model=(modules.cstr if shard['domain']=='cstr' else modules.multistage_extraction)(int_method='numpy', **shard['hidden_generator_parameters'])
    arrays=np.load(Path(manifest).parent/shard['path']);n=task['context_size'];rows=np.array(task['selected_rows'][n:]);xs=arrays['observation'][rows];acts=arrays['action'][rows].copy()
    middle=(np.array(spec['high'])+spec['low'])/2
    acts[:,0]=np.where(acts[:,0]<middle[0],spec['high'][0],spec['low'][0])
    alt=[]
    for x,a in zip(xs,acts):
        if shard['domain']=='cstr':
            ref=solve_ivp(lambda t,z:model(z,a),(0,spec['dt']),x,method='DOP853',rtol=1e-9,atol=1e-11)
            if not ref.success:raise FloatingPointError('External fork failed')
            y=ref.y[:,-1]
        else:y=rk4(model,x.copy(),a,spec['dt'])
        alt.append(y-x)
    copy=dict(task)
    for key in ['X_state','X_action']:
        copy[key]=np.concatenate([task[key][:,:n],np.repeat(task[key][:,n:],2,axis=1)],axis=1)
    na=len(spec['low']);copy['X_action'][0,n+1::2,:na]=2*(acts-spec['low'])/(np.array(spec['high'])-spec['low'])-1
    raw=np.repeat(task['raw_y'][n:],2,axis=0);raw[1::2]=alt
    copy['raw_y']=np.r_[task['raw_y'][:n],raw]
    copy['current_state']=np.r_[task['current_state'][:n],np.repeat(task['current_state'][n:],2,axis=0)]
    copy['current_target_state']=np.r_[task['current_target_state'][:n],np.repeat(task['current_target_state'][n:],2,axis=0)]
    copy['Y']=((copy['raw_y']-task['scaler']['target_mean'])/task['scaler']['target_scale']).astype(np.float32)[None]
    copy['group_ids']=np.r_[task['group_ids'][:n],np.repeat(task['group_ids'][n:],2,axis=0)];copy['group_ids'][n+1::2,3]=1
    return copy


@torch.no_grad()
def evaluate_external(model, manifest, split, contexts, device, pairs=True, history_override=None, support_seed=None, target_chunk_size=2, time_feature='source'):
    meta=json.loads(Path(manifest).read_text());records=[];model.eval()
    for shard in meta['shards']:
        if shard['split']!=split:continue
        if checksum(Path(manifest).parent/shard['path'])!=shard['sha256']:raise ValueError('External dataset checksum mismatch')
        for c in contexts:
            task=make_external_task(manifest,shard,c,64, history=history_override if shard["domain"]=="industrial_benchmark" else None, eligible_history=30 if history_override is not None and shard["domain"]=="industrial_benchmark" else None, support_seed=support_seed, time_feature=time_feature)
            scaled=predict_task(model,task,device,chunk_size=target_chunk_size).cpu().numpy();n=task['context_size'];ref=np.array(shard['spec']['references'])
            pred=scaled*task['scaler']['target_scale']+task['scaler']['target_mean'];error=pred-task['raw_y'][n:]
            record={'domain':shard['domain'],'world':shard['world'],'context':c,'history':task['history'],
                    'per_target_rmse':np.sqrt(np.mean(error**2,0)).tolist(),'per_target_mae':np.mean(np.abs(error),0).tolist(),'nrmse':float(np.sqrt(np.mean((error/ref)**2))),
                    'query_ids':task['group_ids'][n:].tolist(), 'reference_scales':ref.tolist(),'source_time_unit':task['feature_spec']['source_time_unit'],'model_time_unit':task['feature_spec']['model_time_unit'],'model_dt':task['feature_spec']['model_dt']}
            if pairs:
                pair=paired_task(task,shard,manifest)
                if pair is not None:
                    ps=predict_task(model,pair,device).cpu().numpy();p=ps*task['scaler']['target_scale']+task['scaler']['target_mean'];actual=pair['raw_y'][n:]
                    pe=p[1::2]-p[::2];ae=actual[1::2]-actual[::2];nonzero=np.abs(ae/ref)>1e-5
                    record.update(effect_nrmse=float(np.sqrt(np.mean(((pe-ae)/ref)**2))),effect_per_target_rmse=np.sqrt(np.mean((pe-ae)**2,0)).tolist(),nonzero_effect_sign_accuracy=float(np.mean(np.sign(pe[nonzero])==np.sign(ae[nonzero]))) if nonzero.any() else None)
            records.append(record)
    macro={}
    for c in contexts:
        macro[str(c)]={domain:float(np.mean([r['nrmse'] for r in records if r['context']==c and r['domain']==domain])) for domain in sorted({r['domain'] for r in records})}
    return {'records':records,'domain_macro_nrmse':macro,'split':split,'manifest_sha256':checksum(manifest),'time_feature':time_feature,
            'limitations':['PC original NumPy model equations with independent rollout harness, not Gym reward evaluation','IB history contains only public past observations/actions, no hidden Markov state','IB effect/recursive rollout not implemented; no claim of stochastic joint simulation','NRMSE is target-scale RMS within each world, then world/domain macro averaged']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--checkpoint');p.add_argument('--base-checkpoint',default=DEFAULT_CHECKPOINT);p.add_argument('--model-kind',choices=['caq','tabicl','caq-concat'],default='caq');p.add_argument('--split',choices=['external_val','external_test'],default='external_val');p.add_argument('--context-sizes',type=int,nargs='+',default=[128]);p.add_argument('--ib-history',type=int,choices=[1,8,30]);p.add_argument('--time-feature',choices=['source','si_seconds'],default='source');p.add_argument('--device',default='cuda:0');p.add_argument('--out',required=True);args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True) if args.checkpoint else None
    model=build_model(state['base_checkpoint'] if state else args.base_checkpoint,args.device,state['args'].get('model_kind','caq') if state else args.model_kind)
    if state:model.load_state_dict(state['model'],strict=True)
    model.target_selector=state['args'].get('target_selector',False) if state else True
    model.target_anchor=state['args'].get('target_anchor',False) if state else False
    model.absolute_state=state['args'].get('absolute_state',False) if state else False
    model.persistence_residual=state['args'].get('persistence_residual',False) if state else False
    model.persistence_residual_scope=state['args'].get('persistence_residual_scope','all') if state else 'all'
    report=evaluate_external(model,args.manifest,args.split,args.context_sizes,args.device,history_override=args.ib_history,time_feature=args.time_feature)
    report.update(checkpoint=args.checkpoint or 'A0 original initialization',checkpoint_sha256=checksum(args.checkpoint or args.base_checkpoint),selected_step=state['step'] if state else 0,device=args.device,torch=str(torch.__version__),evaluation_source_sha256=checksum(__file__),adapter_source_sha256=checksum(Path(__file__).with_name('industrial_adapter.py')))
    write_json(args.out,report);print(json.dumps({'out':args.out,'domain_macro_nrmse':report['domain_macro_nrmse']},indent=2),flush=True)

if __name__=='__main__':main()
