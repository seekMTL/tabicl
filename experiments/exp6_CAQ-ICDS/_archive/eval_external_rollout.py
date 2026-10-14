"""Recursive continuous-state rollout given action plans; no teacher forcing."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from industrial_adapter import predict_task
from train_industrial import build_model
from icds.benchmarks.external import make_external_task
from icds.storage import checksum, write_json


@torch.no_grad()
def rollout(model, manifest, split, device, context=128, queries=16, time_feature='source'):
    metadata=json.loads(Path(manifest).read_text());records=[]
    for shard in metadata['shards']:
        if shard['split']!=split or shard['domain']=='industrial_benchmark':continue
        arrays=np.load(Path(manifest).parent/shard['path']);task=make_external_task(manifest,shard,context,64,time_feature=time_feature)
        pool=np.array(task['selected_rows'][context:]);eligible=pool[arrays['ids'][pool,4] < metadata['steps']-20]
        rows=eligible[:queries]
        if len(rows)<queries:raise ValueError('Insufficient recursive start states')
        n=context;d=task['state_dim'];spec=shard['spec'];ref=np.array(spec['references']);x=arrays['observation'][rows].copy();initial=x.copy()
        dm=np.array(task['feature_spec']['dynamic_mean']);ds=np.array(task['feature_spec']['dynamic_scale']);invalid=0
        view=dict(task);view['X_state']=np.concatenate([task['X_state'][:,:n],task['X_state'][:,n:n+queries]],1).copy()
        view['X_action']=np.concatenate([task['X_action'][:,:n],task['X_action'][:,n:n+queries]],1).copy()
        view['Y']=np.zeros((1,n+queries,d),np.float32);view['Y'][:,:n]=task['Y'][:,:n]
        # Keep the task row contract aligned when replacing 64 queries by rollout starts.
        # Query labels are placeholders, never inputs or teacher-forced targets.
        view['raw_y']=np.r_[task['raw_y'][:n],np.zeros((queries,d))]
        view['group_ids']=np.r_[task['group_ids'][:n],arrays['ids'][rows]]
        view['current_target_state']=np.r_[task['current_target_state'][:n],x].copy()
        view['current_state']=np.r_[task['current_state'][:n],x].copy()
        for h in range(1,21):
            a=arrays['action'][rows+h-1];na=len(spec['low'])
            view['current_target_state'][n:]=x
            view['current_state'][n:]=x
            view['X_state'][0,n:,:d]=((x-dm)/ds).astype(np.float32)
            view['X_action'][0,n:,:na]=(2*(a-spec['low'])/(np.array(spec['high'])-spec['low'])-1).astype(np.float32)
            delta=predict_task(model,view,device).cpu().numpy()*task['scaler']['target_scale']+task['scaler']['target_mean']
            x=x+delta
            invalid+=int((~np.isfinite(x).all(1) | (x<0).any(1)).sum())
            if not np.isfinite(x).all():raise FloatingPointError('Nonfinite recursive forecast; no silent correction')
            if h in [1,5,20]:
                actual=arrays['next_observation'][rows+h-1];err=x-actual
                records.append({'domain':shard['domain'],'world':shard['world'],'horizon':h,'physical_time':h*spec['dt'],'physical_time_unit':task['feature_spec']['source_time_unit'],'physical_time_seconds':h*task['feature_spec']['model_dt'] if time_feature=='si_seconds' else h*spec['dt']*(60 if shard['domain']=='cstr' else 3600),'context':n,'query_start_ids':arrays['ids'][rows].tolist(),
                                'nrmse':float(np.sqrt(np.mean((err/ref)**2))),'per_target_rmse':np.sqrt(np.mean(err**2,0)).tolist(),
                                'invalid_state_predictions_cumulative':invalid,'state_corrections':0,
                                'persistence_nrmse':float(np.sqrt(np.mean(((initial-actual)/ref)**2)))})
    return {'records':records,'protocol':'initial public state + fixed support; predicted states recur, real action plan given, no teacher forcing, no clipping','split':split,'manifest_sha256':checksum(manifest),'time_feature':time_feature,
            'limitations':['Continuous fully observed PC processes only; no hidden IB state rollout or stochastic joint rollout','CSTR equation uses minutes; extraction time uses hours from upstream parameters','Invalid negative predictions counted and propagated without correction']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--split',choices=['external_val','external_test'],default='external_val');p.add_argument('--out',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--time-feature',choices=['source','si_seconds'],default='source');args=p.parse_args()
    torch.set_num_threads(4);state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=build_model(state['base_checkpoint'],args.device,state['args'].get('model_kind','caq'));model.load_state_dict(state['model'],strict=True);model.target_selector=state['args'].get('target_selector',False);model.target_anchor=state['args'].get('target_anchor',False);model.absolute_state=state['args'].get('absolute_state',False);model.persistence_residual=state['args'].get('persistence_residual',False);model.persistence_residual_scope=state['args'].get('persistence_residual_scope','all');model.eval()
    result=rollout(model,args.manifest,args.split,args.device,time_feature=args.time_feature);result['checkpoint_sha256']=checksum(args.checkpoint);write_json(args.out,result)
    print(json.dumps({'out':args.out,'records':len(result['records'])}),flush=True)

if __name__=='__main__':main()
