"""Audit v2 data contracts, then exercise real training-sized GPU batches."""
import argparse
import gc
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from industrial_adapter import (ABSOLUTE_INPUT_SCALES, ICDS_ROOT,
                                absolute_input_scales, predict_task, to_target_batch)
from industrial_task_sampler import IndustrialTaskSampler
from industrial_common import isolated_seed, scm_batch
from train_industrial import DEFAULT_CHECKPOINT, build_model
from train_large6_counterfactual import SCMReplaySampler
from icds.registry import SCALES, scale_for
from icds.storage import checksum, load_shard, write_json

MANIFEST = ICDS_ROOT / 'data/industrial-v2/manifest.json'
EXP = Path(__file__).resolve().parent
FLAGS = dict(target_selector=True, target_anchor=True, absolute_state=True,
             persistence_residual=True, persistence_residual_scope='next_observation',
             input_scale_profile='industrial_v2')


def audit():
    samplers = {s: IndustrialTaskSampler(MANIFEST, s, 960, 256, 1, 42,
                    target_floor_profile='event_v2', external_context_mode='observed')
                for s in ['meta_train', 'meta_val', 'meta_test']}
    for a,b in [('meta_train','meta_val'),('meta_train','meta_test'),('meta_val','meta_test')]:
        assert not set(samplers[a].worlds) & set(samplers[b].worlds)
    entries=[]; reference=hashlib.sha256(); minimum_pool=1000000
    representatives={}; max_features=0
    for split,sampler in samplers.items():
        for wid in sampler.worlds:
            shard=sampler.shards[wid]
            program=json.loads((sampler.root/shard['program']).read_text())
            arrays=load_shard(sampler.root,shard)
            state_scales=program.get('state_scales') or [SCALES[u] for u in program['state_units']]
            pools=[int(np.sum((arrays['ids'][:,1]==ep)&(arrays['ids'][:,3]==0)))
                   for ep in np.unique(arrays['ids'][:,1])]
            minimum_pool=min(minimum_pool,sum(sorted(pools)[:len(pools)//2]))
            for paired in [False,True]:
                task=sampler.fixed_tasks([wid],paired=paired)[0][1]
                n=task['context_size']; sel=task['selected_rows']; ids=task['group_ids']
                assert not set(ids[:n,1]) & set(ids[n:,1])
                assert not set(ids[:n,2]) & set(ids[n:,2])
                assert len(np.unique(sel))==len(sel)
                if paired:
                    np.testing.assert_array_equal(ids[n::2,2],ids[n+1::2,2])
                raw=arrays['next_state'][sel].astype(float)-arrays['state'][sel].astype(float)
                kinds=program.get('state_kinds') or ['continuous']*len(program['state_names'])
                for j,k in enumerate(kinds):
                    if k in {'count','binary'}: raw[:,j]=arrays['next_state'][sel,j]
                np.testing.assert_array_equal(task['raw_y'],raw)
                mean=raw[:n].mean(0)
                floors=np.array([s['target_floor'] for s in state_scales])
                scale=np.maximum(raw[:n].std(0),floors)
                for j,k in enumerate(kinds):
                    if k=='binary': mean[j],scale[j]=0.,1.
                    if program['family']=='hybrid_line' and program['state_names'][j].endswith('_wear'):
                        scale[j]=max(scale[j],.01)
                np.testing.assert_allclose(task['scaler']['target_mean'],mean,rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(task['scaler']['target_scale'],scale,rtol=1e-12,atol=1e-12)
                dynamic=np.c_[arrays['state'][sel],arrays['exogenous_observed'][sel]].astype(float)
                dm=dynamic[:n].mean(0)
                ds=np.maximum(dynamic[:n].std(0),np.r_[[s['dynamic_floor'] for s in state_scales],
                              scale_for(task['exogenous_unit'],'dynamic_floor')])
                for j,k in enumerate(kinds):
                    if k=='binary':dm[j],ds[j]=0.,1.
                np.testing.assert_allclose(task['scaler']['dynamic_mean'],dm,rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(task['scaler']['dynamic_scale'],ds,rtol=1e-12,atol=1e-12)
                np.testing.assert_array_equal(task['target_spec']['reference_scales'],
                                              [s['target_reference'] for s in state_scales])
                X,A,support,query=to_target_batch(task,'cpu',target_columns=[0],**FLAGS)
                assert all(torch.isfinite(t).all() for t in [X,A,support,query])
                max_features=max(max_features,X.shape[-1])
                if not paired:
                    # Both profiles must preserve targets, references and action input.
                    old=to_target_batch(task,'cpu',target_columns=[0],**{**FLAGS,'input_scale_profile':'legacy'})
                    for i in [1,2,3]: torch.testing.assert_close(old[i],[X,A,support,query][i],rtol=0,atol=0)
                    reference.update(np.asarray(task['target_spec']['reference_scales']).tobytes())
                    entries.append(dict(split=split,world=wid,scenario=sampler.scenarios[wid],
                                        mechanism=program['mechanism'],state_dim=task['state_dim'],
                                        features=int(X.shape[-1]),action_dim=arrays['action'].shape[1]))
                    if split=='meta_train':
                        key=program['mechanism'] if program['family']=='hybrid_line' else sampler.scenarios[wid]
                        if key not in representatives or X.shape[-1]>representatives[key]['features']:
                            representatives[key]=entries[-1]
    assert minimum_pool>=960
    assert len({e['scenario'] for e in entries})==8
    return dict(status='passed',worlds=len(entries),tensor_tasks=len(entries)*2,
                split_world_counts={k:len(v.worlds) for k,v in samplers.items()},
                minimum_factual_context_pool=minimum_pool,max_typed_features=max_features,
                reference_scales_sha256=reference.hexdigest(),input_scales=ABSOLUTE_INPUT_SCALES,
                support_only_scalers_recomputed=True,typed_targets_recomputed=True,
                profile_preserves_targets_actions_and_references=True,
                world_episode_source_isolation_verified=True,world_inventory=entries,
                gpu_representatives=list(representatives.values())), samplers['meta_train']


def gpu_checks(sampler,representatives,device):
    torch.cuda.set_device(device); torch.set_num_threads(4)
    results={}
    scm=SCMReplaySampler(batch_size=1,min_features=5,max_features=12,
                         min_seq_len=1216,max_seq_len=1217,n_jobs=1)
    for kind in ['caq','tabicl']:
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
        model=build_model(DEFAULT_CHECKPOINT,device,kind)
        for name,value in FLAGS.items():setattr(model,name,value)
        native=model.native if kind=='tabicl' else model
        assert all(not p.requires_grad for p in native.icl_predictor.parameters())
        frozen={k:p.detach().cpu().clone() for k,p in model.named_parameters() if not p.requires_grad}
        params=[p for p in model.parameters() if p.requires_grad]
        opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
        records=[]
        for entry in representatives:
            wid=entry['world'];task=sampler.fixed_tasks([wid],paired=True)[0][1]
            columns=list(range(task['state_dim'])) if entry['scenario'] in {'mass_spring','dc_motor'} else [0]
            for col in columns:
                model.train();opt.zero_grad(set_to_none=True)
                X,A,support,query=to_target_batch(task,device,target_columns=[col],**FLAGS)
                A.requires_grad_(True)
                pred=model(X,X_action=A,y_train=support).mean(-1)
                assert pred.shape==query.shape and torch.isfinite(pred).all()
                loss=(pred-query).square().mean();loss.backward()
                assert torch.isfinite(loss) and A.grad is not None and torch.isfinite(A.grad).all()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
                norm=torch.nn.utils.clip_grad_norm_(params,1.)
                opt.step()
                assert all(torch.isfinite(p).all() for p in params)
                record=dict(model=kind,world=wid,scenario=entry['scenario'],mechanism=entry['mechanism'],
                            target=task['target_spec']['state_names'][col],features=X.shape[-1],
                            loss=float(loss.detach()),gradient_norm=float(norm),
                            action_gradient_abs_max=float(A.grad.abs().max()),
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
                records.append(record);print(json.dumps(record),flush=True)
            model.eval()
            with torch.no_grad():
                pred_eval=predict_task(model,task,device,chunk_size=2)
            assert pred_eval.shape==task['raw_y'][960:].shape and torch.isfinite(pred_eval).all()
        model.train();opt.zero_grad(set_to_none=True)
        with isolated_seed(42):X,y,n=scm_batch(scm,device)
        pred=model(X,y_train=y[:,:n]).mean(-1);loss=(pred-y[:,n:]).square().mean();loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
        opt.step()
        for name,param in model.named_parameters():
            if name in frozen: torch.testing.assert_close(param.detach().cpu(),frozen[name],rtol=0,atol=0)
        results[kind]=dict(status='passed',records=records,scm_loss=float(loss.detach()),
                           frozen_icl_unchanged=True,trainable_parameters=sum(p.numel() for p in params),
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        del X,A,support,query,pred,pred_eval,loss,opt,model,params,native,frozen
        gc.collect();torch.cuda.empty_cache()
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1');p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    cpu,sampler=audit()
    print(json.dumps({k:v for k,v in cpu.items() if k!='world_inventory'}),flush=True)
    report=dict(manifest=str(MANIFEST),manifest_sha256=checksum(MANIFEST),
                base_checkpoint_sha256=checksum(DEFAULT_CHECKPOINT),context=960,query=256,
                cpu=cpu,gpu=gpu_checks(sampler,cpu['gpu_representatives'],torch.device(args.device)))
    report['status']='passed';write_json(args.out,report)
    print('PREFLIGHT_PASSED',args.out,flush=True)


if __name__=='__main__':main()
