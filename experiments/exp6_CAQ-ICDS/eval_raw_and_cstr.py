"""Read-only baseline audit and PC-Gym CSTR transfer of frozen hr0 models.

No external labels update neural weights; all estimators see identical row IDs.
Only development splits are supported. Predictions and provenance are persisted.
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from industrial_adapter import ICDS_ROOT, predict_task, persistence_target
from industrial_task_sampler import IndustrialTaskSampler
from compare_three import (OfficialTabICLRegressorAdapter, load, CAQ_CKPT,
                           TABICL_FT_CKPT, MANIFEST)
from train_industrial import DEFAULT_CHECKPOINT
from icds.benchmarks.external import make_external_task
from icds.storage import checksum, load_shard, write_json
from tabicl import TabICLRegressor

EXTERNAL = ICDS_ROOT / 'data/external-m4-v2/manifest.json'
EXP = Path(__file__).resolve().parent


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def raw_icds_features(sampler, wid, task):
    """Original units, actual action width, no selector/anchor/duplicate columns.

    Keep all public information: full state, observation if distinct, observed
    driver, visible parameters, their availability flags, and timestep.
    Hidden parameter values are never exposed.
    """
    shard = sampler.shards[wid]
    arrays = load_shard(sampler.root, shard)
    world = json.loads((sampler.root / shard['world']).read_text())
    program = json.loads((sampler.root / shard['program']).read_text())
    rows = np.asarray(task['selected_rows'])
    blocks = [arrays['state'][rows]]
    names = list(program['state_names'])
    if not np.array_equal(arrays['observation'][rows], arrays['state'][rows]):
        blocks.append(arrays['observation'][rows])
        names += ['observation_' + n for n in program['state_names']]
    blocks += [arrays['action'][rows], arrays['exogenous_observed'][rows]]
    names += list(program['action_names']) + ['current_exogenous']
    keys = sorted(world['parameter_scales'])
    values = [world['parameters'][k] if k in world['observed_parameters'] else 0.
              for k in keys]
    masks = [float(k in world['observed_parameters']) for k in keys]
    blocks.append(np.tile(values + masks + [world['dt']], (len(rows), 1)))
    names += keys + ['available_' + k for k in keys] + ['dt_seconds']
    return np.concatenate(blocks, axis=1).astype(np.float32), names


def build_tasks(suite, contexts, support_seeds, query_size):
    entries = []
    if suite == 'icds':
        sampler = IndustrialTaskSampler(MANIFEST, 'meta_val', 960, query_size,
                                        seed=42, target_floor_profile='event_v2',
                                        external_context_mode='observed')
        for wid, task in sampler.fixed_tasks():
            x, names = raw_icds_features(sampler, wid, task)
            entries.append((f'world{wid}-c960-s42', task, x,
                            dict(world=wid, context=960, support_seed=42,
                                 family=sampler.shards[wid]['family'], raw_columns=names)))
    else:
        manifest = json.loads(EXTERNAL.read_text())
        for shard in manifest['shards']:
            if shard['domain'] != 'cstr' or shard['split'] != 'external_val':
                continue
            path = EXTERNAL.parent / shard['path']
            if checksum(path) != shard['sha256']:
                raise ValueError(f'Checksum mismatch: {path}')
            with np.load(path) as arrays:
                for seed in support_seeds:
                    for context in contexts:
                        task = make_external_task(
                            EXTERNAL, shard, context, query_size,
                            support_seed=seed, time_feature='si_seconds')
                        rows = np.asarray(task['selected_rows'])
                        # The only public dynamic quantities are Ca, T and Tc.
                        x = np.c_[arrays['observation'][rows], arrays['action'][rows],
                                  np.full(len(rows), task['feature_spec']['model_dt'])].astype(np.float32)
                        entries.append((f"world{shard['world']}-c{context}-s{seed}", task, x,
                                        dict(world=shard['world'], context=context,
                                             support_seed=seed, family='cstr',
                                             raw_columns=['Ca_mol_per_L', 'T_K', 'Tc_K', 'dt_seconds'])))
    for _, task, x, _ in entries:
        n = task['context_size']; ids = task['group_ids']
        assert len(np.unique(task['selected_rows'])) == len(ids)
        assert not set(ids[:n, 1]) & set(ids[n:, 1])
        assert not set(ids[:n, 2]) & set(ids[n:, 2])
        assert np.isfinite(x).all() and x.shape[0] == task['raw_y'].shape[0]
    if suite == 'cstr':
        for world in {e[3]['world'] for e in entries}:
            groups = [e for e in entries if e[3]['world'] == world]
            query_ids = [e[1]['group_ids'][e[1]['context_size']:] for e in groups]
            assert all(np.array_equal(query_ids[0], q) for q in query_ids)
    return entries


def metrics(pred, task):
    n = task['context_size']
    truth = task['raw_y'][n:]
    if pred.shape != truth.shape or not np.isfinite(pred).all():
        raise ValueError('Prediction shape/nonfinite failure')
    error = pred - truth
    ref = np.asarray(task['target_spec']['reference_scales'])
    return dict(fixed_scale_mse=float(np.mean((error / ref) ** 2)),
                per_target_fixed_scale_mse=np.mean((error / ref) ** 2, axis=0).tolist(),
                per_target_rmse=np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
                per_target_mae=np.mean(np.abs(error), axis=0).tolist(),
                target_names=task['target_spec']['state_names'],
                reference_scales=ref.tolist())


def raw_predict(kind, x, task, device, estimators):
    n = task['context_size']; y = task['raw_y'][:n]
    if kind == 'persistence':
        return persistence_target(task)[n:]
    if kind == 'ridge_raw':
        reg = make_pipeline(StandardScaler(), Ridge(alpha=1e-5, solver='svd'))
        return reg.fit(x[:n].astype(np.float64), y).predict(x[n:].astype(np.float64)).reshape(-1, y.shape[1])
    results = []
    for j in range(y.shape[1]):
        if kind == 'official_raw':
            reg = TabICLRegressor(n_estimators=estimators, model_path=DEFAULT_CHECKPOINT,
                                 allow_auto_download=False, device=str(device), random_state=42)
        elif kind == 'boosting_raw':
            reg = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                                                min_samples_leaf=10, random_state=42)
        else:
            raise ValueError(kind)
        reg.fit(x[:n], y[:, j])
        results.append(reg.predict(x[n:]))
        del reg
    return np.asarray(results).T


def summarize(records):
    result = {}
    for model in sorted({r['model'] for r in records}):
        result[model] = {}
        for context in sorted({r['context'] for r in records}):
            group = [r for r in records if r['model'] == model and r['context'] == context]
            if not group:
                continue
            worlds = sorted({r['world'] for r in group})
            means = [np.mean([r['fixed_scale_mse'] for r in group if r['world'] == w]) for w in worlds]
            result[model][str(context)] = dict(
                macro_fixed_scale_mse=float(np.mean(means)), world_count=len(worlds),
                task_count=len(group), family_macro_fixed_scale_mse={
                    f:float(np.mean([r['fixed_scale_mse'] for r in group if r['family'] == f]))
                    for f in sorted({r['family'] for r in group})})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['icds', 'cstr'], required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--contexts', nargs='+', type=int, default=[128, 512, 960])
    parser.add_argument('--support-seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--query-size', type=int, default=256)
    parser.add_argument('--estimators', type=int, default=8)
    parser.add_argument('--models', nargs='+', default=['official_raw', 'official_typed', 'caq', 'tabicl_ft', 'ridge_raw', 'boosting_raw', 'persistence'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Reuse complete model groups after checking task identities and checkpoint/data hashes')
    args = parser.parse_args()
    if args.out.exists() and not args.resume:
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(4); torch.manual_seed(42)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    entries = build_tasks(args.suite, args.contexts, args.support_seeds, args.query_size)
    data_manifest = Path(MANIFEST) if args.suite == 'icds' else EXTERNAL
    provenance = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                      split='meta_val' if args.suite == 'icds' else 'external_val',
                      manifest=str(data_manifest), manifest_sha256=checksum(data_manifest),
                      checkpoints={str(p):checksum(p) for p in [Path(DEFAULT_CHECKPOINT),CAQ_CKPT,TABICL_FT_CKPT]},
                      source_hashes={str(p):checksum(p) for p in [Path(__file__), EXP/'industrial_adapter.py',
                          EXP/'industrial_task_sampler.py', EXP/'compare_three.py', EXP/'train_industrial.py',
                          ICDS_ROOT/'icds/export.py', ICDS_ROOT/'icds/benchmarks/external.py']},
                      torch=str(torch.__version__), numpy=np.__version__,
                      notes=['No weight updates to neural models.',
                             'Raw target uses physical delta (or next value for ICDS count/binary); fit sees context only.',
                             'All models share row IDs; neural typed inputs retain their training protocol.',
                             'CSTR is PC-Gym equations with an existing custom rollout harness, not official RL scores.',
                             'CSTR external_val has historical use; development transfer test, not an untouched final test.',
                             'CSTR time feature is converted from .05 minutes to 3 seconds; integration truth unchanged.',
                             'Support repetitions are not independent training seeds.'])
    if args.resume:
        old=json.loads((args.out/'protocol.json').read_text())
        for k,v in old['arguments'].items():
            if k != 'resume' and provenance['arguments'][k] != v:
                raise ValueError(f'Resume argument mismatch: {k}')
        assert old['manifest_sha256']==provenance['manifest_sha256']
        assert old['checkpoints']==provenance['checkpoints']
        provenance['previous_protocols']=old.get('previous_protocols',[])+[{
            k:v for k,v in old.items() if k!='previous_protocols'}]
    write_json(args.out/'protocol.json', provenance)
    task_rows = []
    for key,t,x,meta in entries:
        n=t['context_size']
        task_rows.append(dict(task=key, **meta, raw_shape=list(x.shape),
                              raw_x_sha256=array_hash(x), context_rows=np.asarray(t['selected_rows'][:n]).tolist(),
                              query_rows=np.asarray(t['selected_rows'][n:]).tolist(),
                              query_ids_sha256=array_hash(t['group_ids'][n:]),
                              context_episode_ids=np.unique(t['group_ids'][:n,1]).tolist(),
                              query_episode_ids=np.unique(t['group_ids'][n:,1]).tolist()))
    if args.resume:
        assert json.loads((args.out/'tasks.json').read_text())==task_rows
    write_json(args.out/'tasks.json', task_rows)
    records=[]
    completed=set()
    if args.resume:
        previous=[json.loads(line) for line in (args.out/'records.jsonl').read_text().splitlines()]
        write_json(args.out/f'records-before-resume-{time.time_ns()}.json',previous)
        expected_keys={e[0] for e in entries}
        taskmap={e[0]:e[1] for e in entries}
        for kind in args.models:
            group=[r for r in previous if r['model']==kind]
            if len(group)!=len(entries) or {r['task'] for r in group}!=expected_keys:
                continue
            for row in group:
                task=taskmap[row['task']];n=task['context_size']
                with np.load(args.out/f"{kind}-{row['task']}.npz") as saved:
                    np.testing.assert_array_equal(saved['query_ids'],task['group_ids'][n:])
                    np.testing.assert_array_equal(saved['actual'],task['raw_y'][n:])
                    assert np.isclose(metrics(saved['predicted'],task)['fixed_scale_mse'],row['fixed_scale_mse'],rtol=1e-12)
            records.extend(group);completed.add(kind)
        with (args.out/'records.jsonl').open('w') as stream:
            for row in records:stream.write(json.dumps(row)+'\n')
        print('RESUMED_COMPLETE_MODELS',sorted(completed),flush=True)
    for kind in args.models:
        if kind in completed:
            continue
        model=None
        if kind in {'caq','tabicl_ft'}:
            model=load(CAQ_CKPT if kind=='caq' else TABICL_FT_CKPT,device).eval()
        elif kind=='official_typed':
            model=OfficialTabICLRegressorAdapter(DEFAULT_CHECKPOINT,device,args.estimators)
        for key, task, x, meta in entries:
            started=time.perf_counter()
            if model is None:
                pred=raw_predict(kind,x,task,device,args.estimators)
            else:
                with torch.no_grad():
                    scaled=predict_task(model,task,device).cpu().numpy()
                pred=scaled*task['scaler']['target_scale']+task['scaler']['target_mean']
            row=dict(model=kind,task=key,**meta,**metrics(pred,task),seconds=time.perf_counter()-started)
            records.append(row)
            np.savez_compressed(args.out/f'{kind}-{key}.npz',predicted=pred,
                                actual=task['raw_y'][task['context_size']:],
                                query_ids=task['group_ids'][task['context_size']:])
            with (args.out/'records.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        del model;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
        write_json(args.out/'report.json',dict(status='running',summary=summarize(records),records=records))
    expected=len(entries)*len(args.models)
    if len(records)!=expected:raise RuntimeError('Incomplete evaluation')
    write_json(args.out/'report.json',dict(status='completed',expected_records=expected,
                                          summary=summarize(records),records=records))
    print('COMPLETED',json.dumps(summarize(records)),flush=True)


if __name__=='__main__':
    main()
