"""Verify saved evaluations, row identity, upstream provenance and CSTR truth."""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from eval_raw_and_cstr import EXP, EXTERNAL, ICDS_ROOT, array_hash, build_tasks, metrics
from icds.benchmarks.external import pc_models
from icds.storage import checksum, write_json


def verify():
    root = EXP / 'checkpoints/raw-input-cstr-audit'
    evidence = {}
    manifest = json.loads(EXTERNAL.read_text())
    source = ICDS_ROOT / 'external_sources/pc-gym/src/pcgym/model_classes.py'
    assert checksum(source) == manifest['provenance']['pc-gym']['source_hashes']['src/pcgym/model_classes.py']
    errors=[]
    for shard in manifest['shards']:
        if shard['domain'] != 'cstr' or shard['split'] != 'external_val':
            continue
        path=EXTERNAL.parent/shard['path']
        assert checksum(path)==shard['sha256']
        model=pc_models().cstr(int_method='numpy', **shard['hidden_generator_parameters'])
        with np.load(path) as arrays:
            for row in [0,768,1536]:
                x,u=arrays['observation'][row],arrays['action'][row]
                result=solve_ivp(lambda time,state:model(state,u),(0,.05),x,
                                 method='Radau',rtol=1e-11,atol=1e-12)
                assert result.success
                errors.append(float(np.max(np.abs((result.y[:,-1]-arrays['next_observation'][row])/shard['spec']['references']))))
    assert max(errors)<1e-4
    evidence['cstr_reference_check']=dict(upstream_commit=manifest['provenance']['pc-gym']['commit'],
                                         upstream_source_sha256=checksum(source),
                                         checks=len(errors), max_normalized_error=max(errors))
    for suite in ['icds','cstr']:
        folder=root/suite
        report=json.loads((folder/'report.json').read_text())
        protocol=json.loads((folder/'protocol.json').read_text())
        args=protocol['arguments']
        assert report['status']=='completed'
        assert checksum(protocol['manifest'])==protocol['manifest_sha256']
        for path,digest in protocol['source_hashes'].items():
            assert checksum(path)==digest, path
        for path,digest in protocol['checkpoints'].items():
            assert checksum(path)==digest, path
        entries=build_tasks(suite,args['contexts'],args['support_seeds'],args['query_size'])
        taskmap={key:task for key,task,x,meta in entries}
        saved_tasks=json.loads((folder/'tasks.json').read_text())
        assert len(saved_tasks)==len(entries)
        saved_tasks={t['task']:t for t in saved_tasks}
        assert set(saved_tasks)==set(taskmap)
        for key,task,x,meta in entries:
            n=task['context_size']; saved=saved_tasks[key]
            assert saved['raw_x_sha256']==array_hash(x)
            assert saved['raw_shape']==list(x.shape)
            assert saved['raw_columns']==meta['raw_columns']
            assert saved['context_rows']==np.asarray(task['selected_rows'][:n]).tolist()
            assert saved['query_rows']==np.asarray(task['selected_rows'][n:]).tolist()
            assert saved['query_ids_sha256']==array_hash(task['group_ids'][n:])
        records=report['records']
        assert len(records)==report['expected_records']==len(entries)*len(args['models'])
        assert {(r['model'],r['task']) for r in records}=={(m,k) for m in args['models'] for k in taskmap}
        for row in records:
            task=taskmap[row['task']];n=task['context_size']
            with np.load(folder/f"{row['model']}-{row['task']}.npz") as pred:
                np.testing.assert_array_equal(pred['query_ids'],task['group_ids'][n:])
                np.testing.assert_array_equal(pred['actual'],task['raw_y'][n:])
                recomputed=metrics(pred['predicted'],task)
                for field in ['fixed_scale_mse','per_target_fixed_scale_mse','per_target_rmse','per_target_mae']:
                    np.testing.assert_allclose(recomputed[field],row[field],rtol=1e-12,atol=1e-12)
        # Independently recompute report aggregates, not only individual records.
        for model,contexts in report['summary'].items():
            for context,aggregate in contexts.items():
                rows=[r for r in records if r['model']==model and r['context']==int(context)]
                worlds=sorted({r['world'] for r in rows})
                world_means=[np.mean([r['fixed_scale_mse'] for r in rows if r['world']==w]) for w in worlds]
                assert aggregate['task_count']==len(rows)
                assert aggregate['world_count']==len(worlds)
                np.testing.assert_allclose(aggregate['macro_fixed_scale_mse'],np.mean(world_means),rtol=1e-12)
                for family,value in aggregate['family_macro_fixed_scale_mse'].items():
                    np.testing.assert_allclose(value,np.mean([r['fixed_scale_mse'] for r in rows if r['family']==family]),rtol=1e-12)
        evidence[suite]=dict(tasks=len(entries),records=len(records), predictions_recomputed=True,
                              exact_query_and_label_match=True,source_and_checkpoint_hashes_verified=True,
                              raw_input_and_row_hashes_verified=True,all_metrics_and_aggregates_verified=True)
    previous=json.loads((EXP/'checkpoints/compare-three/meta_val-hr0.json').read_text())
    current=json.loads((root/'icds/report.json').read_text())['summary']
    differences={}
    for now,old in [('caq','caq'),('tabicl_ft','tabicl_ft'),('official_typed','tabicl_official_zero_shot')]:
        a=current[now]['960']['macro_fixed_scale_mse']
        b=previous[old]['prediction']['macro_fixed_scale_mse']
        differences[now]=dict(current=a,previous=b,absolute_difference=abs(a-b))
        assert np.isclose(a,b,rtol=1e-3,atol=1e-6), differences[now]
    evidence['legacy_reproduction']=differences
    evidence['status']='passed'
    write_json(root/'verification.json',evidence)
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':
    verify()
