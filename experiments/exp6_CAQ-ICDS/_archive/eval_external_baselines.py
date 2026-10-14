"""Fit conventional models ONLY on the same external support, never query labels."""
import argparse
import json
import warnings
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from icds.benchmarks.external import make_external_task
from icds.storage import checksum, write_json


def fit_typed_baselines(task, persistence_residual=False, scope='next_observation'):
    """Each scalar fit receives exactly the public feature table used by CAQ."""
    from industrial_adapter import to_target_batch, residual_baseline_scaled
    X,A,support,_=to_target_batch(task,'cpu',target_selector=True,target_anchor=True,
                                absolute_state=True,persistence_residual=persistence_residual,
                                persistence_residual_scope=scope)
    n=task['context_size'];outputs={name:[] for name in ['ridge','mlp','boosting']}
    correction=residual_baseline_scaled(task,scope)[n:] if persistence_residual else np.zeros_like(task['raw_y'][n:])
    for j in range(len(support)):
        features=np.c_[X[j].numpy(),A[j].numpy()];target=support[j].numpy()
        design=np.c_[features,np.ones(len(features))]
        coef=np.linalg.solve(design[:n].T@design[:n]+1e-5*np.eye(design.shape[1]),design[:n].T@target)
        mlp=MLPRegressor(hidden_layer_sizes=(64,64),max_iter=200,random_state=42,early_stopping=False)
        mlp.fit(features[:n],target)
        tree=HistGradientBoostingRegressor(max_iter=60,max_leaf_nodes=15,min_samples_leaf=5,random_state=42)
        tree.fit(features[:n],target)
        for name,scaled in [('ridge',design[n:]@coef),('mlp',mlp.predict(features[n:])),('boosting',tree.predict(features[n:]))]:
            outputs[name].append((scaled+correction[:,j])*task['scaler']['target_scale'][j]+task['scaler']['target_mean'][j])
    return {name:np.array(values).T for name,values in outputs.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--split',default='external_val',choices=['external_val','external_test']);p.add_argument('--context-sizes',nargs='+',type=int,default=[32,128,512,2048]);p.add_argument('--history-study',action='store_true');p.add_argument('--typed-public-inputs',action='store_true');p.add_argument('--persistence-residual',action='store_true');p.add_argument('--persistence-residual-scope',choices=['all','next_observation'],default='next_observation');p.add_argument('--support-seeds',nargs='+',type=int);p.add_argument('--time-feature',choices=['source','si_seconds'],default='source');p.add_argument('--out',required=True);args=p.parse_args()
    if args.persistence_residual and not args.typed_public_inputs:p.error('Residual baseline requires typed public inputs')
    meta=json.loads(Path(args.manifest).read_text());records=[]
    warnings.filterwarnings('ignore',category=ConvergenceWarning)
    for shard in meta['shards']:
        if shard['split']!=args.split:continue
        assert checksum(Path(args.manifest).parent/shard['path'])==shard['sha256']
        histories=([1,8,30] if args.history_study else [1,8]) if shard['domain']=='industrial_benchmark' else [1]
        for H in histories:
            for support_seed in args.support_seeds or [None]:
                for c in args.context_sizes:
                    task=make_external_task(args.manifest,shard,c,64,history=H,eligible_history=30 if args.history_study and shard["domain"]=="industrial_benchmark" else None,time_feature=args.time_feature,support_seed=support_seed)
                    if args.typed_public_inputs:
                        predictions=fit_typed_baselines(task,args.persistence_residual,args.persistence_residual_scope)
                        actual=task['raw_y'][c:];ref=np.array(shard['spec']['references'])
                        predictions['persistence']=task['current_observation'][c:] if task['target_mode']=='next_observation' else np.zeros_like(actual)
                        for model,pred in predictions.items():
                            records.append({'model':model,'support_seed':support_seed,'domain':shard['domain'],'world':shard['world'],'context':c,'history':H,'nrmse':float(np.sqrt(np.mean(((pred-actual)/ref)**2))), 'per_target_rmse':np.sqrt(np.mean((pred-actual)**2,0)).tolist(),'query_ids':task['group_ids'][c:].tolist()})
                        print(json.dumps({'domain':shard['domain'],'world':shard['world'],'context':c,'history':H,'typed_public_inputs':True}),flush=True)
                        continue
                    X=np.c_[task['X_state'][0],task['X_action'][0]];y=task['Y'][0];actual=task['raw_y'][c:];ref=np.array(shard['spec']['references'])
                    design=np.c_[X,np.ones(len(X))];coef=np.linalg.solve(design[:c].T@design[:c]+1e-5*np.eye(design.shape[1]),design[:c].T@y[:c])
                    predictions={'persistence':task['current_observation'][c:] if task['target_mode']=='next_observation' else np.zeros_like(actual),'ridge':(design[c:]@coef)*task['scaler']['target_scale']+task['scaler']['target_mean']}
                    mlp=MLPRegressor(hidden_layer_sizes=(64,64),max_iter=200,random_state=42,early_stopping=False)
                    mlp.fit(X[:c],y[:c]);pred=mlp.predict(X[c:]);pred=pred.reshape(len(actual),-1)
                    predictions['mlp']=pred*task['scaler']['target_scale']+task['scaler']['target_mean']
                    outputs=[]
                    for target in range(y.shape[1]):
                        tree=HistGradientBoostingRegressor(max_iter=60,max_leaf_nodes=15,min_samples_leaf=5,random_state=42)
                        tree.fit(X[:c],y[:c,target]);outputs.append(tree.predict(X[c:]))
                    predictions['boosting']=np.array(outputs).T*task['scaler']['target_scale']+task['scaler']['target_mean']
                    for model,pred in predictions.items():
                        records.append({'model':model,'support_seed':support_seed,'domain':shard['domain'],'world':shard['world'],'context':c,'history':H,'nrmse':float(np.sqrt(np.mean(((pred-actual)/ref)**2))), 'per_target_rmse':np.sqrt(np.mean((pred-actual)**2,0)).tolist(),'query_ids':task['group_ids'][c:].tolist()})
                    print(json.dumps({'domain':shard['domain'],'world':shard['world'],'context':c,'history':H}),flush=True)
    for domain,world,H in {(r['domain'],r['world'],r['history']) for r in records}:
        group=[r for r in records if (r['domain'],r['world'],r['history'])==(domain,world,H)]
        if any(r['query_ids']!=group[0]['query_ids'] for r in group):
            raise ValueError('Baseline support repetitions/context sizes changed query identities')
    write_json(args.out,{'records':records,'split':args.split,'time_feature':args.time_feature,'support_seeds':args.support_seeds,'manifest_sha256':checksum(args.manifest),'typed_public_inputs':args.typed_public_inputs,'persistence_residual':args.persistence_residual,'persistence_residual_scope':args.persistence_residual_scope,'parameters':{'mlp':'64/64, Adam, 200 epochs, seed42; independent scalar fits for typed protocol','boosting':'HistGB 60 iter,15 leaves,minleaf5,seed42','ridge':'1e-5 penalty; support-only scaled targets'},'limitations':['No baseline hyperparameters selected on query labels','IB histories use common30 eligibility only under --history-study; default H1/H8 have separate pools; compare neural default to H8','Legacy default omits appended public absolute-state/target-anchor representations; typed protocol matches CAQ public scalar feature tables']})

if __name__=='__main__':main()
