"""Repeated support draws, fixed query and episode partition across context sizes."""
import argparse
import torch
from eval_external import evaluate_external
from train_industrial import build_model
from icds.storage import checksum, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--split',choices=['external_val','external_test'],default='external_val')
    p.add_argument('--support-seeds',type=int,nargs='+',default=[101,102,103])
    p.add_argument('--context-sizes',type=int,nargs='+',default=[32,128,512,2048])
    p.add_argument('--time-feature',choices=['source','si_seconds'],default='source')
    args=p.parse_args()
    torch.set_num_threads(4)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=build_model(state['base_checkpoint'],args.device,state['args'].get('model_kind','caq'))
    model.load_state_dict(state['model'],strict=True)
    for flag in ['target_selector','target_anchor','absolute_state','persistence_residual']:
        setattr(model,flag,state['args'].get(flag,False))
    model.persistence_residual_scope=state['args'].get('persistence_residual_scope','all')
    records=[]
    for seed in args.support_seeds:
        report=evaluate_external(model,args.manifest,args.split,args.context_sizes,args.device,pairs=False,support_seed=seed,target_chunk_size=1,time_feature=args.time_feature)
        records.extend(dict(row,support_seed=seed) for row in report['records'])
        print({'support_seed':seed,'domain_macro_nrmse':report['domain_macro_nrmse']},flush=True)
    for domain,world in {(r['domain'],r['world']) for r in records}:
        group=[r for r in records if (r['domain'],r['world'])==(domain,world)]
        if any(r['query_ids']!=group[0]['query_ids'] for r in group):
            raise ValueError('Repeated-support query mismatch')
    write_json(args.out,dict(records=records,checkpoint_sha256=checksum(args.checkpoint),
                            manifest_sha256=checksum(args.manifest),split=args.split,time_feature=args.time_feature,
                            protocol='Three support permutations; fixed query and episode partition; nested context within each permutation',
                            limitations=['Support repetitions are not independent worlds or training seeds','IB next public observation, continuous PC state delta']))


if __name__=='__main__':
    main()
