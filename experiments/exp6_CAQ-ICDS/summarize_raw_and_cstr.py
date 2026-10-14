"""Summarize completed development evaluations; worlds are bootstrap units."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from industrial_adapter import ICDS_ROOT  # Register the frozen generator package path.
from icds.storage import write_json


def main():
    root=Path(__file__).resolve().parent/'checkpoints/raw-input-cstr-audit'
    reports={s:json.loads((root/s/'report.json').read_text()) for s in ['icds','cstr']}
    assert all(r['status']=='completed' for r in reports.values())
    records=reports['cstr']['records']
    comparisons=[]
    rng=np.random.default_rng(42)
    for context in [128,512,960]:
        groups={}
        for m in sorted({r['model'] for r in records}):
            group=[r for r in records if r['model']==m and r['context']==context]
            worlds=sorted({r['world'] for r in group})
            groups[m]=np.asarray([np.mean([r['fixed_scale_mse'] for r in group if r['world']==w]) for w in worlds])
        index=rng.integers(0,len(worlds),size=(50000,len(worlds)))
        for base in ['official_raw','official_typed','tabicl_ft','ridge_raw','boosting_raw']:
            diff=groups['caq']-groups[base]
            comparisons.append(dict(context=context,comparison='caq - '+base,
                                    mean_difference=float(diff.mean()),
                                    world_bootstrap_95_ci=np.quantile(diff[index].mean(1),[.025,.975]).tolist(),
                                    worlds_won=int((diff<0).sum()),world_count=len(worlds),
                                    relative_mse_reduction=float(1-groups['caq'].mean()/groups[base].mean())))
    physical=[]
    for model in ['caq','tabicl_ft','official_raw','official_typed']:
        rows=[r for r in records if r['model']==model and r['context']==960]
        physical.append(dict(model=model,context=960,
                             mean_task_rmse=np.mean([r['per_target_rmse'] for r in rows],axis=0).tolist(),
                             units=['mol/L','K']))
    write_json(root/'analysis.json',dict(comparisons=comparisons,physical_rmse=physical,
                                       note='Paired bootstrap over six parameter worlds after averaging support draws; one training seed and one external mechanism, not broad significance evidence.'))
    names=['official_raw','official_typed','tabicl_ft','caq','ridge_raw','boosting_raw','persistence']
    labels={'official_raw':'Official raw','official_typed':'Official typed','tabicl_ft':'TabICL-FT',
            'caq':'CAQ','ridge_raw':'Ridge','boosting_raw':'HistGB','persistence':'Persistence'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    values=[reports['icds']['summary'][m]['960']['macro_fixed_scale_mse'] for m in names]
    axes[0].bar(range(len(names)),values,color=['#888','#bbb','#5278b7','#d25e39','#969256','#788773','#bbb'])
    axes[0].set_xticks(range(len(names)),[labels[m] for m in names],rotation=35,ha='right')
    axes[0].set_yscale('log');axes[0].set_ylabel('Fixed-scale MSE (lower is better)')
    axes[0].set_title('ICDS meta_val | context 960 | 26 worlds')
    for m in names[:5]:
        axes[1].plot([128,512,960],[reports['cstr']['summary'][m][str(c)]['macro_fixed_scale_mse'] for c in [128,512,960]],marker='o',label=labels[m])
    axes[1].set_yscale('log');axes[1].set_xticks([128,512,960]);axes[1].set_xlabel('Context rows')
    axes[1].set_ylabel('Fixed-scale MSE (lower is better)')
    axes[1].set_title('CSTR development | 6 worlds × 3 support draws')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.savefig(root/'comparison.png',dpi=180)
    print(json.dumps(dict(comparisons=comparisons,physical_rmse=physical),indent=2))


if __name__=='__main__':
    main()
