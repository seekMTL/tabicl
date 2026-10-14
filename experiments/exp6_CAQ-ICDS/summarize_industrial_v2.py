"""Verify completed matched runs and persist best/final/scenario comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from industrial_adapter import ICDS_ROOT
from icds.storage import checksum, write_json

EXP=Path(__file__).resolve().parent


def summarize(phase):
    root=EXP/'checkpoints/industrial-v2'
    expected=200 if phase=='smoke' else 30000
    runs={};curves={};settings={}
    for kind in ['caq','tabicl']:
        path=root/(f'smoke-{kind}-seed42' if phase=='smoke' else f'{kind}-A4-seed42-hr0')
        s=json.loads((path/'summary.json').read_text());a=json.loads((path/'args.json').read_text())
        assert s['status']=='completed' and s['steps']==expected
        assert s['successful_updates']==expected and s['skipped_steps']==0 and s['rolled_back']==0
        assert s['industrial_updates']==int(.8*expected) and s['scm_updates']==int(.2*expected)
        assert s['manifest_sha256']==checksum(ICDS_ROOT/'data/industrial-v2/manifest.json')
        assert s['base_checkpoint_sha256']==checksum(a['checkpoint'])
        history=[json.loads(line) for line in (path/'history.jsonl').read_text().splitlines()]
        assert [r['step'] for r in history]==list(range(1,expected+1))
        assert all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm_before_clip']) for r in history)
        before=json.loads((path/'before.json').read_text())
        after=json.loads((path/'after.json').read_text())
        evals=json.loads((path/'evaluations.json').read_text())
        assert evals[-1]['step']==expected
        assert all(len(e['validation']['worlds'])==52 for e in evals)
        candidates=[(0,before['validation'])]+[(e['step'],e['validation']) for e in evals]
        best_step,best=min(candidates,key=lambda e:e[1]['macro_fixed_scale_mse'])
        np.testing.assert_allclose(best['macro_fixed_scale_mse'],s['best_validation_fixed_scale_mse'],rtol=1e-12)
        np.testing.assert_allclose(after['validation']['macro_fixed_scale_mse'],s['validation_fixed_scale_mse_after'],rtol=1e-12)
        assert s['best_step']==best_step
        for filename,step in [('best.pt',best_step),('final.pt',expected)]:
            saved=torch.load(path/filename,map_location='cpu',weights_only=True)
            assert saved['step']==step and saved['args']==a
            del saved
        curves[kind]=dict(steps=[step for step,_ in candidates],
                          validation_mse=[v['macro_fixed_scale_mse'] for _,v in candidates])
        settings[kind]={k:v for k,v in a.items() if k not in ['out','model_kind']}
        runs[kind]=dict(best_step=best_step,best_mse=best['macro_fixed_scale_mse'],
                       final_mse=after['validation']['macro_fixed_scale_mse'],
                       before_mse=before['validation']['macro_fixed_scale_mse'],
                       best_scenarios=best['scenario_macro_fixed_scale_mse'],
                       final_scenarios=after['validation']['scenario_macro_fixed_scale_mse'],
                       best_families=best['family_macro_fixed_scale_mse'],
                       final_families=after['validation']['family_macro_fixed_scale_mse'],
                       peak_allocated_bytes=s['gpu_peak_memory_bytes'],peak_reserved_bytes=s['gpu_peak_reserved_bytes'],
                       elapsed_seconds=s['elapsed_seconds'],wall_elapsed_seconds=s['wall_elapsed_seconds'],
                       industrial_task_sequence_sha256=s['industrial_task_sequence_sha256'],
                       successful_updates=s['successful_updates'],scm_mse_before=s['scm_mse_before'],scm_mse_after=s['scm_mse_after'])
    assert settings['caq']==settings['tabicl']
    assert runs['caq']['industrial_task_sequence_sha256']==runs['tabicl']['industrial_task_sequence_sha256']
    comparisons={scenario:dict(caq=runs['caq']['best_scenarios'][scenario],
                              tabicl_ft=runs['tabicl']['best_scenarios'][scenario],
                              caq_relative_reduction=1-runs['caq']['best_scenarios'][scenario]/runs['tabicl']['best_scenarios'][scenario])
                 for scenario in runs['caq']['best_scenarios']}
    report=dict(status='verified_completed',phase=phase,runs=runs,best_scenario_comparison=comparisons,
                caq_best_relative_reduction=1-runs['caq']['best_mse']/runs['tabicl']['best_mse'],
                metric='World/target macro MSE normalized by unchanged public physical reference scales',
                limitations=['One training seed; internal held-out parameter worlds, not unseen prototype transfer.',
                             'Do not compare old/new overall scores directly: prototype mix and held-out worlds differ.',
                             'Short-run results are only integration checks, not final performance evidence.'])
    write_json(root/f'{phase}-comparison.json',report)
    if phase=='formal':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
        for kind,color in [('caq','#d25e39'),('tabicl','#5278b7')]:
            ax.plot(curves[kind]['steps'],curves[kind]['validation_mse'],label=kind,color=color,marker='.')
        ax.set_yscale('log');ax.set_xlabel('Training step');ax.set_ylabel('Reference-normalized macro MSE')
        ax.set_title('Industrial-v2 | seed42 | context960/query256');ax.legend();ax.grid(alpha=.2)
        fig.savefig(root/'formal-validation.png',dpi=180)
        lines=['# Industrial-v2 seed42 已完成结果','',
               '指标为固定物理参考尺度归一化后的world/target宏平均MSE，越低越好。两个模型使用相同任务序列、相同官方初始化检查点和30k尝试步数。','',
               '| 模型 | 初始 | 最佳 | 最佳步 | 最后 | 训练及评估耗时秒 |',
               '|---|---:|---:|---:|---:|---:|']
        for kind,r in runs.items():
            lines.append(f"| {kind} | {r['before_mse']:.6g} | {r['best_mse']:.6g} | {r['best_step']} | {r['final_mse']:.6g} | {r['elapsed_seconds']:.1f} |")
        lines+=['','| 原型 | CAQ best | FT best | CAQ相对误差降低 |','|---|---:|---:|---:|']
        for name,r in comparisons.items():
            lines.append(f"| {name} | {r['caq']:.6g} | {r['tabicl_ft']:.6g} | {r['caq_relative_reduction']:.1%} |")
        lines+=['','模型best分别按统一验证主指标选取。单seed结果不能宣称统计稳定优势。新旧整体分数的数据配比和验证world不同，不能直接作为扩充数据收益。',
                '完整曲线、物理单位误差及目标名称见各模型evaluations.json/after.json；当前没有在meta_test上做性能选型。','']
        (root/'formal-results.md').write_text('\n'.join(lines))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['smoke','formal'],default='formal')
    summarize(parser.parse_args().phase)
