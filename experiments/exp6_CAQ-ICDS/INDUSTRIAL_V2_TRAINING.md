# industrial-v2：seed42正式训练记录

日期：2026-10-09。本文记录新版接入与运行协议；最终性能以完成后的comparison报告为准。

## 数据、目标与对照

- 新manifest：`/home/lizitao/project/Industrial_synthetic_data/data/industrial-v2/manifest.json`，generator_version=0.3.0。
- 512个world、4,096条episode、1,834,294行、78个program，8类原型；训练/验证/测试408/52/52个world。
- CAQ、TabICL-FT均从官方`tabicl-regressor-v2-20260212.ckpt`重新初始化。不是旧工业模型续训。
- CAQ的action只进入专用causal通道；FT把相同action作为普通输入列。两者均训练列/行模块，冻结ICL predictor，CAQ另训练动作模块。
- full-state hr0：连续变量预测next_state−state，count/binary预测下一值。训练的配对query仅组织同源不同动作行，A4仍对所有query行计算普通MSE，不加因果效应损失。
- 保留旧正式hr0的采样方式：context取事实branch0，训练query取事实branch0与第一个替代动作branch1；branch2/3已存储，但此次配对协议不抽取它们。验证query为事实行。不能把1,834,294条存储行说成全部逐行参与过本轮训练。
- context/query来自同一设备的不同episode，source不交叉；meta_train/meta_val/meta_test的world互不重合。
- 工业任务、目标和行号均由seed与step确定。保存工业任务序列SHA256，用于证明两模型接收相同任务。

## 单位适配

新增`--input-scale-profile industrial_v2`。只改变target-anchor、absolute-state及absolute外部驱动的固定输入编码。旧checkpoint默认legacy，其旧单位编码不变。

| 新单位 | 输入除数 | 依据 |
|---|---:|---|
| m | 1 | 以1m表示位移输入 |
| m/s | 5 | 以5m/s表示速度输入 |
| N | 50 | 公开力命令范围最高50N |
| A | 50 | 公开电压上限24V、电阻下限0.5Ω对应约48A尺度 |
| rad/s | 500 | 公开电压24V、转矩/反电动势常数下限0.05对应约480rad/s驱动尺度 |

这些是固定工程输入尺度，不是数据逐world统计，不依赖隐藏设备参数或query标签，也没有截断数值。K/J/m3/count/s等已存在单位及磨损尺度保留。未知新单位在新profile下会报错，不再静默除以1。

生成器的dynamic/target标准化仍只由support决定；所有`target_reference`保持原值。主指标是每目标按原有物理参考尺度归一化后的MSE，再按world/target宏平均，不是物理单位MSE。

评估增加8原型和具体机制的分解、目标名称/单位/参考尺度。另报原型等权平均作为诊断，best仍按原有world宏平均主指标选择。原型world数不同，因此两种聚合并不相同。

## 检查与运行设置

`preflight_industrial_v2.py`已检查全部512个world、1,024个普通/配对CPU任务，重算标签与support标准化，核对参考尺度和world/episode/source隔离。两模型在8类原型及6种产线布局上完成训练尺寸forward/backward、AdamW更新、SCM replay和全目标eval；冻结ICL参数保持不变。机械和电机的三个目标分别检查。

检查结果：`checkpoints/industrial-v2-preflight.json`；日志：`Logs/industrial-v2-preflight.log`。

- context=960、query=256、targets_per_task=1；typed输入最大135列，FT另拼8列action。
- 每个world的四条context episode共有1,024条事实行。960可以不重复取样；当前协议无法使用context=1500，GPU容量不是唯一限制。
- 单模型preflight峰值分配：CAQ约7.30GiB、FT约7.46GiB；峰值预留约8.31/8.35GiB。
- 第一次200步并发短跑两模型均完成，0跳步/回滚，任务hash相同；但其缓存峰值预留分别约10.65/12.24GiB，接近总显存边界。因此给每个进程设置45% allocator预算，再做同设置200步短跑后正式启动。首轮短跑保留为`smoke-uncapped-{caq,tabicl}-seed42`。
- GPU0已有任务占用约19GiB，只剩约4.6GiB；因此先在GPU1进行两模型并发短跑，确认共同运行余量后正式启动。
- 短跑每模型200步，100步评估，cosine周期仍为30,000步。用于验证联调/并发，不用于判定最终性能。
- 正式每模型30,000步，24,000个工业任务、6,000个SCM replay任务；lr=1e-4，cosine至5e-6，梯度裁剪1，AdamW weight_decay=1e-4，与旧正式设置一致。
- 每1,000步在全部52个meta_val world上评估；固定8原型训练probe。暂不做paired验证，亦不启动seed43/44。
- 工业任务context/query固定960/256；SCM replay遵循原合成采样器，在约1,216行总长度内随机取自身support/query分界。
- 不使用自动混合精度，保持既有float32训练协议。记录成功更新数、跳过总数、回滚总数、峰值分配/预留显存和实际耗时。
- 两进程共享GPU1，耗时反映并发条件，不应直接用来宣称独占GPU下的模型效率差异。
- 修正病态保护的计数：连续坏批次与累计坏批次分开；参数回滚后清除可能污染的Adam状态。没有坏批次时不改变优化路径。

## 明确启动命令

Python：`/home/lizitao/software/miniconda3/envs/tabicl/bin/python`。

两个tmux窗口：`mytmux1:icds-v2-caq`、`mytmux1:icds-v2-ft`；GPU均为cuda:1，输出通过tee同时显示和保存。

最终候选200步短跑已通过：两模型各200次成功更新、160工业任务＋40 SCM任务、0跳过、0回滚，任务序列hash完全一致。45%预算下峰值预留约10.57/10.54GiB，CPU/GPU检查及正式启动门槛通过。

正式训练已于2026-10-09启动：CAQ PID=2664909，FT PID=2664943。完成监控窗口`mytmux1:icds-v2-watch`绑定这两个PID及其进程启动标识，不会把PID复用或失踪当成仍在训练。监控只在两进程正常结束且完成记录存在后调用汇总核验；异常停止会写失败状态。

监控输出：`Logs/industrial-v2/watch.log`；当前状态：`checkpoints/industrial-v2/progress.json`；最终核验状态：`checkpoints/industrial-v2/completion-audit.json`。正式结果仍在训练中，不能用短跑结果代替30k结果。

```bash
bash /home/lizitao/project/tabicl/experiments/exp6_CAQ-ICDS/run_industrial_v2_seed42.sh smoke caq
bash /home/lizitao/project/tabicl/experiments/exp6_CAQ-ICDS/run_industrial_v2_seed42.sh smoke tabicl
# 两项短跑通过后：
bash /home/lizitao/project/tabicl/experiments/exp6_CAQ-ICDS/run_industrial_v2_seed42.sh formal caq
bash /home/lizitao/project/tabicl/experiments/exp6_CAQ-ICDS/run_industrial_v2_seed42.sh formal tabicl
```

实际Python命令由启动脚本显示，正式命令为：

```bash
/home/lizitao/software/miniconda3/envs/tabicl/bin/python -u /home/lizitao/project/tabicl/experiments/exp6_CAQ-ICDS/train_industrial.py \
  --manifest /home/lizitao/project/Industrial_synthetic_data/data/industrial-v2/manifest.json \
  --checkpoint /home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt \
  --out /home/lizitao/project/tabicl/experiments/exp6_CAQ-ICDS/checkpoints/industrial-v2/caq-A4-seed42-hr0 \
  --device cuda:1 --cuda-memory-fraction 0.45 --model-kind caq --prior A4 --steps 30000 --schedule-steps 30000 \
  --context-size 960 --query-size 256 --targets-per-task 1 --replay-ratio 0.2 \
  --lr 1e-4 --lr-schedule cosine --eval-interval 1000 --seed 42 --paired-query \
  --industrial-sampling-protocol factual --information-track full_state \
  --target-selector --target-anchor --absolute-state --input-scale-profile industrial_v2 \
  --persistence-residual --persistence-residual-scope next_observation \
  --external-context-mode observed --target-floor-profile event_v2 \
  --history-ratio 0 --skip-paired-eval --threads 4
```

FT仅将`--model-kind caq`改为`tabicl`，out改为`.../tabicl-A4-seed42-hr0`。不使用`--resume`。

- 正式日志：`Logs/industrial-v2/{caq,tabicl}-A4-seed42-hr0.log`。
- 正式模型：`checkpoints/industrial-v2/{caq,tabicl}-A4-seed42-hr0/{best.pt,final.pt}`。
- 短跑各有独立`smoke-{caq,tabicl}-seed42`目录，不覆盖旧实验。
- 只保存源码hash、依赖、参数和原始记录，不复制整套源码进checkpoint。

## 完成后的验证与比较

```bash
/home/lizitao/software/miniconda3/envs/tabicl/bin/python experiments/exp6_CAQ-ICDS/summarize_industrial_v2.py --phase formal
```

检查两模型正常完成30k、日志步数完整、实际更新/跳过/回滚数、best/final checkpoint及验证选择一致、任务序列hash相同。输出`checkpoints/industrial-v2/formal-comparison.json`、`formal-results.md`、`formal-validation.png`，包含最佳与最后值、曲线和8原型结果。

新旧数据的原型配比和验证world均不同，不能直接比较两个整体分数来声称扩充数据有效。训练结束前不填写新版最终MSE；单seed仍属于初步结果。

## 2026-10-11：锁定最佳模型的正式测试

使用独立入口`test_industrial_v2.py`，不直接运行硬编码旧pilot的`compare_three.py`。
CAQ锁定19,000步best.pt，FT锁定15,000步best.pt。仅在meta_test的52个设备上比较CAQ、FT和官方TabICLRegressor；不再使用测试结果选择checkpoint。

- context=960、query=256、seed=42固定任务、full_state/hr0、事实query、全部目标。
- 测试设备与训练/验证设备不重合，context/query episode和source不重合。
- 三模型均使用industrial_v2绝对输入单位编码，原有目标参考尺度不变。
- 官方基线保留官方8成员集成、预处理、常量列过滤；每目标fit 960条support后predict 256条query，不更新网络参数。不是零context。
- CAQ动作仍在专用causal通道；FT和官方获得相同动作信息，以普通特征列输入。
- 仅补一步预测、8原型分解和逐目标物理单位RMSE/MAE。此次不补paired/FP或SCM，不额外训练。
- 三模型依次使用GPU1，进程显存预算55%；官方接口预检只检查最大宽度输入的有限预测和输出形状，不计算误差或调参。
- `industrial_common.evaluate`只新增可选进度回调；无回调时旧评估不变。

tmux窗口`mytmux1:icds-v2-test`执行：

```bash
bash /home/lizitao/project/tabicl/experiments/exp6_CAQ-ICDS/run_industrial_v2_test.sh preflight && bash /home/lizitao/project/tabicl/experiments/exp6_CAQ-ICDS/run_industrial_v2_test.sh formal
```

Python仍为`/home/lizitao/software/miniconda3/envs/tabicl/bin/python`。
正式输出目录`checkpoints/industrial-v2/meta-test-seed42-context960-query256-official8`：
`status.json`记录运行/失败/完成状态；`progress.json`记录设备进度；每个模型独立JSON；全部完成后生成`report.json`、`results.md`。
日志`Logs/industrial-v2/meta-test-seed42-context960-query256-official8.log`通过tee同步显示。
不覆盖旧结果，记录manifest/最佳模型/官方初始化/评估源码hash与同一任务序列hash。

三模型正式测试已正常完成，总耗时约778秒：CAQ=0.377522859，FT=0.422901573，官方zero-shot=9.438638782（参考尺度归一化MSE）。
CAQ相对FT低10.73%，并非验证集的48.7%。每模型52设备、431标量目标任务；设备、目标、单位、参考尺度一致性与文件hash已再次核验。
结果见上述目录的`results.md`/`report.json`，局限与产线维护诊断见`analysis.md`。仅单seed内部留出设备测试，不宣称真实工业迁移。
