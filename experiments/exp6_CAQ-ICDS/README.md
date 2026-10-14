# exp6_CAQ-ICDS：CAQ-TabICL工业合成数据训练实验

> 2026-10-09新版训练：显式使用`data/industrial-v2/manifest.json`，8原型、512 world、1,834,294行；新增单位输入适配及GPU联调见[新版训练记录](INDUSTRIAL_V2_TRAINING.md)。启动脚本为`run_industrial_v2_seed42.sh`，旧`run_seed42.sh`仍是pilot对照。本文后续各节保留历史记录。

> 2026-10-08补充：官方原始输入核验与PC-Gym CSTR开发迁移评估已完成，详见[评估说明与结果](RAW_INPUT_CSTR_EVALUATION.md)。ICDS中CAQ/FT为0.176994/0.179654、官方raw为4.635190；CSTR context=960时CAQ/FT为0.001852/0.001270。后文保留早期实验记录，不能将其旧参数或结果误作当前最优协议。CSTR是独立方程来源的开发评估，不是真实工厂或未接触的最终测试。

## 1. 研究目标

论文主角是用户改进的 **CAQ-TabICL模型**。ICDS是论文中的数据方法贡献：在真实工业数据难以大规模采集时，用统一的受控动力学生成器产生训练数据，再检验CAQ-TabICL是否优于相同数据、相同预算下的原生TabICL。

本实验目录负责训练和评测；ICDS项目只负责生成、验证和保存数据。

## 2. 项目与环境

- TabICL项目：`/home/lizitao/project/tabicl`
- 本实验：`/home/lizitao/project/tabicl/experiments/exp6_CAQ-ICDS`
- ICDS项目：`/home/lizitao/project/Industrial_synthetic_data`
- 正式数据入口：`/home/lizitao/project/Industrial_synthetic_data/data/pilot-m3/manifest.json`
- TabICL conda环境：`/home/lizitao/software/miniconda3/envs/tabicl`
- ICDS conda环境：`/home/lizitao/software/miniconda3/envs/icds`
- 原始TabICL回归checkpoint：`/home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt`

不要重新创建Python环境，不要重新生成正式ICDS数据。

## 3. ICDS数据是什么

ICDS将不同工业场景统一表示成：

```text
(当前状态 state, 控制动作 action, 可见外部条件) -> 下一状态 next_state
```

正式pilot包含256个world、2,048条episode、524,288个事实状态、130,713个动作分叉点、916,427行转移，以及113个实际采样到的结构program。

三类机制：

1. `discrete`：稳定稀疏线性/非线性离散动力系统；
2. `physical_ode`：热网络与带阀门执行器的加热液罐；
3. `hybrid_line`：含队列、加工、温度、磨损、故障和维护的双工位产线。

部分source从同一状态替换不同动作，形成共享随机创新下的单步动作分支。数据是受物理方程、稳定性条件和事件规则约束的简化模拟数据，不是一比一校准的真实工厂数据。

训练不得手工读取单个`action.npy`或`state.npy`；应通过manifest和`IndustrialTaskSampler`按相同行号组装任务，并保持world/episode/source隔离。

## 4. 代码职责

- `train_industrial.py`：正式训练入口（v2：state-only CAQ + cosine 调度 + 病态保护）；
- `industrial_adapter.py`：将ICDS任务转换为CAQ/TabICL输入；
- `industrial_task_sampler.py`：抽取训练、验证context/query；
- `industrial_common.py`：共同评测、SCM replay辅助逻辑；
- `eval_industrial.py`：重新评测已保存模型；
- `_archive/`：外部工业基准评测脚本（eval_external*.py，当前论文主线非必须，已归档）；
- `run_seed42.sh`：现有seed 42等预算对照命令；
- `Logs/`：新日志；
- `checkpoints/`：新模型。

`train_industrial.py`复用`../exp5_CAQ/train_large6_counterfactual.py`中的回归SCM replay采样器。不要修改或删除exp5中用户原来的MuJoCo训练实验。

## 5. 当前模型与训练方式

- `caq`（v2 默认，state-only）：action 只进入CAQ专用causal通道，不拼进特征表（用户原设计意图）；
- `tabicl`：状态和动作作为普通特征拼接（原生TabICL信息流，对照组）；
- `caq-concat`（旧变体，仅消融）：action 同时拼接特征 + 走causal通道。

两模型都拿到全部state+action输入信息，只是处理方式不同；均从同一原始TabICL回归checkpoint初始化，冻结`icl_predictor`，训练可训练的列/行模块；CAQ额外训练CAQ模块。训练按80%工业任务、20%原始SCM replay混合，以减轻对预训练能力的遗忘。

v2 训练改进（对齐`train_large6_counterfactual.py`，2026-09-21）：
- `--lr-schedule cosine`（默认）：CosineAnnealingLR从lr退火到lr×0.05，针对seed42固定1e-4后期验证指标大幅波动（CAQ 23k best 0.2169 → 30k 0.5199）；
- 病态批次保护：非有限loss/梯度跳过该步、NaN参数快照回滚，连续10步才停；
- 调度器状态随checkpoint保存，支持断点恢复。

现有seed 42参数：

```text
steps=30000
context_size=960
query_size=256
targets_per_task=1
replay_ratio=0.2
lr=1e-4 + cosine退火至5e-6
eval_interval=1000
prior=A4（三族混合）
history_ratio=0.5，仅hybrid_line使用8步带噪历史
sensor_noise_multiplier=10
```

完整命令见`run_seed42.sh`。

## 6. 已完成的seed 42结果

旧结果暂存在ICDS项目：

- CAQ：`/home/lizitao/project/Industrial_synthetic_data/checkpoints/paper-main-v2/caq-A4-seed42`
- TabICL：`/home/lizitao/project/Industrial_synthetic_data/checkpoints/paper-main-v2/tabicl-A4-seed42`

| 模型 | 最佳固定尺度验证MSE | 最佳步数 | 最后一步MSE |
|---|---:|---:|---:|
| CAQ-Concat | 0.216859 | 23,000 | 0.519857 |
| TabICL-FT | 0.218316 | 30,000 | 0.218316 |

CAQ最佳值比TabICL低约0.67%，只是单seed的小幅正向结果，不能作为论文最终结论。CAQ在23,000步后明显波动，说明当前学习率/训练后期策略可能需要改进。使用CAQ旧结果时应读取`best.pt`，不要把`final.pt`当最佳模型。

主指标`macro_fixed_scale_mse`是各目标先用预先规定、与query标签无关的物理参考尺度归一化，再按world/target宏平均；它不是原始物理单位MSE。

## 7. 新AI接手时应先做什么

1. 阅读本README、`train_industrial.py`和`run_seed42.sh`；
2. 对照`exp5_CAQ/train_large6_counterfactual.py`，检查工业训练是否保留用户原CAQ设计意图；
3. 读取旧seed 42的`args.json`、`summary.json`和`evaluations.json`，分析CAQ在23k后退化的原因；
4. 检查学习率衰减、早停、最佳checkpoint选择、验证方差，以及CAQ/TabICL是否真正保持等预算；
5. 先做短小且可判定的消融，不要直接启动多seed长训练；
6. 修改训练方案后，明确解释改了什么、为什么、预期解决什么问题；
7. 新日志和checkpoint只写入本exp6目录。

不要将“混合三类工业数据不如单类数据”误判为“CAQ模型失败”。论文主问题是：在完全相同的工业数据、输入信息和训练预算下，CAQ-TabICL能否优于原生TabICL。

## 8. 文件保存规则

训练默认保存`best.pt`、`final.pt`、参数、摘要、验证历史、依赖版本和源码哈希，不复制整套源码。默认不保存第三份`latest.pt`；只有需要断点恢复时显式添加`--save-latest`。
