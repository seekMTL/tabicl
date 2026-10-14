# CAQ-TabICL 工业受控动力学 benchmark 方案

调研日期：2026-10-14。本文是可执行的研究方案，不是已经构建完成的数据集或已经运行的模型评测。详细来源与核验状态见 `caq_dynamics_benchmark_sources_2026-10-14.json`。本轮实际下载并解析了 DaISy 的干燥器和玻璃炉数据；读了 KUKA 技术报告与官方加载器；没有生成新评测数据、训练模型或运行 benchmark。

## 研究问题与定位

核心问题：在相同可见状态/观测、控制输入及带标签上下文预算下，模型能否适应工业动力学系统，并准确预测动作之后的响应、递归轨迹和可验证的模拟动作效应？

推荐第一版包含 8–10 个系统：4 个外部过程方程、4–6 个实测输入输出任务。在其中的 4 个过程方程上另外生成成对动作探针。系统方程数、参数实例数、独立记录数、轨迹数与转移数分别报告，参数实例不计作独立机制。

研究贡献应落在：工业任务覆盖、上下文适应、动作输入与扰动区分、信息条件、行为策略偏移、多步误差、配对动作真值及可复現协议。工业动力学与 ICL 系统辨识已有研究，不能仅因 CAQ 类模型少而声称这是第一个工业动力学 benchmark。

已有代表性工作：

- [DynaDojo，NeurIPS 2023 Datasets and Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2023/file/32093649cbbcff773d9a991d8c30a7fe-Paper-Datasets_and_Benchmarks.pdf)：样本量、系统复杂度、OOD 与可扩展生成器/评测接口。
- [PDEBench，NeurIPS 2022 Datasets and Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0a9747136d411fb83f0cf81820d44afb-Abstract-Datasets_and_Benchmarks.html)：多种方程、生成代码和物理/多步指标。空间场接口与当前 CAQ 不匹配，作为方法参考。
- [Deep subspace encoders / SUBNET，Automatica 156 (2023), 111210](https://www.sciencedirect.com/science/article/pii/S0005109823003710)：用历史估计隐状态，作为实测输入输出轨道的重要动力学基线。[代码](https://github.com/GerbenBeintema/deepSI)。
- [From system models to class models，IEEE Control Systems Letters 7 (2023), 3513–3518](https://doi.org/10.1109/LCSYS.2023.3335036)：使用上下文的动力学元模型。[代码](https://github.com/forgi86/sysid-transformers)。已发布模型以特定系统类/接口为中心，不能假设任意 MIMO 数据可直接输入。
- [Dynamics as Prompts / CAPTURE，RA-L 官方代码](https://github.com/XilunZhangRobo/CAPTURE-Sim2Real)：机器人 sim-to-real ICL 相关工作，作为第二阶段候选对照。
- [PC-Gym，Computers & Chemical Engineering 204 (2026), 109363](https://www.sciencedirect.com/science/article/pii/S0098135425003667)：可公开复用的过程方程及扰动设置；本文重新定义动力学评测，不把其 RL reward 当预测指标。

## 与既有训练和开发来源的关系

当前 ICDS industrial-v2 manifest 明确包含 thermal、heated_tank、mass_spring、dc_motor、coupled_tanks、heat_exchanger，另有离散和混合产线任务。必须记录每个候选系统与这些机制的对应关系。

PC-Gym 单 CSTR 已用于本地开发评估，见 `experiments/exp6_CAQ-ICDS/RAW_INPUT_CSTR_EVALUATION.md`。已有开发 world 和记录不能转称全新最终外部测试。可继续用于接口联调、超参数选择和历史对照。新 CSTR 串联回流方程虽不同，仍标记为“反应器来源家族已有开发接触”。

ICDS 原有留出集可以作为独立的训练分布诊断表。它不能替代外部来源/新方程评测。电机、换热器与储罐的外部实测来源可以评估测量与来源迁移，但不一概标为训练中完全未见的物理机制。

## 三种评测任务

### S：完整状态的外部过程动力学

输入：当前物理状态 x_t、实际控制动作 u_t、明确声明可见的外部扰动 e_t、采样间隔 dt。目标为 x_(t+1) 或连续状态增量。状态须足以描述该模拟器的 Markov 转移；若增加执行器延迟、滞后或滤波，应把对应内部状态纳入 S 轨道。

CAQ 与工业预训练 TabICL-FT 固定 checkpoint，在目标系统上不更新权重。测试参数实例可提供带标签 support，允许模型在上下文中适应。所有模型获得相同物理信息，CAQ 通过专用动作通道处理，普通模型可以拼接动作列。

### H：实测/部分可观测输入输出

输入：截至 t 的输出历史和过去输入、当前动作及可见扰动。默认观察窗口 L=16；L=4/32 仅在开发数据上用于诊断或选择，最终值全局冻结。目标为 y_(t+1)。有限历史不保证恢复所有隐状态，这一点作为轨道限制记录。

CAQ 输入适配方案：把输出历史与过去动作放入普通特征，把当前动作送入 CAQ 动作通道。该方案需要新增并验证适配器，当前并未实现；名称与记录注明“CAQ + 固定历史适配”，不能称为原 hr0 模型原生支持部分可观测系统。

单步预测可以读取实际过去观测。递归 rollout 仅在初始化窗口读取真实输出，之后必须递归使用预测，不能在每一步继续注入真实未来输出。未来动作按已知输入序列提供；未来不可观测扰动不得作为额外模型输入。

### A：模拟器上的配对动作效应

从相同状态、相同参数和相同可见扰动出发，施加不同动作，获得两种后继状态。确定性阶段直接复用相同起点；随机阶段使用可恢复 RNG/相同外部噪声轨迹，并说明其定义是该模拟器的共享噪声动作差异。

对比模型预测的响应差与模拟器响应差。真实闭环日志通常没有这种成对真值，不能用动作列打乱的结果替代真实反事实效应评估。S、H、A 各有主结果表，不合并为单个排名。

## 第一版来源清单

以下状态/输入数量来自已阅读的官方文档或源码。它们尚需数值积分、单位和有效范围联调；采用预先定义的数据质量门槛，不按 CAQ 的测试分数筛选。

| 系统 | 类型 | 信息/动作 | 首版角色 | 来源 |
|---|---|---|---|---|
| 多级萃取 | 外部方程仿真 | 10 状态，2 输入 L/G | S 主任务 + A 配对 | [官方模型](https://maximilianb2.github.io/pc-gym/API/model_classes/)、[过程说明](https://maximilianb2.github.io/pc-gym/env/extraction-column/) |
| 蒸馏塔 | 外部方程仿真 | 9 状态，2 输入 R/F | S 主任务 + A 配对 | [官方模型](https://maximilianb2.github.io/pc-gym/API/model_classes/) |
| 串联回流 CSTR | 外部方程仿真 | 4 状态，4 输入 F/L/Tc1/Tc2 | S 主任务 + A 配对；标明来源接触 | [官方模型](https://maximilianb2.github.io/pc-gym/API/model_classes/) |
| 光生物生产 | 外部方程仿真 | 3 状态，光照与营养进料 2 输入 | S 主任务 + A 配对 | [官方过程说明](https://maximilianb2.github.io/pc-gym/env/photoproduction/) |
| 工业干燥器 | 工业测量 | 3 输入/3 输出，867 行，10 s | H；少样本 MIMO | [原始说明](https://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/dryer2.txt) |
| Philips 玻璃炉 | 工业测量 | 3 输入/6 温度输出，1247 行 | H；制造过程 MIMO | [原始说明](https://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/glassfurnace.txt) |
| EMPS | 实验设备测量 | 实际驱动力到位置；闭环数据 | H；摩擦与机械响应 | [官方说明](https://sites.google.com/view/nonlinear-benchmark/benchmarks/emps) |
| Cascaded Tanks | 实验设备测量 | 水泵输入到液位；内部状态未全部观测 | H；溢流与历史依赖 | [官方说明](https://www.nonlinearbenchmark.org/benchmarks/cascaded-tanks) |
| Silverbox | 实验设备测量 | 电子实现的 Duffing 输入输出 | H 扩展；非线性幅值变化 | [官方说明](https://www.nonlinearbenchmark.org/benchmarks/silverbox) |
| Wiener-Hammerstein | 实验设备测量 | 动态级联的输入输出 | H 扩展；动态非线性 | [官方说明](https://www.nonlinearbenchmark.org/benchmarks/wiener-hammerstein) |

DaISy 下载核验：dryer's gzip SHA256 为 `0000546d78f737a8a33bf91ae62235fa8848a730b660ef1a25a4fda12d3b064f`，解析为 867×7（时间+3输入+3输出）；glass-furnace gzip SHA256 为 `61d1693a2d379a9fcf6edf2f474496d5a92eef4f193119265b477f6e58f6e657`，解析为 1247×10（时间+3输入+6输出）。两份文件值均有限。干燥器原料流量是否作为可控动作或可见工况，需要依原描述确认。玻璃炉说明未填写物理采样间隔，先使用记录步长并在数据卡中标注。

EMPS 要采用实际施加的力 `vir`，不要把目标位置替代为同一种动作。官方加载器 commit 已核对为 `f9fb3883086870a27b31917ccde1f78c95d53cb2`；不同数据集有各自的 train/test 和初始化窗口，例如 EMPS=20、Cascaded Tanks=50。benchmark 首版可以保留官方子任务单独重现，也可构造统一短窗口任务，但修改后的成绩不能直接当官方 leaderboard 成绩。

## 第二阶段扩展

| 来源 | 增加的能力 | 为什么延后 |
|---|---|---|
| [KUKA 工业机器人与 DFKI 报告](https://www.dfki.de/en/web/research/projects-and-publications/publication/15661) | 6 输入/6 输出、摩擦、耦合、真实工业机器人 | 准备好的正向数据是力矩与位置；原始速度额外获取会改变信息条件。不是完整 12 状态默认可见。官方 39988/3636 train/test，100 ms；压缩格式/轨迹边界需适配 |
| [CubeSpec Fine Steering Mirror](https://www.nonlinearbenchmark.org/benchmarks/fine-steering-mirror) | 3 输入/3 输出，压电迟滞，多激励幅值 | 周期稳态多正弦且无瞬态；不应当与普通瞬态任务合并混称 |
| [Coupled Electric Drives](https://www.nonlinearbenchmark.org/benchmarks/coupled-electric-drives) | 电机与柔性传动、观测丢失速度符号 | 记录短且输出为绝对速度，历史适配与 context 预算需另定 |
| [DaISy 蒸汽换热器](https://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/exchanger.txt) | 实测非最小相位过程，4000 行，1 s | ICDS 已有换热机制，作为外部测量迁移而非全新机制 |
| [Parallel WH，Automatica 2015](https://www.nonlinearbenchmark.org/benchmarks/parallel-wiener-hammerstein) | 多分支、幅值外推 | 与 WH 相关，汇总时按家族记录 |
| PC-Gym 聚合反应器、反应萃取、结晶 | 新方程、较多动作/状态、刚性动力学 | 原始参考、单位、可行区间、求解一致性先核验；结晶返回 7 分量但 CV/Ln 是派生量，避免把它们当独立状态或独立任务 |
| [Siemens Industrial Benchmark](https://github.com/siemens/industrialbenchmark) | 延迟、异方差、部分可观测工业复杂性 | 工业启发的仿真；3 动作，需要历史，不能称为实测工厂日志 |
| [NeoRL-2](https://github.com/polixir/NeoRL2) 的 Pipeline/DMSD/Fusion | 现成 obs/action/next_obs、保守策略与延迟 | 已确认来源是 2025 预印本和 7 个模拟任务；核验 observation 是否足够、内置策略/执行器状态、模型来源与拆分规则 |
| [DOE/LBNL BOPTEST](https://www.energy.gov/cmei/buildings/boptest-building-operations-testing-framework) | HVAC 动作、天气扰动与长时常 | Modelica/FMU/容器适配与隐热状态增加工作量；来源是 JBPS2021 和国家实验室项目 |
| [Tennessee Eastman](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1) | 大规模化工、多变量与模式变化 | 公共故障数据是否包含有效变化的操纵变量、具体输入输出时序对齐待核验；属于过程仿真 |

PDEBench/FNO 型空间场任务作为远期扩展。普通静态制造质量、纯 RUL 或只含自主演化轨迹的数据，先核对控制输入与目标响应；无有效动作和时序响应者不进入本次动作条件主轨道。

## 新建仿真数据的具体规模与方法

首版 4 个方程，每方程 12 个物理参数实例：4 个开发实例、8 个最终实例；每实例 6 条 512 步轨迹。其中 3 条为 support 池、3 条为 query 池，不共享 episode 或转移源。这是 `4×12×6×512=147456` 条事实转移的规划规模，实际数量以满足预先固定的有效性规则后为准。

参数围绕文献标称值在经过验证的物理范围内采样，优先用约 ±10%–20% 的初始范围进行数值联调，再在开发阶段冻结各参数区间。不能统一对所有参数任意扰动：正参数、质量守恒、组分和温度/浓度边界、稳定/可运行区间均应保持。新参数 seed 不能独自证明新机制。

动作激励按系统时常设计，至少有随机阶跃保持、多正弦、受约束 PRBS/独立通道扰动、保守反馈工况。PRBS 各通道要独立/正交，避免只有相关动作使 MIMO 响应不可区分。随机每步剧烈变化不一定符合慢化工过程。同时记录指令动作与执行后的实际输入；主任务明确采用哪一种。

从每实例选 128 个真实生成轨迹中的锚点，为每个锚点生成 3 组有边界约束的动作比较。干预仅改变可控输入，其他状态、参数和扰动保持一致。主探针使用无随机过程噪声的设置；再扩展共享噪声设置。所有真值与 RNG/内部参数记录放入 evaluator-only 区域。

新构建的挑战优先是以下测试切片，而不是简单给旧 ICDS 换随机种子：

1. 同一方程新参数实例：上下文适应。
2. 相同实例的新动作频谱/幅值：激励与行为策略变化。
3. 单通道、独立多通道、相关多通道的动作覆盖：动作耦合可辨识性。
4. 相同起点不同动作的探针：响应差真值。
5. 传感噪声、执行器饱和、迟滞/延迟：独立标注信息条件。延迟内部状态公开时属于 S，隐藏时属于 H。

第一版只落实 1、2、4 的基本形式；后续扩展 3、5，避免首版配置组合爆炸。

## 固定评测协议

- 主 context 预算 K=64、256；K=960 仅对记录足够的任务作为扩展。短实测记录不补重复行来制造 960 个例子，也不因长度不足从题库中删除。
- 3 个固定 context 抽样 seed，前缀嵌套，共享同一 query 集。它们度量抽样变化，不是 3 个独立训练 seed 或 3 个新系统。
- S 轨道支持按转移抽样；H 轨道提供可重建的连续块和窗口，所有模型拥有相同原始观测历史与相同数量监督转移。不能把散乱行伪装成连续序列供 RNN 使用。
- 官方 train/test 优先；无划分的 DaISy 使用开发阶段确定的连续时间块，例如前 60% support/训练、随后 20% 开发、最后 20% 测试。边界设置至少 L+H 的窗口隔离。具体方案写入数据卡后冻结。
- 对 S 的开发/最终参数实例分离；对只有一个真实设备的记录，声明评估的是新时间段/新轨迹预测，不宣称新设备或新机制泛化。
- 单步、10 步、50 步递归轨迹分别报告；每项附采样间隔和实际预测时长。不能把每步读取真值的多次单步误差称为自由递归多步误差。
- 输入预处理及调参仅依开发/support 部分。主表使用固定 CAQ/FT 权重。若增加测试域梯度更新，建立独立适配表，给出数据与计算预算。
- 主神经比较同时保留一致 typed 适配与官方自然原始接口的透明记录，防止输入包装优势被当作模型能力。模型信息量相同不要求数值编码相同。
- 数据入选/剔除依据可读性、动作语义、覆盖、数值与物理有效性；记录失败。模型误差差不会成为剔除条件。

## 指标与统计

每个目标先定义冻结参考尺度 c_j：有公开工程范围则用该范围/明确物理参考；没有时使用 support/校准池中的输出标准差，并有测量分辨率或开发阶段确定的下限。同一系统各 K 使用相同评分尺度。模型输入标准化仍只拟合当次允许读取的 support。

单步主误差为 `mean(((prediction_j-target_j)/c_j)^2)`。连续目标可以在增量空间训练，误差回到后继状态/输出对应尺度评估。不要因高采样率下增量方差很小，临时采用会放大指标的分母。每个任务额外报告原单位 RMSE/MAE。

先在每个实例内对目标平均，再对实例平均，最后对系统/方程等权平均。同源参数实例、多个幅值测试和多个输出列不会计作多个独立机制。分别汇报 S、H、A；化工、机械、制造等子组作诊断，不在首版强行承诺严格均衡的跨产业覆盖。

主结果同时给出 persistence 基线及相对误差/预测 skill（分母接近零时使用预先规定规则），防止“总误差低”只是相邻状态几乎不变。多步报告轨迹 NMSE、末端误差、无效数值/发散比例和物理越界率；不在评分前隐式裁剪预测来掩盖发散。

动作效应指标是预测响应差与真响应差之差的尺度归一化 MSE。真实日志没有可靠的配对真值时不报告 A 轨道成绩。

每个方法在相同任务上配对比较。置信区间按系统/实例层级或独立记录块抽样，不把相邻转移视为独立样本。小规模首版主要报告逐系统误差与配对区间，避免在少量独立系统上作过强总体显著性结论。训练 seed 的不确定性需通过另训模型评估，不能用 context seed 替代。

## 对照模型的分阶段清单

首版运行 persistence、Ridge/ARX、调参预算明确的树模型、直接状态动作 MLP、CAQ、同训练数据 TabICL-FT、官方 TabICLv2、LimiX-16M。S 轨道尽快补 SINDy with control；H 轨道补现代 neural state-space/SUBNET。树/MLP/SINDy 可以在提供的 support 中拟合参数，CAQ/FT 主表保持冻结，分别标明学习模式。它们比较的是同样下游数据预算下的实用预测能力。

CAQ vs 同源 TabICL-FT 是隔离架构收益的关键比较。CAQ vs 未工业预训练的官方模型包含预训练收益，不能把全部差距归因于动作 query。为强机制比较，后续用同样 ICDS 数据和计算预算预训练普通状态动作 MLP/动力学元模型。

第二阶段核对 sysid-transformers 与 CAPTURE 的数据接口/预训练覆盖/公开 checkpoint；如果只支持 SISO 或机器人专用 schema，重训适配版本并明确命名，不假装把原模型直接当通用过程模型。评测预测模型，控制 policy 的 reward 作为未来闭环扩展。

动作消融可以加动作拼接、删除当前动作、打乱动作的诊断；打乱动作制造的分布变化不能单独证明因果识别。主 A 真值来自已知模拟器。

## 数据与发布目录

```text
benchmark/
  manifest.json
  provenance/             # DOI、URL、版本、许可、hash、ICDS/开发接触关系
  raw_downloads/          # 有明确再发布许可时才打包；否则提供下载脚本
  trajectories/          # 按 source/system/world/episode 存储
  splits/                # 开发/最终实例，support/query，时间块边界
  probes/                # 可控模拟器的配对动作，truth 分离
  tasks/                 # 各 K/L/H 对应同一任务清单和输入 allowlist
  adapters/              # 自然原始、typed、历史适配；动作/扰动区分
  evaluators/            # 固定尺度、rollout、动作效应、层级汇总
  results/               # 逐任务预测、失败原因、运行时间、显存与版本
  data_cards/            # 每个来源的范围、单位、动作语义与信息条件
```

核心 schema 包含 source_id、system_id、world_id、episode_id、t、dt、observation 或完整 state、action_command、action_applied、observed_exogenous、next_observation/next_state、终止标记、单位和缺失掩码。隐藏参数、latent_state、RNG 和 probe 真值仅供 evaluator。时间对齐、执行器有效延迟、终止和边界跳变有显式说明。

PC-Gym 仓库的代码为 MIT；实测数据的再发布许可逐源核验，不能把加载器的开源许可推广到数据。DaISy 本轮仅确认公开下载及引用要求，未确认原始数据再发布许可；首版提供下载和转换脚本，不未经确认打包这些原始记录。

## 快速推进顺序与验收

以下是单人推进的工作分解目标，不是已经测得的运行时间，也不是确定交付日期。实际计算耗时依 GPU、模型和目标数确定。

| 阶段 | 工作目标 | 可检查产物 |
|---|---|---|
| 第 1 天 | 冻结研究问题、轨道和首批来源；确定可见信息/尺度/划分 | 注册表、数据卡、protocol 草案、旧训练/开发来源映射 |
| 第 2–3 天 | 4 个方程的 CPU 数值核验；4 份实测数据转换和窗口隔离；生成小样例 | 固定 manifest、数组、hash、样本图、转移/对齐核验 |
| 第 4 天 | Persistence/Ridge/MLP 和 CAQ/FT/官方小规模接入 | 同一任务的形状、信息、目标、评分一致性记录 |
| 第 5–7 天 | 跑完整首版、上下文曲线、多步与配对探针；补树/LimiX | S/H/A 独立结果、逐系统表、失败表、复现入口 |
| 随后 1–2 周 | 补 KUKA/FSM/IB、DeepSI/ICL 动力学基线、训练 seed | 更强外部证据与现代动力学对照 |

若硬件/时间紧张，先交付 S 轨道的 4 方程 + A 配对主实验，H 轨道的 4 实测数据卡/预处理与先导结果明确标记进度。不能把尚未接通 H 适配或尚未跑完的候选源写成完成的 benchmark。

首版验收：每个任务有来源与信息条件；划分无窗口/episode 泄漏；K 不通过重复行补齐；所有方法读取相同信息和标签预算；评分尺度固定；rollout 真递归；模拟配对恢复同状态/同噪声；官方原始接口可核对；数值失败和物理越界可见；测试集未参与选择；下载/生成/转换/评测入口可复现。
