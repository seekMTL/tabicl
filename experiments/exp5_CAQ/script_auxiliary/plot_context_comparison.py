"""
对比不同 context 长度下 Hopper 训练结果的 MSE 变化
"""
import matplotlib.pyplot as plt
import numpy as np

# 数据来源：四个 log 文件的最终对比部分
contexts = [1024, 2048, 4000, 8000]
labels = ['ctx=1024', 'ctx=2048', 'ctx=4000', 'ctx=8000']

# ===== 汇总指标 =====
reward_mse  = [0.001737, 0.001715, 0.001511, 0.001391]
delta_avg   = [0.001804, 0.001358, 0.001122, 0.000879]
overall     = [0.001799, 0.001388, 0.001155, 0.000922]

baseline_reward = [0.001498, 0.001498, 0.001498, 0.001501]
baseline_delta  = [0.001183, 0.001183, 0.001183, 0.001183]
baseline_overall = [0.001210, 0.001210, 0.001210, 0.001209]

# ===== 逐维 MSE（12维）=====
dim_names = ['reward', 'delta[0]', 'delta[1]', 'delta[2]', 'delta[3]',
             'delta[4]', 'delta[5]', 'delta[6]', 'delta[7]', 'delta[8]',
             'delta[9]', 'delta[10]']

# ctx=1024
caq_1024 = [0.001737, 1.071481e-08, 5.432773e-08, 4.672663e-08, 1.174065e-07,
            2.163206e-06, 0.000172, 0.000271, 0.001663, 0.001021, 0.002301, 0.014418]
# ctx=2048
caq_2048 = [0.001715, 8.626587e-09, 4.854458e-08, 5.271432e-08, 7.535143e-08,
            2.290743e-06, 0.000107, 0.000193, 0.001422, 0.001353, 0.001370, 0.010494]
# ctx=4000
caq_4000 = [0.001511, 8.368676e-09, 3.848670e-08, 3.461069e-08, 6.632447e-08,
            2.261999e-06, 0.000080, 0.000154, 0.001041, 0.000674, 0.001472, 0.008922]
# ctx=8000
caq_8000 = [0.001391, 7.028887e-09, 2.687806e-08, 2.280615e-08, 6.103087e-08,
            1.876705e-06, 0.000073, 0.000133, 0.000850, 0.000478, 0.000881, 0.007251]

# Baseline (以 ctx=1024 的 baseline 为参考，各实验 baseline 几乎一样)
baseline_dims = [0.001498, 4.618744e-09, 6.764704e-08, 3.517397e-08, 8.234905e-08,
                 1.758865e-06, 0.000176, 0.000252, 0.002055, 0.000850, 0.002393, 0.007289]

# ===== 画图 =====
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif',
                        'font.sans-serif': ['Noto Sans CJK SC', 'Noto Sans CJK', 'WenQuanYi Micro Hei']})
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ---- 子图1: 汇总指标随 context 变化 ----
ax1 = axes[0, 0]
x = np.arange(len(contexts))
w = 0.25
bars1 = ax1.bar(x - w, overall, w, label='Overall MSE', color='#e74c3c', alpha=0.85)
bars2 = ax1.bar(x, delta_avg, w, label='Delta(avg) MSE', color='#3498db', alpha=0.85)
bars3 = ax1.bar(x + w, reward_mse, w, label='Reward MSE', color='#2ecc71', alpha=0.85)

# 添加 baseline 参考线
ax1.axhline(y=np.mean(baseline_overall), color='#e74c3c', linestyle='--', alpha=0.5, label='Baseline Overall')
ax1.axhline(y=np.mean(baseline_delta), color='#3498db', linestyle='--', alpha=0.5, label='Baseline Delta(avg)')
ax1.axhline(y=np.mean(baseline_reward), color='#2ecc71', linestyle='--', alpha=0.5, label='Baseline Reward')

# 数值标注
for bar, val in zip(bars1, overall):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00002,
             f'{val:.5f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, delta_avg):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00002,
             f'{val:.5f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val in zip(bars3, reward_mse):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00002,
             f'{val:.5f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylabel('MSE')
ax1.set_title('汇总指标 vs Context 长度\n(数值越低越好)', fontweight='bold')
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(axis='y', alpha=0.3)

# ---- 子图2: CAQ/Baseline Ratio ----
ax2 = axes[0, 1]
overall_ratio = [o/b for o, b in zip(overall, baseline_overall)]
reward_ratio = [r/b for r, b in zip(reward_mse, baseline_reward)]
delta_ratio = [d/b for d, b in zip(delta_avg, baseline_delta)]

ax2.plot(contexts, overall_ratio, 'o-', color='#e74c3c', linewidth=2, markersize=10, label='Overall Ratio')
ax2.plot(contexts, reward_ratio, 's-', color='#2ecc71', linewidth=2, markersize=10, label='Reward Ratio')
ax2.plot(contexts, delta_ratio, 'D-', color='#3498db', linewidth=2, markersize=10, label='Delta(avg) Ratio')
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='CAQ=Baseline (1.0)')
# 标注数值
for i, (cx, r) in enumerate(zip(contexts, overall_ratio)):
    ax2.annotate(f'{r:.2f}x', (cx, r), textcoords="offset points", xytext=(0, 12),
                ha='center', fontsize=9, fontweight='bold', color='#e74c3c')
for i, (cx, r) in enumerate(zip(contexts, delta_ratio)):
    ax2.annotate(f'{r:.2f}x', (cx, r), textcoords="offset points", xytext=(0, -15),
                ha='center', fontsize=9, fontweight='bold', color='#3498db')

ax2.set_xlabel('Context 长度')
ax2.set_ylabel('CAQ / Baseline Ratio')
ax2.set_title('CAQ/Baseline 比值随 Context 变化\n(< 1.0 表示 CAQ 更优)', fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_xticks(contexts)

# ---- 子图3: 逐维 MSE 热力图 (log scale) ----
ax3 = axes[1, 0]
all_dims = np.array([caq_1024, caq_2048, caq_4000, caq_8000])
# 取 log10 便于可视化
all_dims_log = np.log10(all_dims)

im = ax3.imshow(all_dims_log.T, aspect='auto', cmap='RdYlGn_r')
ax3.set_xticks(range(len(labels)))
ax3.set_xticklabels(labels)
ax3.set_yticks(range(len(dim_names)))
ax3.set_yticklabels(dim_names, fontsize=9)
ax3.set_title('逐维 CAQ MSE (log10 色阶)\n绿色=低MSE(好)  红色=高MSE(差)', fontweight='bold')

# 在每个 cell 中标注实际 MSE 值
for i in range(len(labels)):
    for j in range(len(dim_names)):
        val = all_dims[i, j]
        if val < 1e-5:
            text = f'{val:.2e}'
        elif val < 0.001:
            text = f'{val:.1e}'
        else:
            text = f'{val:.4f}'
        color = 'white' if all_dims_log[i, j] < -2.5 else 'black'
        ax3.text(i, j, text, ha='center', va='center', fontsize=6, color=color)

cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
cbar.set_label('log10(MSE)')

# ---- 子图4: 选中维度对比 ----
ax4 = axes[1, 1]
# 选择变化最显著的几个维度
key_dims = [0, 6, 8, 10, 11]  # reward, delta[5], delta[7], delta[9], delta[10]
key_dim_names = [dim_names[i] for i in key_dims]

for i, di in enumerate(key_dims):
    vals = [caq_1024[di], caq_2048[di], caq_4000[di], caq_8000[di]]
    ax4.semilogy(contexts, vals, 'o-', linewidth=2, markersize=8, label=dim_names[di])

# baseline 参考线
for i, di in enumerate(key_dims):
    ax4.axhline(y=baseline_dims[di], linestyle=':', alpha=0.4,
                color=ax4.get_lines()[i].get_color())

ax4.set_xlabel('Context 长度')
ax4.set_ylabel('MSE (log scale)')
ax4.set_title('选中维度 MSE 随 Context 变化\n(虚线 = Baseline)', fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, which='both')
ax4.set_xticks(contexts)

plt.suptitle(f'Hopper-v5: CAQ 训练 Context 长度消融实验对比', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/home/lizitao/project/tabicl/experiments/exp5_CAQ/context_comparison_hopper.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("图表已保存到: experiments/exp5_CAQ/context_comparison_hopper.png")

# ===== 打印表格 =====
print()
print("=" * 90)
print("  📊 Context 长度消融实验 — Hopper-v5")
print("=" * 90)
print(f"{'Context':<10} {'Reward MSE':<15} {'Delta(avg)':<15} {'Overall MSE':<15} {'Overall Ratio':<15} {'训练耗时':<10}")
print("-" * 90)
times = ['1.0h', '2.3h', '7.1h', '22.6h+1.3h']
for i, ctx in enumerate(contexts):
    print(f"{ctx:<10} {reward_mse[i]:<15.6f} {delta_avg[i]:<15.6f} {overall[i]:<15.6f} "
          f"{overall_ratio[i]:<15.2f}x {'':<3}{times[i]}")
print("-" * 90)
print(f"{'Baseline':<10} {np.mean(baseline_reward):<15.6f} {np.mean(baseline_delta):<15.6f} {np.mean(baseline_overall):<15.6f}")
print()

# 逐维对比表
print("=" * 120)
print("  📊 逐维 CAQ MSE 对比")
print("=" * 120)
header = f"{'Dim':<14}" + ''.join(f"{l:<20}" for l in labels)
print(header)
print("-" * 120)
for j, name in enumerate(dim_names):
    vals = [caq_1024[j], caq_2048[j], caq_4000[j], caq_8000[j]]
    # 找最小值高亮
    min_idx = np.argmin(vals)
    row = f"{name:<14}"
    for i, v in enumerate(vals):
        marker = ' ←最优' if i == min_idx else ''
        if v < 1e-4:
            row += f"{v:<20.2e}"
        else:
            row += f"{v:<20.6f}"
    print(row)
print("-" * 120)
# Baseline
bs_row = f"{'Baseline':<14}"
for j, name in enumerate(dim_names):
    if baseline_dims[j] < 1e-4:
        bs_row += f"{baseline_dims[j]:<20.2e}"
    else:
        bs_row += f"{baseline_dims[j]:<20.6f}"
print(bs_row)
print()
print("✅ 结论：context 越大，Overall MSE 越低。ctx=8000 时 Overall=0.000922，ratio=0.76x（即 CAQ 比 Baseline 优 24%）")
