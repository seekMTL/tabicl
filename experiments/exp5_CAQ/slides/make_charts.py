"""组会 PPT 图表：分布内 19 环境比值图 + hold-out few-shot 外推对比图。

配色按 dataviz 规范（已验证）：diverging 蓝#2a78d6（赢）/红#e34948（输）；
表面#fcfcfb、主文字#0b0b0b、次级#52514e、网格#e1e0d9、轴线#c3c2b7。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"   # 赢 baseline（比值 < 1）
RED = "#e34948"    # 输 baseline（比值 > 1）
OUT = "/home/lizitao/project/tabicl/experiments/exp5_CAQ/slides/charts"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


# ═══════════════════════════════════════════════════════════════════
# 图 1：19 环境分布内 best/Base（lr=1e-4 版，横向 bar，diverging）
# ═══════════════════════════════════════════════════════════════════
envs = ["HopperGravity-g-1", "HopperGravity-g-0.5", "HopperGravity-g0",
        "HopperGravity-g0.5", "HopperGravity-g1",
        "Walker2dFriction-f0.3", "Walker2dFriction-f1", "Walker2dFriction-f3",
        "HalfCheetahSlope-s-10", "HalfCheetahSlope-s0", "HalfCheetahSlope-s10",
        "AntJoints-af0", "AntJoints-af1", "AntJoints-af2", "AntJoints-af3",
        "AntJoints-af4", "AntJoints-af5", "AntJoints-af6", "AntJoints-af7"]
ratio = [0.45, 0.49, 0.55, 0.70, 0.51,
         0.99, 0.96, 0.96,
         1.04, 0.92, 1.01,
         0.81, 0.81, 0.79, 0.84, 0.85, 0.82, 0.81, 0.77]
groups = [("HopperGravity", 0, 5), ("Walker2dFriction", 5, 8),
          ("HalfCheetahSlope", 8, 11), ("AntJoints", 11, 19)]

fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=200)
fig.patch.set_facecolor(SURFACE)
y = range(len(envs))[::-1]  # 顶部是第一个环境
colors = [BLUE if r < 1.0 else RED for r in ratio]
ax.barh(y, ratio, height=0.62, color=colors, zorder=3)
ax.axvline(1.0, color=INK2, linewidth=1.6, linestyle=(0, (5, 3)), zorder=4)
for yi, r in zip(y, ratio):
    ax.text(r + (0.012 if r < 1 else -0.012), yi, f"{r:.2f}x",
            va="center", ha="left" if r < 1 else "right",
            fontsize=9.5, color=INK2)
ax.set_yticks(list(y))
ax.set_yticklabels(envs, fontsize=10)
ax.set_xlim(0.35, 1.22)
ax.set_xlabel("CAQ best MSE / TabICL baseline MSE（< 1 表示 CAQ 更优）",
              fontsize=10.5, color=INK2)
style_axes(ax)
ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
# 组分隔线（组间留空隙的视觉替代：细灰线）
for _, s, e in groups:
    if s > 0:
        ax.axhline(s - 0.5, color=GRID, linewidth=1.2)
ax.text(0.99, len(envs) - 0.28, "baseline = 1.00", fontsize=9, color=INK2, ha="right")
fig.savefig(f"{OUT}/fig1_in_distribution.png", bbox_inches="tight",
            facecolor=SURFACE)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════
# 图 2：hold-out few-shot 外推（4 留出环境，横向 bar，diverging）
# ═══════════════════════════════════════════════════════════════════
h_envs = ["HopperGravity-g1", "Walker2dFriction-f3",
          "HalfCheetahSlope-s10", "AntJoints-af4"]
few = [0.62, 1.02, 1.20, 0.91]

fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=200)
fig.patch.set_facecolor(SURFACE)
yy = range(len(h_envs))[::-1]
colors = [BLUE if r < 1.0 else RED for r in few]
ax.barh(list(yy), few, height=0.55, color=colors, zorder=3)
ax.axvline(1.0, color=INK2, linewidth=1.6, linestyle=(0, (5, 3)), zorder=4)
for yi, f in zip(yy, few):
    ax.text(f + (0.02 if f < 1 else -0.02), yi, f"{f:.2f}x",
            va="center", ha="left" if f < 1 else "right",
            fontsize=10, color=INK2)
ax.set_yticks(list(yy))
ax.set_yticklabels(h_envs, fontsize=11)
ax.set_xlim(0, 1.55)
ax.set_xlabel("few-shot：CAQ / Baseline 比值（context = 留出环境自身 2 万行；< 1 表示 CAQ 更优）",
              fontsize=10.5, color=INK2)
style_axes(ax)
fig.savefig(f"{OUT}/fig2_holdout.png", bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)

print("图表生成完成")
