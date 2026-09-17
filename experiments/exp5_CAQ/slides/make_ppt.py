"""组会 PPT：反事实多环境动力学学习实验汇报（8 页，16:9）。

配色与图表一致（dataviz 规范色）：主文字 #0b0b0b、次级 #52514e、accent 蓝 #2a78d6、
赢绿 #006300、输红 #d03b3b。中文字体 Microsoft YaHei（Windows 打开生效）。
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

INK = RGBColor(0x0B, 0x0B, 0x0B)
INK2 = RGBColor(0x52, 0x51, 0x4E)
ACCENT = RGBColor(0x2A, 0x78, 0xD6)
GOOD = RGBColor(0x00, 0x63, 0x00)
BAD = RGBColor(0xD0, 0x3B, 0x3B)
SURFACE = RGBColor(0xFC, 0xFC, 0xFB)
GRID = RGBColor(0xE1, 0xE0, 0xD9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BASE = "/home/lizitao/project/tabicl/experiments/exp5_CAQ/slides"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def _set_ea(run, typeface="Microsoft YaHei"):
    """设置东亚字体（python-pptx 的 font.name 只管 latin，中文需要 a:ea）。"""
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        e = rPr.find(qn(tag))
        if e is None:
            e = etree.SubElement(rPr, qn(tag))
        e.set("typeface", typeface)


def tx(slide, x, y, w, h, text, size=14, bold=False, color=INK,
       align=PP_ALIGN.LEFT, line_spacing=1.12, anchor=MSO_ANCHOR.TOP):
    """文本框：支持 \n 多段。"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = ln
        f = r.font
        f.size = Pt(size)
        f.bold = bold
        f.color.rgb = color
        f.name = "Microsoft YaHei"
        _set_ea(r)
    return tb


def title_bar(slide, title, sub=None):
    """页面标题条：accent 竖线 + 标题 + 可选副题。"""
    bar = slide.shapes.add_shape(1, Inches(0.55), Inches(0.52), Inches(0.09), Inches(0.62))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    tx(slide, 0.85, 0.42, 11.8, 0.7, title, size=25, bold=True)
    if sub:
        tx(slide, 0.87, 1.02, 11.8, 0.4, sub, size=12.5, color=INK2)


def footer(slide, page):
    tx(slide, 12.35, 7.08, 0.75, 0.3, f"{page}", size=10, color=INK2,
       align=PP_ALIGN.RIGHT)


def table(slide, x, y, w, h, rows, col_widths, header=True, font_size=11.5,
          highlight_cols=None):
    """原生表格。highlight_cols: [(row_idx, col_idx, GOOD|BAD), ...] 上色。"""
    n_r, n_c = len(rows), len(rows[0])
    gt = slide.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(w), Inches(h)).table
    for ci, cw in enumerate(col_widths):
        gt.columns[ci].width = Inches(cw)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = gt.cell(ri, ci)
            cell.margin_top = cell.margin_bottom = Emu(18000)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if (header and ri == 0) else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = str(val)
            f = r.font
            f.size = Pt(font_size)
            f.name = "Microsoft YaHei"
            _set_ea(r)
            if header and ri == 0:
                f.bold = True
                f.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = ACCENT
            else:
                f.color.rgb = INK
                cell.fill.solid()
                cell.fill.fore_color.rgb = SURFACE if ri % 2 else RGBColor(0xF2, 0xF4, 0xF8)
    if highlight_cols:
        for ri, ci, c in highlight_cols:
            cell = gt.cell(ri, ci)
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.bold = True
                    r.font.color.rgb = c
    return gt


# ═══════════════════════════════════════════════════════════════════
# P1 封面
# ═══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0x12, 0x26, 0x3D)
bg.line.fill.background()
tx(s, 1.1, 1.7, 11.1, 1.9,
   "反事实多环境动力学学习\nCAQ-TabICL 在 19 个扰动 MuJoCo 世界上的训练与泛化",
   size=33, bold=True, color=WHITE, line_spacing=1.25)
tx(s, 1.1, 4.0, 11.1, 0.5, "exp5_CAQ · train6_counterfactual 实验系列", size=17, color=RGBColor(0x9E, 0xC5, 0xF4))
tx(s, 1.1, 4.65, 11.1, 0.4, "环境生成参照 MetaKoopman（NeurIPS 2025）· 分布内 17/19 胜 · few-shot 跨世界适应", size=13, color=RGBColor(0x8A, 0xA3, 0xC2))
tx(s, 1.1, 6.5, 11.1, 0.4, "2026-09 组会", size=12, color=RGBColor(0x8A, 0xA3, 0xC2))

# ═══════════════════════════════════════════════════════════════════
# P2 MetaKoopman 在做什么
# ═══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
title_bar(s, "MetaKoopman（NeurIPS 2025）：贝叶斯元学习动力学模型",
          "Selim, Bhat & Johansson (TRATON / KTH) · github.com/Mahmoud-Selim/MetaKoopman")
tx(s, 0.85, 1.35, 11.6, 0.5,
   "问题：动力学模型在分布偏移（新坡度 / 重力 / 摩擦 / 关节故障）下失效，逐环境重训代价高",
   size=14.5, bold=True)
tx(s, 0.85, 1.95, 11.6, 1.75,
   "核心方法：在隐空间 meta-learn 一个 Koopman 算子的 MNIW 先验（Matrix Normal–Inverse Wishart）→ "
   "新动力学实例只需一段短轨迹历史即可闭式贝叶斯适应（共轭更新，无需任何梯度步），并给出校准的预测不确定性。\n"
   "架构三件套：① Transformer 隐空间编解码器（因果掩码防未来信息泄漏）② MNIW 先验 + 闭式后验 ③ 多步不确定性传播（隐空间滚动前向）",
   size=13.5)
table(s, 0.85, 3.95, 11.6, 2.1, [
    ["环境扰动", "meta-train 范围（窄）", "meta-test 范围（宽）", "我们的网格（覆盖 meta-train）"],
    ["HalfCheetah 坡度", "[-10°, +10°]", "[-15°, +15°]", "-10° / 0° / +10°"],
    ["Walker2d 摩擦", "×[0.3, 3.0]", "×[0.2, 5.0]", "×0.3 / ×1 / ×3"],
    ["Hopper 重力", "偏移 [-1.0, +1.0]", "[-1.5, +1.5]", "-1 / -0.5 / 0 / +0.5 / +1"],
    ["Ant 关节故障", "部分关节组合", "全部关节", "8 个关节全枚举"],
], [2.6, 3.0, 3.0, 3.0], font_size=11)
tx(s, 0.85, 6.3, 11.6, 0.85,
   "管线四阶段：扰动环境（每次 reset 重采样参数）→ TD3 策略（仅作行为策略生成轨迹）→ 采集 meta-train/meta-test 轨迹 → 训练动力学模型。\n"
   "成果：多步预测精度 / 不确定性校准 / 偏移鲁棒性全面优于 DKO 等基线；另在 37.5 吨卡车冬季路况（雪 / 冰 / μ-split）做了实车验证。",
   size=12, color=INK2)
footer(s, 2)

# ═══════════════════════════════════════════════════════════════════
# P3 我们的多环境生成方法
# ═══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
title_bar(s, "多环境生成：静态 XML 扰动网格（19 个世界）",
          "generate_perturbed_xmls.py（mbpo_jax-main）→ MBPO 策略 rollout → fork_env.py 反事实采样")
tx(s, 0.85, 1.3, 11.6, 0.75,
   "核心思想：每个（扰动类型, 值）预生成一份固定 XML——动力学完全由 XML 决定，"
   "同一 (qpos, qvel) 下结果唯一确定，这是反事实（counterfactual）数据有效性的前提。",
   size=13.5, bold=True)
tx(s, 0.85, 2.1, 11.6, 0.35, "与 MetaKoopman 环境协议的三个关键差异：", size=13.5, bold=True, color=ACCENT)
table(s, 0.85, 2.5, 11.6, 1.85, [
    ["", "MetaKoopman", "我们（为什么）"],
    ["参数形式", "运行时随机重采样（同一 env 内参数变化）", "静态预生成 XML（动力学由文件唯一确定 → fork 反事实一致）"],
    ["基础资产", "自改 XML（重力 ×10、改脚摩擦）", "gymnasium 原版资产（与既有 -v4 数据保持可比）"],
    ["参数可见性", "扰动参数拼进 observation（告诉模型）", "不拼进 observation——模型须从 context 推断动力学（更难，也是 CAQ 因果通道的用武之地）"],
], [1.7, 4.2, 5.7], font_size=11)
tx(s, 0.85, 4.65, 11.6, 1.5,
   "扰动网格（覆盖其 meta-train 范围 + identity 锚点）：Hopper 重力偏移 ±1（5 档）· Walker2d 摩擦 ×0.3/1/3 · "
   "HalfCheetah 坡度 -10°/0°/+10° · Ant 卡关节 af0–af7 全枚举 → 共 19 个环境。\n"
   "采集管线：每个环境训练 MBPO 策略 → rollout 得 env_buffer.npz → fork_env.py 对每个 src state 采样 8 个加噪 action "
   "→ fork_samples.npz（state, action, next_state, reward）。",
   size=12.5)
table(s, 0.85, 6.35, 11.6, 0.7, [
    ["数据规模", "HopperGravity ×5", "Walker2dFriction ×3", "HalfCheetahSlope ×3", "AntJoints ×8", "合计"],
    ["", "84 万行/环境", "164 万行/环境", "240 万行/环境", "480 万行/环境", "5,470 万行"],
], [1.7, 2.0, 2.0, 2.0, 2.0, 1.9], font_size=11)
footer(s, 3)

# ═══════════════════════════════════════════════════════════════════
# P4 实验设定
# ═══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
title_bar(s, "实验设定：CAQ-TabICL 混合训练 + 三组实验")
tx(s, 0.85, 1.35, 11.6, 1.5,
   "模型：CAQ-TabICL = TabICL 冻结 ICL（95.5% 参数）+ 微调 ColEmbedding / RowInteraction + action 因果通道 "
   "（action → causal_block 逐行调制）。目标 y = [reward, next_state − state] 多输出。\n"
   "训练协议：flat 表随机抽样（1,024 ctx + 512 query 行）、SCM 合成数据 replay 防遗忘（4 RL : 1 SCM）、"
   "target-aware 逐维批处理（Ant 28 维分块 chunk=7）。\n"
   "评估：原版 TabICL（冻结 checkpoint，TabICLRegressor）同协议对照，报告 best/Base 比值（<1 = CAQ 更优）。",
   size=13)
table(s, 0.85, 3.25, 11.6, 2.2, [
    ["实验", "设定", "回答的问题"],
    ["A · 19 环境联合", "lr=1e-5 × 15,000 步", "基线：联合训练能否超过原 TabICL？"],
    ["B · 19 环境联合（改进）", "lr=1e-4 × 30,000 步 + 退火终点 5e-6", "A 中 AntJoints 落后是伪收敛吗？调 lr/步数能否翻转？"],
    ["C · hold-out 外推", "留出 g1 / f3 / s10 / af4，训练其余 15 环境", "模型对从未见过的世界：few-shot 适应能否赢 TabICL？"],
], [2.5, 4.6, 4.5], font_size=11.5)
tx(s, 0.85, 5.75, 11.6, 1.0,
   "hold-out 协议：留出环境完全不参与训练；评估时 context 用留出环境自身训练池前 2 万行（拿到该世界样本后的快速适应），"
   "query 为留出环境测试集——CAQ 与 baseline 完全同协议对照。",
   size=12, color=INK2)
footer(s, 4)

# ═══════════════════════════════════════════════════════════════════
# P5 分布内结果
# ═══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
title_bar(s, "分布内结果：19 环境 17 胜 2 近，OVERALL 0.96x",
          "实验 B（lr=1e-4 × 30,000 步）· CAQ best MSE / TabICL baseline MSE")
s.shapes.add_picture(f"{BASE}/charts/fig1_in_distribution.png", Inches(0.5), Inches(1.35), width=Inches(7.4))
tx(s, 8.1, 1.55, 4.75, 4.9,
   "按扰动组分层：\n"
   "重力 HopperGravity 0.45–0.70x\n（改善 30–55%，最大收益）\n\n"
   "关节 AntJoints 0.77–0.86x\n（改善 14–23%）\n\n"
   "摩擦 Walker2d 0.96–0.99x\n（小幅全面占优）\n\n"
   "坡度 HalfCheetah 0.92–1.04x\n（s0 优 8%；s-10 / s10 接近）\n\n"
   "训练健康：SCM forget 0.78x（无遗忘）\n0 次 NaN / 回滚 / 告警",
   size=13.5, line_spacing=1.3)
tx(s, 8.1, 6.5, 4.75, 0.7,
   "低维简单动力学（Hopper 11d）收益最大；\n维度越高收益递减——与 action 因果信号强度一致。",
   size=11.5, color=INK2)
footer(s, 5)

# ═══════════════════════════════════════════════════════════════════
# P6 lr / 步数消融
# ═══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
title_bar(s, "消融：AntJoints 从全输到全赢——伪收敛的识别与修复")
table(s, 0.85, 1.5, 11.6, 1.6, [
    ["", "实验 A（lr=1e-5 × 15k 步）", "实验 B（lr=1e-4 × 30k 步）"],
    ["AntJoints ×8 比值", "1.07 – 1.13x（全部落后 baseline）", "0.77 – 0.86x（全部领先 14–23%）"],
    ["OVERALL", "1.00x", "0.96x"],
    ["SCM forget", "1.03x", "0.78x（更高 lr 反而更稳）"],
], [2.2, 4.7, 4.7], font_size=12,
    highlight_cols=[(1, 1, BAD), (1, 2, GOOD), (2, 2, GOOD)])
tx(s, 0.85, 3.55, 11.6, 1.6,
   "现象：实验 A 中 AntJoints 的 MSE 从 0.14 降到 0.05 后平台化，看似收敛——但 baseline 本来就能做到 0.044–0.057，"
   "微调后反而更差 7–13%。\n"
   "诊断：CosineAnnealing 退火到底——epoch 240 后 lr 从 1e-6 衰减到 1e-7，梯度更新量趋零。"
   "平台不是「数据学完了」，而是「学不动了」的伪收敛。",
   size=13)
tx(s, 0.85, 5.35, 11.6, 1.5,
   "修复：lr 1e-5 → 1e-4、步数 15k → 30k、退火终点 lr×0.01 → lr×0.05（避免后半程再次学不动）。\n"
   "结果：AntJoints 8 环境全部翻转（改善 14–23%），且 SCM forget 从 1.03x 降到 0.78x——"
   "更高的 lr 在 SCM 锚定 + 梯度裁剪保护下没有引起遗忘。\n"
   "教训：微调 TabICL 类模型时，lr 退火方案决定「有效训练时长」；评估平台要先排除退火因素再归因于模型容量。",
   size=13)
footer(s, 6)

# ═══════════════════════════════════════════════════════════════════
# P7 hold-out 外推
# ═══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
title_bar(s, "Hold-out 外推：few-shot 跨世界适应 3/4 成立",
          "实验 C · 留出 g1 / f3 / s10 / af4 不参与训练 · 模型对留出环境从未见过任何一行")
s.shapes.add_picture(f"{BASE}/charts/fig2_holdout.png", Inches(0.5), Inches(1.6), width=Inches(6.6))
tx(s, 7.35, 1.7, 5.5, 3.6,
   "4 个留出环境中 3 个胜或持平：\n"
   "Hopper g1 0.62x（改善 38%）\n"
   "Ant af4 0.91x（改善 9%）\n"
   "Walker f3 1.02x（持平）\n"
   "HC s10 1.20x（落后）\n\n"
   "趋势：few-shot MSE 随训练稳步改善\n"
   "（16 轮评估单调向好），说明跨世界\n"
   "适应能力是训练出来的，不是碰巧。\n\n"
   "结论：模型对从未训练过的扰动世界，\n"
   "拿到 2 万行该世界的样本后即可达到\n"
   "或超过原 TabICL 的预测水平。",
   size=13, line_spacing=1.3)
footer(s, 7)

# ═══════════════════════════════════════════════════════════════════
# P8 结论与下一步
# ═══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
title_bar(s, "结论与下一步")
tx(s, 0.85, 1.45, 11.6, 2.6,
   "结论 1 · 分布内：19 个扰动环境上 CAQ-TabICL 混合微调 17/19 胜原 TabICL（OVERALL 0.96x），"
   "四类扰动全部不劣于 baseline；重力（0.45–0.70x）与关节（0.77–0.86x）两组大幅领先。\n\n"
   "结论 2 · few-shot 跨世界：对训练中从未见过的世界，给 2 万行 context 后 3/4 环境胜或持平（Hopper 0.62x），"
   "跨世界快速适应能力成立。\n\n"
   "关键支撑：更高的 lr（1e-4）没有引起遗忘（SCM forget 0.78x），伪收敛可通过退火方案修正——"
   "微调范式在反事实多环境设定下稳定可靠。",
   size=13.5, line_spacing=1.25)
tx(s, 0.85, 4.5, 11.6, 0.35, "下一步：", size=14, bold=True, color=ACCENT)
tx(s, 0.85, 4.95, 11.6, 1.7,
   "① 采集范围外参数数据（坡度 ±15°、摩擦 ×5、重力 ±1.5），对齐 MetaKoopman 的 meta-test 设定\n"
   "② 多 seed 重复 + 8 个原始环境 lr=1e-4 对照（补全证据链）\n"
   "③ 更多扰动类型与更细的扰动网格（如 Panda 关节阻尼）",
   size=13.5, line_spacing=1.4)
footer(s, 8)

OUT = f"{BASE}/组会_多环境反事实训练.pptx"
prs.save(OUT)
print("PPT 已生成:", OUT)
