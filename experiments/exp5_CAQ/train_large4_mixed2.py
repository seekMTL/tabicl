"""CAQ-TabICL 混合训练 v2：多环境 RL 数据 + SCM replay 防遗忘。

在 train_large4_mixed.py 基础上的改进（审查结论落地）:
  1. 块状环境轮流（--env_block_size）：同一环境连续多步再切换，
     减少环境间梯度交替更新的相互干扰（v4 中 Ant 2.4x 的头号嫌疑）
  2. SCM loss 可加权（--scm_loss_weight）：SCM 步 loss 尺度(~1.0)天然
     大于 RL 步(~0.25)，步数比 4:1 不等于梯度能量比
  3. 最终对比同时报告各环境历史最优（best_mse）与最后一次评估
  4. empty_cache 移到 optimizer.step() 之后（原来在 backward 前，无效）

设计要点（同 v1）:
  - 训练参数: ColEmbedding + RowInteraction(native blocks) + CAQ(action_encoder+causal_block)
    冻结: ICLearning（95.5% 参数）
  - 数据混合: 每 K 步 RL 数据（环境块状轮流）→ 1 步 SCM 合成数据（80/20 比例）
  - SCM replay: action=zeros → CAQ 零初始化保证退化为原生 RowInteraction 行为
    → 为 Col/Row 参数提供"预训练分布"锚点，防止灾难性遗忘
  - SCM 数据为回归模式（num_classes=0 → y 连续 z-scored），与回归 checkpoint 匹配
  - action_dim 统一为 MAX_ACTION_DIM=8，低维 action 右侧补零（对线性层无害）

用法:
  # 49GB 显存服务器：大 context + 大 SCM batch
  python train_large4_mixed2.py --envs "Hopper,Walker2d,HalfCheetah,Ant-v4" \\
      --n_context 2048 --n_query 512 --scm_batch_size 8 --scm_max_seq_len 1024 \\
      --env_block_size 4 --lr 1e-5 --device cuda
  # 单环境对照
  python train_large4_mixed2.py --envs "Ant-v4" --device cuda
  # 纯 RL，无 replay
  python train_large4_mixed2.py --envs "Hopper" --rl_ratio 1.0
"""

from __future__ import annotations

import sys, time, argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tabicl._sklearn.regressor import TabICLRegressor
from tabicl._sklearn.preprocessing import PreprocessingPipeline
from tabicl._model.caq_model import load_caq_model
from tabicl.prior._dataset import SCMPrior
from tabicl.prior._prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP

CKPT = "/home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt"
DATA_BASE = "/home/lizitao/project/other/sample_collect"

# 环境配置：state/action/output 维度由数据自动推断，此处仅记录目录和 epoch
ENV_CONFIGS = {
    "Hopper":      {"dir": "Hopper-v5-ensemble", "epoch": 280},
    "Walker2d":    {"dir": "walker2d",           "epoch": 300},
    "HalfCheetah": {"dir": "halfcheetah-1st",    "epoch": 200},
    "Ant-v4":      {"dir": "ant-v4",             "epoch": 300},
}

# action_dim 上限：4 个环境最大为 8（Ant），低维 action 右侧补零
# 补零对 Linear 层（加权和）无害：零输入不贡献、不稀释
MAX_ACTION_DIM = 8


# ═══════════════════════════════════════════════════════════════════
# SCM replay 采样器（回归模式）
# ═══════════════════════════════════════════════════════════════════

class RegressionSCMPrior(SCMPrior):
    """回归模式 SCM 生成器：num_classes=0 → y 保持连续 z-scored。

    与父类的区别：跳过分类 sanity_check（连续 y 的 train/test 唯一值集合
    几乎不可能相等，父类检查会死循环重试）。
    """

    @torch.no_grad()
    def generate_dataset(self, params):
        if params["prior_type"] == "mlp_scm":
            from tabicl.prior._mlp_scm import MLPSCM
            prior_cls = MLPSCM
        elif params["prior_type"] == "tree_scm":
            from tabicl.prior._tree_scm import TreeSCM
            prior_cls = TreeSCM
        else:
            raise ValueError(f"Unknown prior type {params['prior_type']}")

        while True:
            from tabicl.prior._reg2cls import Reg2Cls
            X, y = prior_cls(**params)()
            X, y = Reg2Cls(params)(X, y)

            X, y = X.unsqueeze(0), y.unsqueeze(0)
            d = torch.tensor([params["num_features"]], device=self.device, dtype=torch.long)

            X, d = self.delete_unique_features(X, d)
            if (d > 0).all():
                # 回归模式：不检查类别分布，只保证存在有效特征
                return X.squeeze(0), y.squeeze(0), d.squeeze(0)


class SCMReplaySampler:
    """SCM 合成数据采样器（回归模式），用于防遗忘 replay。

    复用 SCMPrior 的底层方法（hp_sampling / generate_dataset / get_prior /
    sample_seq_len / sample_train_size），但手动构造参数并强制 num_classes=0，
    生成与回归 checkpoint 匹配的连续 y 数据。

    与 PriorDataset 的区别：
      - 分类模式（PriorDataset 默认）：y 分箱为 0..num_classes-1
      - 回归模式（本采样器）：y 保持 z-scored 连续值
    """

    def __init__(
        self,
        batch_size: int = 4,
        min_features: int = 5,
        max_features: int = 100,
        min_seq_len: int = 128,
        max_seq_len: int = 512,
        min_train_size: float = 0.1,
        max_train_size: float = 0.9,
        n_jobs: int = 1,
    ):
        self.batch_size = batch_size
        self.min_features = min_features
        self.max_features = max_features
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.n_jobs = n_jobs

        self.scm = RegressionSCMPrior(
            batch_size=batch_size,
            batch_size_per_gp=batch_size,  # 每个数据集独立采样 HP（简化：无组共享）
            min_features=min_features,
            max_features=max_features,
            max_classes=0,  # 回归模式
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=True,  # log-uniform 采样序列长度，提高多样性
            seq_len_per_gp=False,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            replay_small=False,
            prior_type="mix_scm",  # 70% MLPSCM + 30% TreeSCM
            fixed_hp=DEFAULT_FIXED_HP,
            sampled_hp=DEFAULT_SAMPLED_HP,
            n_jobs=1,
            num_threads_per_generate=1,
            device="cpu",
        )

    def get_batch(self):
        """生成一批 SCM 数据（batch 内 seq_len / train_size 统一，便于 CAQ 前向）。

        Returns:
            X:          (B, seq_len, max_d) 特征（已截断到批次内最大有效特征数）
            y:          (B, seq_len) 连续标签（z-scored）
            d:          (B,) 每张表的有效特征数
            train_size: int 训练/测试分割点（batch 内统一）
        """
        B = self.batch_size

        # batch 内统一 seq_len 和 train_size（与 TabICL 预训练的非 per-gp 模式一致）
        seq_len = self.scm.sample_seq_len(
            self.min_seq_len, self.max_seq_len, log=True, replay_small=False
        )
        train_size = self.scm.sample_train_size(
            self.min_train_size, self.max_train_size, seq_len
        )
        # 长序列限制特征数（复用预训练的 adjust_max_features 策略）
        gp_max_features = self.scm.adjust_max_features(seq_len, self.max_features)

        # 为每个数据集构造参数（num_classes=0 → 回归模式）
        param_list = []
        for _ in range(B):
            group_sampled_hp = self.scm.hp_sampling()
            params = {
                **self.scm.fixed_hp,
                "seq_len": seq_len,
                "train_size": train_size,
                "max_features": gp_max_features,
                **{k: v() if callable(v) else v for k, v in group_sampled_hp.items()},
                "prior_type": self.scm.get_prior(),
                "num_features": round(np.random.uniform(self.min_features, gp_max_features)),
                "num_classes": 0,  # 关键：回归模式
                "device": self.scm.device,
            }
            param_list.append(params)

        # 并行生成（与 SCMPrior.get_batch 相同的 joblib 模式）
        if self.n_jobs > 1:
            import joblib
            with joblib.parallel_config(
                n_jobs=self.n_jobs, backend="loky", inner_max_num_threads=1
            ):
                results = joblib.Parallel()(
                    joblib.delayed(self.scm.generate_dataset)(params) for params in param_list
                )
        else:
            results = [self.scm.generate_dataset(params) for params in param_list]

        X_list, y_list, d_list = zip(*results)
        X = torch.stack(X_list)  # (B, seq_len, gp_max_features)
        y = torch.stack(y_list)  # (B, seq_len)
        d = torch.stack(d_list)  # (B,)

        # 截断到批次内最大有效特征数，节省显存（与 Trainer.align_micro_batch 一致）
        max_d = d.max().item()
        if X.shape[-1] > max_d:
            X = X[..., :max_d]

        return X, y, d, train_size

    def __repr__(self):
        return (
            f"SCMReplaySampler(\n"
            f"  mode: regression (num_classes=0, continuous y)\n"
            f"  batch_size: {self.batch_size}\n"
            f"  features: {self.min_features} - {self.max_features}\n"
            f"  seq_len: {self.min_seq_len} - {self.max_seq_len} (log-uniform)\n"
            f"  train_size ratio: {self.min_train_size} - {self.max_train_size}\n"
            f"  n_jobs: {self.n_jobs}\n"
            f")"
        )


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def format_mse(val: float) -> str:
    if val == 0 or val >= 1e-5:
        return f"{val:.6f}"
    return f"{val:.6e}"


def load_data(env_dir: str, epoch_idx: int):
    data = np.load(f"{DATA_BASE}/{env_dir}/env_pool_epoch_{epoch_idx:04d}.npz")
    X_state = data["states"].astype(np.float32)
    X_action = data["actions"].astype(np.float32)
    delta = (data["next_states"] - data["states"]).astype(np.float32)
    y = np.concatenate([data["rewards"][:, None], delta], axis=1).astype(np.float32)
    return X_state, X_action, y


def preprocess_data(X_state_tr, X_state_te, y_tr, y_te,
                    norm_method="none", seed=42):
    """预处理数据，对齐 TabICL 预训练的 standard_scaling。

    X_state: PreprocessingPipeline（默认 none = 仅 Z-score + OutlierClip）
    y:      逐维 Z-score 标准化
    X_action 在外层保持原样（值域 [-1,1]，action_encoder 可训练自适应）
    """
    preprocessor = PreprocessingPipeline(
        normalization_method=norm_method, random_state=seed
    )
    X_state_tr_pp = preprocessor.fit_transform(X_state_tr).astype(np.float32)
    X_state_te_pp = preprocessor.transform(X_state_te).astype(np.float32)

    y_tr_pp = np.zeros_like(y_tr)
    y_te_pp = np.zeros_like(y_te)
    y_stats = {}
    n_dims = y_tr.shape[1]
    for d in range(n_dims):
        mean = float(np.mean(y_tr[:, d]))
        std = float(np.std(y_tr[:, d]))
        if std < 1e-8:
            std = 1.0
        # 裁剪 z-score 到 ±10σ：近乎恒定的维度（如 Walker2d dim7 极值/std=59）
        # 会产生极端目标值 → MSE 梯度爆炸 → Col/Row 解冻后训练发散为 NaN。
        # TabICL 自身 SCM 管线对特征同样做裁剪（standard_scaling clip ±100）。
        y_tr_pp[:, d] = np.clip((y_tr[:, d] - mean) / std, -10.0, 10.0)
        y_te_pp[:, d] = np.clip((y_te[:, d] - mean) / std, -10.0, 10.0)
        y_stats[d] = (mean, std)
    return X_state_tr_pp, X_state_te_pp, y_tr_pp, y_te_pp, y_stats


def pad_action(X_action, target_dim=MAX_ACTION_DIM):
    """低维 action 右侧补零到 target_dim。对 Linear 层（加权和）无害。"""
    if X_action.shape[-1] < target_dim:
        X_action = np.pad(
            X_action, ((0, 0), (0, target_dim - X_action.shape[-1])),
            mode="constant",
        )
    return X_action


def create_icl_batch_shared(X_state, X_action, y, n_context, n_query, device):
    """创建共享 state/action 的 ICL batch，y 为所有输出维度批量返回。

    与 train_large3 的区别：X_action 补零到 MAX_ACTION_DIM。
    只采样一次索引，state/action 各维度共享，
    y 返回 (OUTPUT_DIMS, T) 形状的批量张量，用于 CAQ 模型的并行 ICL 训练。
    """
    n_total = len(X_state)
    all_idx = np.random.permutation(n_total)
    ctx_idx = all_idx[:n_context]
    q_idx = all_idx[n_context : n_context + n_query]
    Xs_state = np.concatenate([X_state[ctx_idx], X_state[q_idx]])
    Xs_action = pad_action(
        np.concatenate([X_action[ctx_idx], X_action[q_idx]])
    )
    yb = np.stack(
        [np.concatenate([y[ctx_idx, d], y[q_idx, d]]) for d in range(y.shape[1])],
        axis=0,
    )
    return (
        torch.from_numpy(Xs_state).float().unsqueeze(0).to(device),
        torch.from_numpy(Xs_action).float().unsqueeze(0).to(device),
        torch.from_numpy(yb).float().to(device),
    )


def evaluate_env(model, env_data, device, criterion, n_context_eval):
    """评估单个环境：从训练池取前 n_context_eval 样本做 ICL 上下文，预测测试集。

    返回 {dim: MSE(原始尺度)} 与平均值。y_stat 用于将标准化空间的预测
    逆变换回原始尺度，与 Baseline 公平比较。
    """
    X_state_tr = env_data["X_state_tr_pp"]
    X_action_tr = env_data["X_action_tr"]
    y_tr_pp = env_data["y_tr_pp"]
    X_state_te = env_data["X_state_te_pp"]
    X_action_te = env_data["X_action_te"]
    y_te_pp = env_data["y_te_pp"]
    y_te_raw = env_data["y_te_raw"]
    y_tr_raw = env_data["y_tr_raw"]
    y_stats = env_data["y_stats"]
    output_dims = env_data["output_dims"]

    n_ctx = min(n_context_eval, len(X_state_tr))
    model.eval()
    mse_dims = {}
    with torch.no_grad():
        for d in range(output_dims):
            eval_state = np.concatenate([X_state_tr[:n_ctx], X_state_te])
            eval_action = pad_action(np.concatenate([X_action_tr[:n_ctx], X_action_te]))
            eval_y = np.concatenate([y_tr_pp[:n_ctx, d], y_te_pp[:, d]])  # 标准化目标

            X_state_t = torch.from_numpy(eval_state).float().unsqueeze(0).to(device)
            X_action_t = torch.from_numpy(eval_action).float().unsqueeze(0).to(device)
            yb = torch.from_numpy(eval_y).float().unsqueeze(0).to(device)
            y_train = yb[:, :n_ctx]

            out = model(X_state_t, X_action_t, y_train=y_train)
            pred = out.mean(dim=-1).squeeze(0)  # (n_test,)

            # 逆变换到原始尺度
            mean, std = y_stats[d]
            pred = pred * std + mean
            raw = np.concatenate([y_tr_raw[:n_ctx, d], y_te_raw[:, d]])
            target = torch.from_numpy(raw).float().to(device)[n_ctx:]
            mse_dims[d] = criterion(pred, target).item()

    return mse_dims


def evaluate_baseline_full(X_tr, X_te, y_tr, y_te, device="cuda", n_context_eval=100000):
    """Baseline: 输出维度 x TabICLRegressor，state+action 拼接作为特征输入。"""
    n_ctx = min(n_context_eval, len(X_tr))
    X_ctx, y_ctx = X_tr[:n_ctx], y_tr[:n_ctx]
    mse = np.zeros(y_tr.shape[1])
    for d in range(y_tr.shape[1]):
        reg = TabICLRegressor(n_estimators=4, model_path=CKPT, device=device, random_state=42)
        reg.fit(X_ctx, y_ctx[:, d])
        pred = reg.predict(X_te)
        mse[d] = np.mean((pred - y_te[:, d]) ** 2)
    return mse


# ═══════════════════════════════════════════════════════════════════
# 检查点
# ═══════════════════════════════════════════════════════════════════

def get_trainable_state_dict(model):
    """只提取可训练参数，排除冻结的 icl_predictor（27.3M → 减小检查点体积）。"""
    return {k: v for k, v in model.state_dict().items()
            if not k.startswith("icl_predictor")}


def save_checkpoint(save_dir, model, optimizer, scheduler, epoch, mse_all, train_args, is_best=False):
    save_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "mse_caq": {env: {str(d): float(v) for d, v in mse.items()} for env, mse in mse_all.items()},
        "model_state": get_trainable_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "train_args": train_args,
    }
    path = save_dir / f"ckpt_epoch{epoch:04d}.pt"
    torch.save(state, path)
    if is_best:
        torch.save(state, save_dir / "best.pt")

    # 清理旧检查点（保留最新 3 个）
    ckpts = sorted(save_dir.glob("ckpt_epoch*.pt"))
    for old in ckpts[:-3]:
        old.unlink()


def save_progress(save_dir, mse_all, mse_base, epoch, scm_loss, forget_score,
                  scm_loss_baseline, elapsed):
    save_dir.mkdir(parents=True, exist_ok=True)
    progress = {
        "epoch": epoch,
        "mse_caq": {env: {str(d): float(v) for d, v in mse.items()} for env, mse in mse_all.items()},
        "mse_caq_avg": {env: float(np.mean(list(mse.values()))) for env, mse in mse_all.items()},
        "mse_base_avg": {env: float(np.mean(m)) for env, m in mse_base.items()} if mse_base else {},
        "scm_loss": float(scm_loss),
        "forget_score": float(forget_score),
        "scm_loss_baseline": float(scm_loss_baseline),
        "elapsed": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(save_dir / "progress.json", "w") as f:
        json.dump(progress, f, indent=2)


def load_progress(save_dir):
    path = save_dir / "progress.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════
# 训练步骤
# ═══════════════════════════════════════════════════════════════════

def rl_train_step(model, env_data, criterion, device, n_context, n_query):
    """RL 训练步：采样 ICL batch，ColEmbedding+RowInteraction 一次，
    ICL 阶段单次调用全维并行推理（与 train_large3_multenv 相同的模式），
    逐维 MSE 求和后取平均。

    注意：不用分块累积 loss——分块会让每个块的 ICL 计算图同时保留到
    反向传播（峰值 = 块数 × 单块显存），导致 OOM。
    """
    X_state, X_action, yb_all = create_icl_batch_shared(
        env_data["X_state_tr_pp"], env_data["X_action_tr"], env_data["y_tr_pp"],
        n_context, n_query, device,
    )
    y_train_all = yb_all[:, :n_context]  # (OUTPUT_DIMS, n_context)
    output_dims = env_data["output_dims"]

    # ColEmbedding + RowInteraction 只做一次（target_aware=False，y 仅用于提取 train_size）
    dummy_y = y_train_all[0:1]
    representations = model._forward_embeddings(X_state, X_action, y_train=dummy_y)

    # ICL 单次前向全维（与 train_large3_multenv_colrow 相同的已验证模式）
    representations_batch = representations.repeat(output_dims, 1, 1)
    out = model.icl_predictor(representations_batch, y_train=y_train_all)

    total_loss = torch.tensor(0.0, device=device)
    for d in range(output_dims):
        total_loss = total_loss + criterion(out[d].mean(dim=-1), yb_all[d, n_context:])

    return total_loss / output_dims  # 按维度平均，保持各环境 loss 尺度一致


def scm_train_step(model, scm_sampler, criterion, device, loss_weight=1.0):
    """SCM replay 训练步：action=zeros 的合成数据前向 + MSE。

    目的不是 SCM 任务精度，而是为 Col/Row 参数提供"预训练分布"的
    梯度锚点，防止其在 RL 数据上漂移（防遗忘）。

    loss_weight: SCM loss 的加权系数。SCM 步 loss 尺度(~1.0)天然大于
    RL 步(~0.25)，步数比 4:1 不等于梯度能量比，可用此参数调节锚定强度。

    注意：不传 d（feature grouping 模式下 ColEmbedding 不支持 d）。
    X 中各表超出有效特征数的列为零填充（delete_unique_features 保证
    有效特征在前、零填充在后），与预训练时的处理完全一致。
    """
    X, y, d, train_size = scm_sampler.get_batch()
    X = X.to(device)
    y = y.to(device)

    y_train = y[:, :train_size]
    y_test = y[:, train_size:]

    # X_action=None → forward 内部构造零动作 → CAQ 退化为原生 RowInteraction
    pred = model(X, y_train=y_train)
    loss = criterion(pred.mean(dim=-1), y_test) * loss_weight

    return loss


@torch.no_grad()
def eval_scm_loss(model, scm_sampler, criterion, device, n_batches=2):
    """遗忘监控：在 SCM 数据上计算平均 loss（eval 模式）。

    forget_score = 当前 SCM loss / 训练起始 SCM loss。
    SCM loss 上升说明 Col/Row 参数偏离预训练分布。
    病态批次（loss 非有限）自动重采样，最多 3 次。
    """
    model.eval()
    losses = []
    for _ in range(n_batches):
        for attempt in range(3):
            X, y, d, train_size = scm_sampler.get_batch()
            X = X.to(device)
            y = y.to(device)
            y_train = y[:, :train_size]
            y_test = y[:, train_size:]
            pred = model(X, y_train=y_train)  # 不传 d（feature grouping 模式）
            val = criterion(pred.mean(dim=-1), y_test).item()
            if np.isfinite(val):
                losses.append(val)
                break
        else:
            print(f"  ⚠️  SCM 评估连续 3 批非有限，跳过")
    return float(np.mean(losses)) if losses else float("nan")


def snapshot_trainable(model):
    """快照所有可训练参数（用于参数被 NaN 污染后的回滚）。"""
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def restore_trainable(model, snapshot):
    """从快照恢复可训练参数。"""
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data.copy_(snapshot[n])


def params_finite(model):
    """检查所有可训练参数是否有限。"""
    return all(torch.isfinite(p).all() for n, p in model.named_parameters() if p.requires_grad)


def train_mixed(
    model, optimizer, scheduler, criterion,
    env_datasets, scm_sampler,
    device, save_dir, train_args,
    n_context=1024, n_query=512,
    n_steps_per_epoch=50, n_epochs=300, n_context_eval=20000,
    rl_ratio=0.8, eval_interval=20,
    env_block_size=4, scm_loss_weight=1.0,
    start_epoch=0, best_mse=None,
    mse_base=None, t_start=None, prev_elapsed=0,
    scm_loss_baseline=None,
):
    """混合训练主循环。

    每步：K 步 RL（环境块状轮流）→ 1 步 SCM replay（K = round(1/(1-rl_ratio)) - 1）。
    环境块状轮流：同一环境连续 env_block_size 个 RL 步再切换，减少环境间
    梯度交替更新造成的相互干扰（v4 中 Ant 2.4x 的头号嫌疑）。
    每 eval_interval epoch：评估各环境 MSE + SCM 遗忘监控。
    """
    env_names = list(env_datasets.keys())
    if rl_ratio >= 1.0:
        cycle_len = None  # 纯 RL，无 SCM replay
    else:
        cycle_len = max(2, round(1.0 / (1.0 - rl_ratio)))  # 0.8 → 5（4 RL + 1 SCM）

    if best_mse is None:
        best_mse = {env: float("inf") for env in env_names}
    if scm_loss_baseline is None:
        scm_loss_baseline = 1.0  # 兜底，正常会在训练前计算

    global_step = 0
    rl_step_count = 0
    forget_score = 1.0
    skipped_steps = 0   # 病态批次跳过计数（每 epoch 重置）
    rolled_back = 0     # 参数回滚计数（每 epoch 重置）

    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_losses = []
        skipped_steps = 0
        rolled_back = 0

        for _ in range(n_steps_per_epoch):
            is_scm_step = cycle_len is not None and global_step % cycle_len == cycle_len - 1

            if is_scm_step:
                loss = scm_train_step(model, scm_sampler, criterion, device,
                                      loss_weight=scm_loss_weight)
            else:
                # 块状轮流：每 env_block_size 个 RL 步切换环境
                env_name = env_names[(rl_step_count // env_block_size) % len(env_names)]
                rl_step_count += 1
                loss = rl_train_step(
                    model, env_datasets[env_name], criterion, device,
                    n_context, n_query,
                )

            # 病态批次保护：loss 非有限 → 跳过本步（不反向传播、不更新参数）。
            # 罕见病态批次（随机合成数据/极端数值）不应杀死整个训练。
            if not torch.isfinite(loss):
                skipped_steps += 1
                if skipped_steps > 10:
                    raise RuntimeError(
                        f"连续 {skipped_steps} 步 loss 非有限（epoch {epoch}），"
                        f"参数可能已污染，停止训练。最近有效检查点在 best.pt。"
                    )
                global_step += 1
                continue

            # 更新前快照参数，若本步更新导致参数 NaN 则回滚
            snapshot = snapshot_trainable(model)

            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪（TabICL 官方预训练默认值 1.0）：防止极端 y 目标
            # 导致的梯度尖峰推动 Col/Row 权重缓慢漂移最终溢出
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 参数健康检查：梯度数值问题可能让参数变 NaN（loss 仍有限）
            if not params_finite(model):
                restore_trainable(model, snapshot)
                rolled_back += 1
                if rolled_back > 10:
                    raise RuntimeError(
                        f"连续 {rolled_back} 步参数被 NaN 污染（epoch {epoch}），停止训练。"
                    )
                global_step += 1
                continue

            # SCM 步更新完成后释放缓存（激活峰值大，图已释放此时才有效）
            if is_scm_step:
                torch.cuda.empty_cache()

            epoch_losses.append(loss.item())
            global_step += 1

        scheduler.step()
        avg_loss = np.mean(epoch_losses)

        if (epoch + 1) % eval_interval == 0 or epoch == start_epoch:
            torch.cuda.empty_cache()  # 释放训练缓存，为评估腾出显存

            # ── SCM 遗忘监控 ──
            scm_loss = eval_scm_loss(model, scm_sampler, criterion, device)
            forget_score = scm_loss / max(scm_loss_baseline, 1e-12)

            # ── 各环境评估 ──
            improved_any = False
            mse_all = {}
            for env_name in env_names:
                mse_dims = evaluate_env(
                    model, env_datasets[env_name], device, criterion, n_context_eval
                )
                mse_all[env_name] = mse_dims
                avg_mse = np.mean(list(mse_dims.values()))
                if avg_mse < best_mse[env_name]:
                    best_mse[env_name] = avg_mse
                    improved_any = True

            # ── 打印 ──
            skip_info = f" | 跳过{skipped_steps} 回滚{rolled_back}" if (skipped_steps or rolled_back) else ""
            print(f"  epoch {epoch+1:4d}/{n_epochs} | loss={avg_loss:.6f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e} | "
                  f"SCM loss={scm_loss:.4f} (forget={forget_score:.2f}x){skip_info}")
            for env_name in env_names:
                mse_dims = mse_all[env_name]
                dims_str = " ".join(
                    f"{d}:{format_mse(m)}" for d, m in sorted(mse_dims.items())
                )
                print(f"    [{env_name:<12}] avg={format_mse(np.mean(list(mse_dims.values())))} | {dims_str}")

            # 遗忘告警
            if forget_score > 1.2:
                print(f"  ⚠️  遗忘告警: SCM loss 上升至 {forget_score:.2f}x 基线")

            if save_dir:
                save_checkpoint(save_dir, model, optimizer, scheduler,
                                epoch + 1, mse_all, train_args, is_best=improved_any)
                save_progress(save_dir, mse_all, mse_base, epoch + 1,
                              scm_loss, forget_score, scm_loss_baseline,
                              prev_elapsed + (time.time() - t_start if t_start else 0))

    return mse_all, best_mse


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CAQ-TabICL 混合训练（多环境 RL + SCM replay）")
    # 环境
    parser.add_argument("--envs", type=str, default="Hopper,Walker2d,HalfCheetah,Ant-v4",
                        help=f"环境列表（逗号分隔），可选: {list(ENV_CONFIGS.keys())}")
    # 训练
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--n_context", type=int, default=1024)
    parser.add_argument("--n_query", type=int, default=512)
    parser.add_argument("--n_context_eval", type=int, default=20000,
                        help="评估时 ICL 上下文样本数（默认 20K；完整评估可设 100000）")
    parser.add_argument("--n_test", type=int, default=10000, help="各环境统一测试集样本数")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--n_steps_per_epoch", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=20)
    # 混合训练
    parser.add_argument("--rl_ratio", type=float, default=0.8,
                        help="RL 数据比例（0.8 → 每 4 步 RL + 1 步 SCM；1.0 → 纯 RL）")
    parser.add_argument("--env_block_size", type=int, default=4,
                        help="块状环境轮流：同一环境连续多少个 RL 步再切换。"
                             "v4 每步切换环境，环境间梯度交替更新相互干扰（Ant 2.4x 嫌疑）")
    parser.add_argument("--scm_loss_weight", type=float, default=1.0,
                        help="SCM loss 加权系数。SCM 步 loss 尺度(~1.0)天然大于 RL 步(~0.25)，"
                             "步数比 4:1 不等于梯度能量比，可下调以减弱锚定")
    parser.add_argument("--scm_batch_size", type=int, default=4,
                        help="SCM replay 的 batch size（小 batch 控制 ColEmbedding 激活峰值）")
    parser.add_argument("--scm_min_seq_len", type=int, default=128)
    parser.add_argument("--scm_max_seq_len", type=int, default=512)
    parser.add_argument("--scm_n_jobs", type=int, default=1,
                        help="SCM 生成并行度。默认 1（串行）：joblib loky 的每个 worker 会"
                             "初始化 346MB 的 CUDA context，8 进程 = ~2.8GB 显存被白占")
    # 冻结
    parser.add_argument("--freeze_col_embedder", action="store_true", default=False,
                        help="冻结 col_embedder（默认 False，即训练）")
    parser.add_argument("--freeze_native_blocks", action="store_true", default=False,
                        help="冻结 row_interactor 原生 blocks（默认 False，即训练）")
    # 其他
    parser.add_argument("--norm_method", type=str, default="none",
                        choices=["none", "power", "quantile", "quantile_rtdl", "robust"])
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="从指定目录恢复训练")
    parser.add_argument("--skip_baseline", action="store_true", default=False,
                        help="跳过 Baseline 评估（省时间）")
    args = parser.parse_args()

    env_names = [e.strip() for e in args.envs.split(",") if e.strip()]
    for env in env_names:
        assert env in ENV_CONFIGS, f"未知环境 {env}，可选: {list(ENV_CONFIGS.keys())}"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir) if args.save_dir else None

    # ── 加载并预处理各环境数据 ──
    print("📦 加载 RL 环境数据...")
    env_datasets = {}
    for env_name in env_names:
        cfg = ENV_CONFIGS[env_name]
        X_state, X_action, y = load_data(cfg["dir"], cfg["epoch"])

        rng = np.random.RandomState(args.seed)
        idx = rng.permutation(len(X_state))
        X_state, X_action, y = X_state[idx], X_action[idx], y[idx]

        n_train = int(len(X_state) * 0.8)
        n_test = min(args.n_test, len(X_state) - n_train)
        X_state_tr, X_state_te = X_state[:n_train], X_state[n_train:n_train + n_test]
        X_action_tr, X_action_te = X_action[:n_train], X_action[n_train:n_train + n_test]
        y_tr, y_te = y[:n_train], y[n_train:n_train + n_test]

        X_state_tr_pp, X_state_te_pp, y_tr_pp, y_te_pp, y_stats = preprocess_data(
            X_state_tr, X_state_te, y_tr, y_te, norm_method=args.norm_method, seed=args.seed,
        )

        env_datasets[env_name] = {
            "X_state_tr_pp": X_state_tr_pp,
            "X_state_te_pp": X_state_te_pp,
            "X_action_tr": X_action_tr,
            "X_action_te": X_action_te,
            "y_tr_pp": y_tr_pp,
            "y_te_pp": y_te_pp,
            "y_tr_raw": y_tr,
            "y_te_raw": y_te,
            "y_stats": y_stats,
            "output_dims": y.shape[1],
            "X_tr_raw": np.concatenate([X_state_tr, X_action_tr], axis=1),
            "X_te_raw": np.concatenate([X_state_te, X_action_te], axis=1),
        }
        print(f"  [{env_name:<12}] state={X_state.shape[1]}d action={X_action.shape[1]}d "
              f"target={y.shape[1]}d | 训练 {len(X_state_tr):,} 测试 {len(X_state_te):,}")

    # ── Baseline（可选）──
    mse_base = {}
    if not args.skip_baseline:
        print(f"\n📊 Baseline 评估中...")
        for env_name in env_names:
            ed = env_datasets[env_name]
            mse_base[env_name] = evaluate_baseline_full(
                ed["X_tr_raw"], ed["X_te_raw"], ed["y_tr_raw"], ed["y_te_raw"],
                device=str(device), n_context_eval=args.n_context_eval,
            )
            print(f"  [{env_name:<12}] avg MSE={format_mse(np.mean(mse_base[env_name]))}")

    # ── 创建模型 ──
    print(f"\n🧬 创建 CAQ 模型 (action_dim={MAX_ACTION_DIM})...")
    caq = load_caq_model(
        CKPT, action_dim=MAX_ACTION_DIM, device=device,
        freeze_col_embedder=args.freeze_col_embedder,
        freeze_native_blocks=args.freeze_native_blocks,
        freeze_icl=True,
    )
    n_trainable = sum(p.numel() for p in caq.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in caq.parameters())
    print(f"  可训练参数: {n_trainable:,} / {n_total:,} ({n_trainable / n_total * 100:.1f}%)")
    for name, module in [("col_embedder", caq.col_embedder),
                         ("row_interactor", caq.row_interactor),
                         ("icl_predictor", caq.icl_predictor)]:
        n_t = sum(p.numel() for p in module.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in module.parameters())
        print(f"    {name:<20} 可训练 {n_t:,} / {n_all:,}")

    optimizer = AdamW(caq.trainable_parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=args.lr * 0.01)
    criterion = nn.MSELoss()

    # ── SCM replay 采样器 ──
    scm_sampler = SCMReplaySampler(
        batch_size=args.scm_batch_size,
        min_seq_len=args.scm_min_seq_len,
        max_seq_len=args.scm_max_seq_len,
        n_jobs=args.scm_n_jobs,
    )
    print(f"\n{scm_sampler}")

    # ── 恢复训练 ──
    start_epoch, best_mse, prev_elapsed, scm_loss_baseline = 0, None, 0, None
    if args.resume and save_dir:
        best_path = save_dir / "best.pt"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device)
            caq.load_state_dict(ckpt["model_state"], strict=False)
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt["epoch"]
            best_mse = {
                env: float(np.mean(list({int(k): v for k, v in mse.items()}.values())))
                for env, mse in ckpt.get("mse_caq", {}).items()
            }
            progress = load_progress(save_dir)
            if progress:
                prev_elapsed = progress.get("elapsed", 0)
                scm_loss_baseline = progress.get("scm_loss_baseline", None)
                print(f"  📂 恢复: epoch={start_epoch}, 此前训练 {prev_elapsed / 3600:.1f}h")
            else:
                print(f"  📂 从 best.pt 恢复: epoch={start_epoch} (progress.json 缺失)")
    else:
        print("\n  从头开始训练")

    # ── 遗忘基线：训练前评估 SCM loss ──
    if scm_loss_baseline is None:
        print("\n🔬 计算 SCM 遗忘基线（训练前 SCM loss）...")
        scm_loss_baseline = eval_scm_loss(caq, scm_sampler, criterion, device)
        print(f"  SCM baseline loss = {scm_loss_baseline:.6f}")
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_dir / "scm_baseline.txt", "w") as f:
                f.write(str(scm_loss_baseline))

    train_args = {
        "lr": args.lr, "n_epochs": args.n_epochs, "n_context": args.n_context,
        "n_query": args.n_query, "n_steps_per_epoch": args.n_steps_per_epoch,
        "rl_ratio": args.rl_ratio, "weight_decay": 1e-5,
        "norm_method": args.norm_method,
        "freeze_col_embedder": args.freeze_col_embedder,
        "freeze_native_blocks": args.freeze_native_blocks,
        "envs": env_names, "scm_batch_size": args.scm_batch_size,
        "env_block_size": args.env_block_size,
        "scm_loss_weight": args.scm_loss_weight,
    }

    print(f"\n{'='*60}")
    print(f"  🧬 混合训练（{len(env_names)} 环境块状轮流 block={args.env_block_size}，"
          f"rl_ratio={args.rl_ratio}，SCM 权重={args.scm_loss_weight}）")
    print(f"  {args.n_steps_per_epoch} steps/epoch × {args.n_epochs} epochs = "
          f"{args.n_steps_per_epoch * args.n_epochs:,} 步")
    print(f"  col_embedder={'❄️冻结' if args.freeze_col_embedder else '🔥训练'}，"
          f"native_blocks={'❄️冻结' if args.freeze_native_blocks else '🔥训练'}，"
          f"ICL=❄️冻结")
    if save_dir:
        print(f"  检查点: {save_dir}/best.pt")

    # ── 训练 ──
    t0 = time.time()
    mse_all = None
    try:
        mse_all, best_mse_final = train_mixed(
            caq, optimizer, scheduler, criterion,
            env_datasets, scm_sampler,
            device, save_dir=save_dir, train_args=train_args,
            n_context=args.n_context, n_query=args.n_query,
            n_steps_per_epoch=args.n_steps_per_epoch, n_epochs=args.n_epochs,
            n_context_eval=args.n_context_eval,
            rl_ratio=args.rl_ratio, eval_interval=args.eval_interval,
            env_block_size=args.env_block_size, scm_loss_weight=args.scm_loss_weight,
            start_epoch=start_epoch, best_mse=best_mse,
            mse_base=mse_base, t_start=t0, prev_elapsed=prev_elapsed,
            scm_loss_baseline=scm_loss_baseline,
        )
    except Exception as e:
        print(f"  ❌ 训练异常: {e}")
        import traceback; traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\n  ✅ 训练完成，耗时 {elapsed/3600:.1f}h")

    # ── 最终结果 ──
    if mse_all is None:
        print("\n  ❌ 无有效评估结果")
        return

    print(f"\n{'='*70}")
    print(f"  📊 最终对比（{len(env_names)} 环境）")
    print(f"  {'Env':<12} {'CAQ last':>12} {'CAQ best':>12} {'Base':>12} {'best/Base':>10} {'last/Base':>10}")
    print(f"  {'-'*64}")
    overall_c, overall_b, overall_best = [], [], []
    for env_name in env_names:
        c_last = np.mean(list(mse_all[env_name].values()))
        c_best = best_mse_final.get(env_name, c_last)
        overall_c.append(c_last)
        overall_best.append(c_best)
        if mse_base and env_name in mse_base:
            b = np.mean(mse_base[env_name])
            overall_b.append(b)
            print(f"  {env_name:<12} {format_mse(c_last):>12} {format_mse(c_best):>12} "
                  f"{format_mse(b):>12} {c_best/b:>10.2f}x {c_last/b:>10.2f}x")
        else:
            print(f"  {env_name:<12} {format_mse(c_last):>12} {format_mse(c_best):>12} "
                  f"{'-':>12} {'-':>10} {'-':>10}")
    if overall_b and len(overall_b) == len(overall_c):
        print(f"  {'-'*64}")
        print(f"  {'OVERALL':<12} {format_mse(np.mean(overall_c)):>12} {format_mse(np.mean(overall_best)):>12} "
              f"{format_mse(np.mean(overall_b)):>12} {np.mean(overall_best)/np.mean(overall_b):>10.2f}x "
              f"{np.mean(overall_c)/np.mean(overall_b):>10.2f}x")

    if save_dir:
        print(f"\n  检查点保存在: {save_dir}/")


if __name__ == "__main__":
    main()
