"""Target-aware industrial continuation training with original regression SCM replay.

v2（2026-09-21）相对上一轮 seed42（caq-concat）的变更，训练思路对齐
exp5_CAQ/train_large6_counterfactual.py：
  - CAQ 默认 state-only：action 不拼进特征表，只走 causal 通道（原设计意图）；
    旧行为保留为 --model-kind caq-concat 供消融
  - 新增 --lr-schedule cosine（默认）：CosineAnnealingLR 从 lr 退火到 lr*0.05，
    解决固定 1e-4 后期权重摆动导致的验证指标大幅波动
    （seed42 CAQ 23k best 0.2169 后 30k 退化到 0.5199）
  - 病态批次保护（仿 train_large6）：非有限 loss/梯度跳过该步，NaN 参数回滚，
    连续 10 步才停止，不再单步 raise 杀死长训练
"""
import argparse
import importlib.metadata
import json
import random
import time
import hashlib
from pathlib import Path
import numpy as np
import torch
from industrial_adapter import ICDS_ROOT, to_target_batch
from industrial_task_sampler import IndustrialTaskSampler
from industrial_common import evaluate, evaluate_scm, scm_batch, isolated_seed
from icds.noise import stable_seed
from icds.storage import write_json, checksum
from icds.validate import validate
from tabicl._model.caq_model import load_caq_model
from train_large6_counterfactual import SCMReplaySampler

DEFAULT_CHECKPOINT = "/home/lizitao/project/mbpo_pyt_tabpfn/ckpt/tabiclv2/tabicl-regressor-v2-20260212.ckpt"


class ActionConcatTabICL(torch.nn.Module):
    """Original architecture with actions in the same feature channel."""
    def __init__(self, native):
        super().__init__()
        self.native = native

    def forward(self, X, X_action=None, y_train=None):
        if X_action is not None:
            X = torch.cat([X, X_action], dim=-1)
        return self.native(X, y_train=y_train)


class ActionConcatCAQ(ActionConcatTabICL):
    """Expose action to pretrained columns and the separate CAQ channel.

    v2 起仅 --model-kind caq-concat 使用（消融对照）。CAQ 默认设计（caq）
    是 action 不拼进特征表，只走 causal 通道——见 build_model。
    """
    def forward(self, X, X_action=None, y_train=None):
        if X_action is not None:
            X = torch.cat([X, X_action], dim=-1)
        return self.native(X, X_action=X_action, y_train=y_train)


def build_model(checkpoint, device, model_kind="caq"):
    """caq: state-only（action 只走 CAQ causal 通道，不拼进特征表，原设计意图）；
    tabicl: action 拼接为普通特征列（原生 TabICL 信息流，对照组）；
    caq-concat: 旧变体（action 同时拼接特征 + 走 causal 通道），仅保留做消融。"""
    if model_kind == "tabicl":
        from tabicl._model.tabicl import TabICL
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        native = TabICL(**state["config"])
        native.load_state_dict(state["state_dict"], strict=True)
        native.col_embedder.target_aware = True
        for param in native.icl_predictor.parameters():
            param.requires_grad = False
        return ActionConcatTabICL(native.float().to(device))
    model = load_caq_model(checkpoint, action_dim=8, device=device,
                          freeze_col_embedder=False, freeze_native_blocks=False,
                          freeze_icl=True, target_aware=True)
    return ActionConcatCAQ(model) if model_kind == "caq-concat" else model


def snapshot_trainable(model):
    """快照所有可训练参数（仿 train_large6，用于 NaN 污染后的回滚）。"""
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def restore_trainable(model, snapshot):
    """从快照恢复可训练参数。"""
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data.copy_(snapshot[n])


def params_finite(model):
    """检查所有可训练参数是否有限。"""
    return all(torch.isfinite(p).all() for n, p in model.named_parameters() if p.requires_grad)


def save_checkpoint(path, model, optimizer, scheduler, args, step):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "base_checkpoint": args.checkpoint, "args": vars(args), "step": step}, tmp)
    tmp.replace(path)


def paired_supervision(pred, query, task, sampler, effect_loss="mse", effect_scope="all"):
    """Query siblings stay together; graph used only as training supervision."""
    n = task["context_size"]
    ids = task["group_ids"][n:]
    if len(ids) % 2 or not np.array_equal(ids[::2, 2], ids[1::2, 2]):
        raise ValueError("Expected adjacent query sibling pairs")
    predicted_effect = pred[:, 1::2] - pred[:, ::2]
    actual_effect = query[:, 1::2] - query[:, ::2]
    effect = torch.nn.functional.huber_loss(predicted_effect, actual_effect) if effect_loss == "huber" else torch.mean((predicted_effect - actual_effect)**2)
    wid = int(ids[0, 0])
    program = json.loads((sampler.root / sampler.shards[wid]["program"]).read_text())
    if task.get("information_track") == "observed_history":
        # Forecast track only until its available-input causal mean is validated.
        return predicted_effect.sum() * 0, predicted_effect.sum() * 0
    if effect_scope == "deterministic" and program["family"] == "hybrid_line":
        return predicted_effect.sum() * 0, predicted_effect.sum() * 0
    if program.get("graph_precision", "exact") != "exact":
        return effect, predicted_effect.sum() * 0
    graph = np.asarray(task["step_action_graph"],dtype=int) if "step_action_graph" in task else np.asarray(program["step_action"], dtype=int)[task["target_indices"]]
    actions = task["X_action"][0, n:, :graph.shape[1]]
    changed = np.abs(actions[1::2] - actions[::2]) > 1e-6
    zero_mask = torch.as_tensor((changed.astype(int) @ graph.T == 0).T, device=pred.device)
    zero = predicted_effect[zero_mask].square().mean() if zero_mask.any() else predicted_effect.sum() * 0
    return effect, zero


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--sensor-noise-multiplier", type=float, default=1.)
    p.add_argument("--information-track", choices=["full_state", "observed_history"], default="full_state")
    p.add_argument("--history-ratio", type=float, default=0.)
    p.add_argument("--history-length", type=int, default=8)
    p.add_argument("--history-families", nargs="+", choices=["discrete","physical_ode","hybrid_line"])
    p.add_argument("--external-context-mode", choices=["original", "observed", "hidden"], default="original")
    p.add_argument("--target-floor-profile", choices=["original", "event_v2"], default="original")
    p.add_argument("--prior", choices=["A1", "A2", "A4", "A5"], default="A4")
    p.add_argument("--model-kind", choices=["caq", "tabicl", "caq-concat"], default="caq")
    p.add_argument("--paired-query", action="store_true", help="Matched A4/A5 query sources and actions")
    p.add_argument("--industrial-sampling-protocol", choices=["factual", "episode_flat", "row_flat"],
                   default="factual",
                   help="episode_flat: episode-isolated random rows from all factual/counterfactual branches")
    p.add_argument("--effect-loss", choices=["mse", "huber"], default="mse")
    p.add_argument("--effect-scope", choices=["all", "deterministic"], default="all")
    p.add_argument("--effect-weight", type=float, default=0.25)
    p.add_argument("--zero-weight", type=float, default=0.1)
    p.add_argument("--resume", help="Continue model and AdamW state from an industrial checkpoint")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cuda-memory-fraction", type=float, default=1.,
                   help="Per-process CUDA allocator budget; useful for matched runs sharing one GPU")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--context-size", type=int, default=64)
    p.add_argument("--query-size", type=int, default=32)
    p.add_argument("--absolute-state", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--input-scale-profile", choices=["legacy", "industrial_v2"], default="legacy",
                   help="Fixed absolute input encoding scales; target metric references are unchanged")
    p.add_argument("--persistence-residual", action=argparse.BooleanOptionalAction, default=False,
                   help="Parameterize the same forecast target as a correction to public-current persistence")
    p.add_argument("--persistence-residual-scope", choices=["all", "next_observation"], default="all")
    p.add_argument("--target-anchor", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--target-selector", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--targets-per-task", type=int, default=2)
    p.add_argument("--replay-ratio", type=float, default=0.2)
    p.add_argument("--loss", choices=["mse", "huber"], default="mse")
    p.add_argument("--scm-max-features", type=int, default=12)
    p.add_argument("--skip-paired-eval", action="store_true")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--schedule-steps", type=int, default=None, help="Cosine horizon; defaults to total steps")
    p.add_argument("--lr-schedule", choices=["cosine", "none"], default="cosine",
                   help="cosine: CosineAnnealingLR 退火至 lr*0.05（仿 train_large6，默认）；none: 固定 lr")
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--save-latest", action="store_true",
                   help="Also keep a resumable latest.pt; disabled by default to avoid a third large checkpoint")
    args = p.parse_args()
    wall_started = time.perf_counter()
    if not 0 < args.cuda_memory_fraction <= 1:
        p.error("cuda-memory-fraction must be in (0, 1]")
    if not 0 <= args.history_ratio <= 1 or args.history_length < 1 or not np.isfinite(args.sensor_noise_multiplier) or args.sensor_noise_multiplier <= 0:
        p.error("Invalid history ratio/length")
    if args.prior == "A5" and (args.query_size % 2 or args.effect_weight < 0 or args.zero_weight < 0):
        p.error("A5 requires even query size and nonnegative supervision weights")
    if args.schedule_steps is not None and args.schedule_steps < args.steps:
        p.error("schedule-steps must be at least steps")
    if args.industrial_sampling_protocol in {"episode_flat", "row_flat"}:
        if args.paired_query or args.prior == "A5":
            p.error("episode_flat is incompatible with paired query supervision")
        if args.information_track != "full_state" or args.history_ratio != 0:
            p.error("episode_flat currently requires full_state and history_ratio=0")
    if args.steps < 1 or args.targets_per_task < 1 or args.eval_interval < 1 or not 0 <= args.replay_ratio < 1:
        p.error("Invalid steps/targets/evaluation/replay ratio")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "args.json", vars(args))
    # A compact hash manifest is enough for normal experiment tracking.  The
    # repositories remain the source of truth, so do not duplicate all Python
    # files into every checkpoint directory.
    from industrial_adapter import TABICL_ROOT
    source_paths = list(Path(__file__).parent.glob("*industrial*.py")) + [
        ICDS_ROOT / "icds/export.py",
        ICDS_ROOT / "icds/registry.py",
        ICDS_ROOT / "icds/industrial_templates.py",
        ICDS_ROOT / "icds/history.py",
        TABICL_ROOT / "experiments/exp5_CAQ/train_large6_counterfactual.py",
        TABICL_ROOT / "src/tabicl/_model/caq_model.py",
        TABICL_ROOT / "src/tabicl/_model/tabicl.py",
    ]
    write_json(out / "source-hashes.json", {str(p): checksum(p) for p in source_paths})
    versions = {}
    for package in ["torch", "numpy", "scipy", "PyYAML", "scikit-learn"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    write_json(out / "dependency-versions.json", versions)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; no automatic CPU fallback")
    if device.type == "cuda":
        # 关键：把进程默认 CUDA 设备设为训练设备。TabICL 的 InferenceManager
        # （eval 模式下 col_embedder/row_interactor 走它）在 device=None 时
        # 用 torch.device("cuda") = 进程默认设备做显存估计与分块输出分配。
        # 若默认设备是 cuda:0 而模型在 cuda:1，GPU 0 显存波动触发分块路径时
        # 输出会落在 cuda:0，与 cuda:1 的模型参数混算报 device mismatch
        # （2026-09-22 两进程在 step 10000 评估同时崩溃的根因）。
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device)
    validation = validate(args.manifest, level="full")
    write_json(out / "dataset-validation.json", validation)
    train = IndustrialTaskSampler(args.manifest, "meta_train", args.context_size,
                                  args.query_size, args.targets_per_task, args.seed,
                                  families={"A1": ["physical_ode"], "A2": ["discrete"]}.get(args.prior), target_floor_profile=args.target_floor_profile, external_context_mode=args.external_context_mode, information_track=args.information_track, history_length=args.history_length, history_ratio=args.history_ratio, sensor_noise_multiplier=args.sensor_noise_multiplier, history_families=args.history_families, sampling_protocol=args.industrial_sampling_protocol)
    val = IndustrialTaskSampler(args.manifest, "meta_val", args.context_size,
                                args.query_size, args.targets_per_task, args.seed,
                                target_floor_profile=args.target_floor_profile, external_context_mode=args.external_context_mode, information_track=args.information_track, history_length=args.history_length, history_ratio=args.history_ratio, sensor_noise_multiplier=args.sensor_noise_multiplier, history_families=args.history_families)
    # One fixed probe for each represented prototype.
    probe_worlds = []
    for scenario in sorted(set(train.scenarios.values())):
        candidates = [w for w in train.worlds if train.scenarios[w] == scenario]
        if candidates:
            probe_worlds.append(candidates[0])
    probe_tasks = train.fixed_tasks(probe_worlds)
    val_tasks = val.fixed_tasks()
    row_tasks = val.fixed_row_tasks() if args.industrial_sampling_protocol == "row_flat" else []
    paired_tasks = [] if args.skip_paired_eval else val.fixed_tasks(paired=True)
    scm = SCMReplaySampler(batch_size=1, min_features=5, max_features=args.scm_max_features,
                           min_seq_len=args.context_size + args.query_size,
                           max_seq_len=args.context_size + args.query_size + 1, n_jobs=1)
    model = build_model(args.checkpoint, device, args.model_kind)
    model.target_selector = args.target_selector
    model.target_anchor = args.target_anchor
    model.absolute_state = args.absolute_state
    model.input_scale_profile = args.input_scale_profile
    model.persistence_residual = args.persistence_residual
    model.persistence_residual_scope = args.persistence_residual_scope
    resume_state = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if resume_state is not None:
        model.load_state_dict(resume_state["model"], strict=True)
    params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    initial_params = {name: p.detach().cpu().clone() for name, p in params.items()}
    optimizer = torch.optim.AdamW(params.values(), lr=args.lr, weight_decay=1e-4)
    scheduler = None
    if args.lr_schedule == "cosine":
        # 仿 train_large6_counterfactual.py 的 CosineAnnealingLR：整段退火到
        # lr*0.05，后期小步长稳定权重，直接针对 seed42 固定 1e-4 后期
        # 验证指标大幅摆动（23k best 0.2169 → 30k 0.5199）的问题。
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.schedule_steps or args.steps, eta_min=args.lr * 0.05)
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        if scheduler is not None and resume_state.get("scheduler") is not None:
            scheduler.load_state_dict(resume_state["scheduler"])
        else:
            # 无调度器状态（固定 lr 或旧版 checkpoint）：统一用命令行 lr
            for group in optimizer.param_groups:
                group["lr"] = args.lr
    parameter_counts = {"trainable": sum(p.numel() for p in params.values()),
                        "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad)}
    before = {"train_probe": evaluate(model, probe_tasks, device, train),
              "validation": evaluate(model, val_tasks, device, val),
              "paired_validation": evaluate(model, paired_tasks, device, val, paired=True) if paired_tasks else None,
              "scm_mse": evaluate_scm(model, scm, device)}
    write_json(out / "before.json", before)
    row_best = float("inf")
    if row_tasks:
        before["row_validation"] = evaluate(model, row_tasks, device, val)
        row_best = before["row_validation"]["macro_fixed_scale_mse"]
        save_checkpoint(out / "best-row.pt", model, optimizer, scheduler, args, 0)
        write_json(out / "before.json", before)
    print(json.dumps({"event": "initialized", "parameters": parameter_counts,
                      "train_probe_mse": before["train_probe"]["macro_support_standardized_mse"],
                      "val_support_standardized_mse": before["validation"]["macro_support_standardized_mse"],
                      "val_fixed_scale_mse": before["validation"]["macro_fixed_scale_mse"],
                      "scm_mse": before["scm_mse"]}), flush=True)
    best = before["validation"]["macro_fixed_scale_mse"]
    best_step = 0
    save_checkpoint(out / "best.pt", model, optimizer, scheduler, args, 0)
    history, evals = [], []
    industrial_step = replay_steps = 0
    gradient_groups = {}
    skipped_steps = rolled_back = 0
    consecutive_bad_steps = successful_updates = 0
    task_sequence = hashlib.sha256()
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        replay = int(step * args.replay_ratio + 1e-9) > int((step-1) * args.replay_ratio + 1e-9)
        t0 = time.perf_counter()
        if replay:
            with isolated_seed(stable_seed(args.seed, "scm_update", step) % (2**32)):
                X, y, n = scm_batch(scm, device)
            pred = model(X, y_train=y[:, :n]).mean(dim=-1)
            loss = torch.mean((pred - y[:, n:])**2)
            kind, wid, target_rows = "scm", None, y[:, n:].numel()
            replay_steps += 1
        else:
            task = train.sample(industrial_step, paired=args.paired_query or args.prior == "A5")
            X, A, support, query = to_target_batch(task, device, target_selector=args.target_selector, target_anchor=args.target_anchor, absolute_state=args.absolute_state, persistence_residual=args.persistence_residual, persistence_residual_scope=args.persistence_residual_scope, input_scale_profile=args.input_scale_profile)
            pred = model(X, X_action=A, y_train=support).mean(dim=-1)
            loss = torch.mean((pred-query)**2) if args.loss == "mse" else torch.nn.functional.huber_loss(pred, query)
            effect_loss = zero_loss = torch.zeros((), device=device)
            if args.prior == "A5":
                effect_loss, zero_loss = paired_supervision(pred, query, task, train, args.effect_loss, args.effect_scope)
                loss = loss + args.effect_weight * effect_loss + args.zero_weight * zero_loss
            kind, wid, target_rows = "industrial", int(task["group_ids"][0, 0]), query.numel()
            industrial_step += 1
            task_sequence.update(np.asarray(task["group_ids"], dtype=np.int64).tobytes())
            task_sequence.update(np.asarray(task["target_indices"], dtype=np.int64).tobytes())
        # 病态批次保护（仿 train_large6）：非有限 loss 跳过本步、不更新参数。
        # 偶发病态批次不应杀死整个长训练；连续 10 步才视为参数已污染。
        if not torch.isfinite(loss):
            skipped_steps += 1
            consecutive_bad_steps += 1
            if consecutive_bad_steps >= 10:
                raise FloatingPointError("Nonfinite loss/gradient/parameters for 10 consecutive steps")
            if scheduler is not None:
                scheduler.step()
            print(json.dumps({"event": "skip", "step": step, "kind": kind, "reason": "nonfinite_loss"}), flush=True)
            continue
        # 更新前快照可训练参数，本步更新若产生 NaN 则回滚（仿 train_large6）。
        snapshot = snapshot_trainable(model)
        loss.backward()
        grads_finite = True
        for name, param in params.items():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    grads_finite = False
                    break
                group_name = name.removeprefix("native.")
                group = "col" if group_name.startswith("col_embedder") else ("caq" if "caq" in group_name or "causal" in group_name or "action_encoder" in group_name else "row")
                gradient_groups[group] = max(gradient_groups.get(group, 0), float(param.grad.detach().norm()))
        if not grads_finite:
            skipped_steps += 1
            consecutive_bad_steps += 1
            if consecutive_bad_steps >= 10:
                raise FloatingPointError("Nonfinite loss/gradient/parameters for 10 consecutive steps")
            if scheduler is not None:
                scheduler.step()
            print(json.dumps({"event": "skip", "step": step, "kind": kind, "reason": "nonfinite_gradient"}), flush=True)
            continue
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params.values(), 1.0))
        optimizer.step()
        if not params_finite(model):
            restore_trainable(model, snapshot)
            rolled_back += 1
            # A bad Adam moment must not poison the next update after rollback.
            optimizer.state.clear()
            consecutive_bad_steps += 1
            if consecutive_bad_steps >= 10:
                raise FloatingPointError("Nonfinite loss/gradient/parameters for 10 consecutive steps")
            if scheduler is not None:
                scheduler.step()
            print(json.dumps({"event": "rollback", "step": step, "kind": kind}), flush=True)
            continue
        consecutive_bad_steps = 0
        successful_updates += 1
        if scheduler is not None:
            scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row = {"step": step, "kind": kind, "world_id": wid, "loss": float(loss.detach()),
               "gradient_norm_before_clip": grad_norm, "target_rows": target_rows,
               "update_seconds": time.perf_counter()-t0,
               "targets": task["target_indices"].tolist() if kind == "industrial" else None,
               "query_label_abs_max": float(query.abs().max()) if kind == "industrial" else None,
               "effect_loss": float(effect_loss.detach()) if kind == "industrial" else None,
               "zero_loss": float(zero_loss.detach()) if kind == "industrial" else None,
               "family": train.shards[wid]["family"] if kind == "industrial" else None,
               "scenario": train.scenarios[wid] if kind == "industrial" else None,
               "information_track": task.get("information_track", "full_state") if kind == "industrial" else None,
               "scaling_version": task["scaling_version"] if kind == "industrial" else "original_scm",
               "lr": float(optimizer.param_groups[0]["lr"])}
        history.append(row)
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if step % 10 == 0:
            print(json.dumps(row), flush=True)
        if step % args.eval_interval == 0 or step == args.steps:
            # 评估走 InferenceManager（eval 模式），其批大小估计基于默认设备的
            # 可用显存；先释放训练缓存让估计更宽松、减少分块、加速评估。
            if device.type == "cuda":
                torch.cuda.empty_cache()
            result = {"step": step, "train_probe": evaluate(model, probe_tasks, device, train),
                      "validation": evaluate(model, val_tasks, device, val),
                      "scm_mse": evaluate_scm(model, scm, device)}
            evals.append(result)
            if row_tasks:
                result["row_validation"] = evaluate(model, row_tasks, device, val)
                row_score = result["row_validation"]["macro_fixed_scale_mse"]
                if row_score < row_best:
                    row_best = row_score
                    save_checkpoint(out / "best-row.pt", model, optimizer, scheduler, args, step)
                print(json.dumps({"event": "row_evaluation", "step": step,
                                  "row_fixed_scale_mse": row_score}), flush=True)
            score = result["validation"]["macro_fixed_scale_mse"]
            # Keep a recent optimizer/model state even when validation does not
            # improve.  Long paper runs can then be audited or resumed without
            # falling back to a much older best-validation checkpoint.
            if args.save_latest:
                save_checkpoint(out / "latest.pt", model, optimizer, scheduler, args, step)
            if score < best:
                best = score
                best_step = step
                save_checkpoint(out / "best.pt", model, optimizer, scheduler, args, step)
            write_json(out / "evaluations.json", evals)
            print(json.dumps({"event": "evaluation", "step": step, "val_fixed_scale_mse": score,
                              "train_probe_mse": result["train_probe"]["macro_support_standardized_mse"],
                              "scm_mse": result["scm_mse"]}), flush=True)
    save_checkpoint(out / "final.pt", model, optimizer, scheduler, args, args.steps)
    after = {"train_probe": evaluate(model, probe_tasks, device, train),
             "validation": evaluate(model, val_tasks, device, val),
             "paired_validation": evaluate(model, paired_tasks, device, val, paired=True) if paired_tasks else None,
             "scm_mse": evaluate_scm(model, scm, device)}
    write_json(out / "after.json", after)
    if row_tasks:
        after["row_validation"] = evaluate(model, row_tasks, device, val)
        write_json(out / "after.json", after)
    changes = {name: float((p.detach().cpu()-initial_params[name]).norm()) for name, p in params.items()}
    first = [r["loss"] for r in history if r["kind"] == "industrial"][:20]
    last = [r["loss"] for r in history if r["kind"] == "industrial"][-20:]
    summary = {"status": "completed", "steps": args.steps, "industrial_updates": industrial_step,
               "successful_updates": successful_updates,
               "industrial_task_sequence_sha256": task_sequence.hexdigest(),
               "scm_updates": replay_steps, "parameter_counts": parameter_counts,
               "parameters_changed": sum(v > 0 for v in changes.values()),
               "max_parameter_change_l2": max(changes.values()), "max_gradient_norm_by_group": gradient_groups,
               "lr_schedule": args.lr_schedule, "eta_min": args.lr * 0.05 if args.lr_schedule == "cosine" else None,
               "skipped_steps": skipped_steps, "rolled_back": rolled_back,
               "first20_industrial_loss_mean": float(np.mean(first)),
               "last20_industrial_loss_mean": float(np.mean(last)),
               "fixed_train_probe_mse_before": before["train_probe"]["macro_support_standardized_mse"],
               "fixed_train_probe_mse_after": after["train_probe"]["macro_support_standardized_mse"],
               "validation_mse_before": before["validation"]["macro_support_standardized_mse"],
               "validation_mse_after": after["validation"]["macro_support_standardized_mse"],
               "best_validation_fixed_scale_mse": best,
               "best_step": best_step,
               "scenario_validation_fixed_scale_mse_after": after["validation"]["scenario_macro_fixed_scale_mse"],
               "validation_fixed_scale_mse_before": before["validation"]["macro_fixed_scale_mse"],
               "validation_fixed_scale_mse_after": after["validation"]["macro_fixed_scale_mse"], "scm_mse_before": before["scm_mse"], "scm_mse_after": after["scm_mse"],
               "elapsed_seconds": time.perf_counter()-started,
               "wall_elapsed_seconds": time.perf_counter()-wall_started,
               "target_rows": sum(r["target_rows"] for r in history),
               "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
               "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
               "base_checkpoint_sha256": checksum(args.checkpoint), "manifest_sha256": checksum(args.manifest),
               "device": str(device), "torch": torch.__version__,
               "cuda_memory_fraction": args.cuda_memory_fraction,
               "input_protocol": "typed_absolute_state_v3" if args.absolute_state else ("typed_target_anchor_v2" if args.target_anchor else ("typed_target_selector_v1" if args.target_selector else "typed_y_condition_only")),
               "input_scale_profile": args.input_scale_profile,
               "resume_checkpoint": args.resume,
               "resume_checkpoint_sha256": checksum(args.resume) if args.resume else None,
               "prior": args.prior, "model_kind": args.model_kind,
               "industrial_sampling_protocol": args.industrial_sampling_protocol,
               "information_track": args.information_track, "history_ratio": args.history_ratio, "history_length": args.history_length, "sensor_noise_multiplier": args.sensor_noise_multiplier, "persistence_residual": args.persistence_residual,
               "scope": "Matched-budget industrial prior experiment; transfer conclusions require frozen external evaluation"}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
