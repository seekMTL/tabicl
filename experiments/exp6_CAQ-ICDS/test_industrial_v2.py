"""Locked-checkpoint, three-model held-out-world forecasts for industrial-v2.

No selection or optimizer uses meta_test labels. The official sklearn pipeline
receives identical public typed inputs, including the new absolute-unit encoding.
"""
import argparse
import gc
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from industrial_adapter import to_target_batch
from industrial_common import evaluate
from industrial_task_sampler import IndustrialTaskSampler
from compare_three import OfficialTabICLRegressorAdapter, load
from icds.storage import checksum, write_json
from icds.validate import validate

EXP = Path(__file__).resolve().parent
ROOT = EXP / "checkpoints/industrial-v2"
MANIFEST = "/home/lizitao/project/Industrial_synthetic_data/data/industrial-v2/manifest.json"
MODELS = ("caq", "tabicl_ft", "tabicl_official_zero_shot")


def audit_inputs(args):
    validate(args.manifest, level="schema")
    saved = {}
    for name, path in (("caq", args.caq_checkpoint), ("tabicl_ft", args.ft_checkpoint)):
        state = torch.load(path, map_location="cpu", weights_only=True)
        cfg = state["args"]
        summary = json.loads((Path(path).parent / "summary.json").read_text())
        assert summary["status"] == "completed"
        assert state["step"] == summary["best_step"], "Use locked validation-best checkpoints"
        assert summary["manifest_sha256"] == checksum(args.manifest)
        assert cfg["input_scale_profile"] == "industrial_v2"
        assert cfg["context_size"] == args.context_size and cfg["query_size"] == args.query_size
        assert cfg["history_ratio"] == 0 and cfg["information_track"] == "full_state"
        assert cfg["industrial_sampling_protocol"] == "factual"
        assert all(cfg[k] for k in ("target_selector", "target_anchor", "absolute_state"))
        assert cfg["persistence_residual"] and cfg["persistence_residual_scope"] == "next_observation"
        assert cfg["target_floor_profile"] == "event_v2" and cfg["external_context_mode"] == "observed"
        assert summary["base_checkpoint_sha256"] == checksum(state["base_checkpoint"])
        saved[name] = {
            "path": str(Path(path).resolve()), "sha256": checksum(path),
            "selected_step": state["step"], "base_checkpoint": state["base_checkpoint"],
            "base_checkpoint_sha256": summary["base_checkpoint_sha256"],
            "validation_best_mse": summary["best_validation_fixed_scale_mse"],
            "args": cfg,
        }
        del state
    assert saved["caq"]["base_checkpoint_sha256"] == saved["tabicl_ft"]["base_checkpoint_sha256"]
    settings = [{k: v for k, v in saved[name]["args"].items() if k not in ("out", "model_kind")}
                for name in ("caq", "tabicl_ft")]
    assert settings[0] == settings[1]
    sampler = IndustrialTaskSampler(args.manifest, "meta_test", args.context_size, args.query_size,
                                    seed=42, target_floor_profile="event_v2", external_context_mode="observed",
                                    information_track="full_state", history_ratio=0.)
    assert not (set(sampler.worlds) & (set(sampler.splits["meta_train"]) | set(sampler.splits["meta_val"])))
    tasks = sampler.fixed_tasks()
    fingerprint = hashlib.sha256()
    for wid, task in tasks:
        n = task["context_size"]
        ids = task["group_ids"]
        assert not (set(ids[:n, 1]) & set(ids[n:, 1]))
        assert not (set(ids[:n, 2]) & set(ids[n:, 2]))
        assert len(np.unique(task["selected_rows"])) == n + args.query_size
        assert np.isfinite(task["Y"]).all()
        fingerprint.update(str(wid).encode())
        fingerprint.update(np.asarray(ids, dtype=np.int64).tobytes())
        fingerprint.update(np.asarray(task["target_indices"], dtype=np.int64).tobytes())
    assert len(tasks) == 52 and len(set(sampler.scenarios.values())) == 8
    return saved, sampler, tasks, fingerprint.hexdigest()


def official_model(base, device, estimators):
    model = OfficialTabICLRegressorAdapter(base, device, estimators)
    # Crucial: the legacy adapter otherwise silently falls back to old unit scales.
    model.input_scale_profile = "industrial_v2"
    return model


def result_markdown(report):
    lines = ["# Industrial-v2：锁定best.pt的正式meta_test结果", "",
             "主指标：固定物理参考尺度归一化后，先按目标、再按设备宏平均的MSE；越低越好。", "",
             "context=960 / query=256，52个从未参与训练或最佳模型选择的测试设备；8类原型。", "",
             "| 模型 | 主指标MSE | 原型等权MSE（辅助） | 评估耗时秒 |", "|---|---:|---:|---:|"]
    for name in MODELS:
        rec = report["models"][name]
        pred = rec["prediction"]
        lines.append(f"| {name} | {pred['macro_fixed_scale_mse']:.6g} | "
                     f"{pred['scenario_balanced_fixed_scale_mse']:.6g} | {rec['elapsed_seconds']:.1f} |")
    lines += ["", "| 原型（测试设备数） | CAQ | TabICL-FT | 官方zero-shot |", "|---|---:|---:|---:|"]
    first = report["models"]["caq"]["prediction"]
    for scenario in sorted(first["scenario_macro_fixed_scale_mse"]):
        values = [report["models"][m]["prediction"]["scenario_macro_fixed_scale_mse"][scenario] for m in MODELS]
        lines.append(f"| {scenario} ({first['scenario_world_counts'][scenario]}) | " +
                     " | ".join(f"{v:.6g}" for v in values) + " |")
    worlds = first["worlds"]
    lines += ["", f"参考ridge MSE：{np.mean([w['ridge_fixed_scale_mse'] for w in worlds]):.6g}；"
              f"persistence MSE：{np.mean([w['persistence_fixed_scale_mse'] for w in worlds]):.6g}。", "",
              "官方zero-shot使用8成员官方TabICLRegressor.fit/predict，不进行梯度训练，960条context不是0条。",
              "三模型使用相同公开typed信息与industrial_v2单位编码；官方自带预处理、常量列过滤和集成仍保留。",
              "CAQ动作进入causal通道；FT和官方将同样动作作为普通特征列。",
              "单训练seed、内部未见设备测试，不等于未见原型或真实工厂迁移。",
              "未使用测试集调参、选择checkpoint、修改参考尺度；本轮只补预测指标，不补paired/SCM。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--caq-checkpoint", default=str(ROOT / "caq-A4-seed42-hr0/best.pt"))
    parser.add_argument("--ft-checkpoint", default=str(ROOT / "tabicl-A4-seed42-hr0/best.pt"))
    parser.add_argument("--context-size", type=int, default=960)
    parser.add_argument("--query-size", type=int, default=256)
    parser.add_argument("--official-estimators", type=int, default=8)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--cuda-memory-fraction", type=float, default=.55)
    parser.add_argument("--out", required=True, help="New directory; existing output is never overwritten")
    parser.add_argument("--preflight-only", action="store_true", help="No scoring: check official forward on widest test input")
    args = parser.parse_args()
    if not 0 < args.cuda_memory_fraction <= 1 or args.official_estimators < 1:
        parser.error("Invalid GPU memory budget or estimator count")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "checking", "args": vars(args), "models": {}}
    started = time.monotonic()
    try:
        torch.set_num_threads(4)
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device)
        saved, sampler, tasks, fingerprint = audit_inputs(args)
        report.update(status="running", manifest_sha256=checksum(args.manifest), checkpoints=saved,
                      task_sequence_sha256=fingerprint,
                      scenario_world_counts=dict(Counter(sampler.scenarios.values())),
                      metric="World/target macro MSE using unchanged public physical reference scales",
                      source_hashes={name: checksum(EXP / name) for name in
                                     ("test_industrial_v2.py", "compare_three.py", "industrial_common.py",
                                      "industrial_adapter.py", "industrial_task_sampler.py")})
        write_json(out / "status.json", report)
        print(json.dumps({"status": "input_audit_passed", "worlds": len(tasks), "task_sha256": fingerprint,
                          "prototype_counts": report["scenario_world_counts"],
                          "locked_steps": {k: v["selected_step"] for k, v in saved.items()}}, ensure_ascii=False), flush=True)
        base = saved["caq"]["base_checkpoint"]
        if args.preflight_only:
            wid, task = max(tasks, key=lambda item: item[1]["X_state"].shape[-1] + 2*item[1]["state_dim"])
            model = official_model(base, device, args.official_estimators)
            X, A, support, _ = to_target_batch(task, device, [0], target_selector=True, target_anchor=True,
                                              absolute_state=True, input_scale_profile="industrial_v2")
            with torch.no_grad():
                pred = model(X, X_action=A, y_train=support)
            assert pred.shape == (1, args.query_size, 1) and torch.isfinite(pred).all()
            report.update(status="preflight_passed", checked_world=wid,
                          checked_typed_feature_width=X.shape[-1]+A.shape[-1],
                          elapsed_seconds=time.monotonic()-started)
            write_json(out / "status.json", report)
            print("OFFICIAL_PREFLIGHT_PASSED", wid, X.shape, pred.shape, flush=True)
            return
        for name in MODELS:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            model = (official_model(base, device, args.official_estimators) if name == MODELS[2]
                     else load(Path(saved[name]["path"]), device).eval())
            model_started = time.monotonic()
            print("START_MODEL", name, flush=True)

            def progress(index, total, rec):
                status = {"status": "running", "model": name, "worlds_done": index, "worlds_total": total,
                          "last_world_id": rec["world_id"], "last_scenario": rec["scenario"],
                          "elapsed_seconds": time.monotonic()-started,
                          "completed_models": list(report["models"])}
                write_json(out / "progress.json", status)
                print(json.dumps(status, ensure_ascii=False), flush=True)

            pred = evaluate(model, tasks, device, sampler, progress=progress)
            assert all(np.isfinite(w["fixed_scale_mse"]) for w in pred["worlds"])
            rec = {"prediction": pred, "elapsed_seconds": time.monotonic()-model_started}
            if device.type == "cuda":
                rec.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            report["models"][name] = rec
            write_json(out / f"{name}.json", rec)
            write_json(out / "status.json", report)
            print("MODEL_COMPLETED", name, pred["macro_fixed_scale_mse"], flush=True)
            del model
        report.update(status="completed", elapsed_seconds=time.monotonic()-started)
        write_json(out / "report.json", report)
        (out / "results.md").write_text(result_markdown(report))
        write_json(out / "status.json", report)
        write_json(out / "progress.json", {"status": "completed", "completed_models": list(MODELS)})
        print(result_markdown(report), flush=True)
        print("ALL_THREE_MODELS_COMPLETED", out / "results.md", flush=True)
    except BaseException as error:
        report.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic()-started)
        write_json(out / "status.json", report)
        raise


if __name__ == "__main__":
    main()
