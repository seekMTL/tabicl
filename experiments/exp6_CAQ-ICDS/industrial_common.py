"""Shared industrial evaluation: physical units, grouped metrics, paired effects."""
from contextlib import contextmanager
import json
import random
import numpy as np
import torch
from industrial_adapter import predict_task


@contextmanager
def isolated_seed(seed):
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    gpu = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.random.set_rng_state(cpu)
        if gpu is not None:
            torch.cuda.set_rng_state_all(gpu)


@torch.no_grad()
def evaluate(model, tasks, device, sampler, paired=False, progress=None):
    was_training = model.training
    model.eval()
    records = []
    for wid, task in tasks:
        pred_scaled = predict_task(model, task, device).cpu().numpy()
        n = task["context_size"]
        scale, mean = task["scaler"]["target_scale"], task["scaler"]["target_mean"]
        predicted = pred_scaled * scale + mean
        actual = task["raw_y"][n:]
        p = json.loads((sampler.root / sampler.shards[wid]["program"]).read_text())
        errors = predicted - actual
        baseline = task["current_target_state"][n:].copy() if task.get("target_mode") == "next_observation" else np.zeros_like(actual)
        for i, kind in enumerate(task["target_spec"]["kinds"]):
            if kind in {"count", "binary"}:
                baseline[:, i] = task["current_target_state"][n:, i]
        persistence_error = actual-baseline
        reference_scales = np.array(task["target_spec"]["reference_scales"])
        # Context ridge sees exactly the same features/actions/support as CAQ.
        X = np.c_[task["X_state"][0], task["X_action"][0], np.ones(n + len(actual))]
        ridge_coef = np.linalg.solve(X[:n].T @ X[:n] + 1e-5 * np.eye(X.shape[1]),
                                     X[:n].T @ task["raw_y"][:n])
        ridge_pred = X[n:] @ ridge_coef
        rec = {"world_id": wid, "family": p["family"], "scenario": sampler.scenarios[wid],
               "mechanism": p["mechanism"], "target_names": task["target_spec"]["state_names"],
               "reference_scales": reference_scales.tolist(), "target_units": task["target_spec"]["units"],
               "information_track": task.get("information_track", "full_state"), "target_mode": task.get("target_mode", "typed_full_state"),
               "per_target_rmse": np.sqrt(np.mean(errors**2, axis=0)).tolist(),
               "per_target_mae": np.mean(np.abs(errors), axis=0).tolist(),
               "persistence_per_target_rmse": np.sqrt(np.mean(persistence_error**2, axis=0)).tolist(),
               "ridge_per_target_rmse": np.sqrt(np.mean((ridge_pred-actual)**2, axis=0)).tolist(),
               "support_standardized_mse": float(np.mean((errors/scale)**2)),
               "persistence_support_standardized_mse": float(np.mean((persistence_error/scale)**2)),
               "per_target_nrmse": (np.sqrt(np.mean(errors**2, axis=0))/reference_scales).tolist(),
               "fixed_scale_mse": float(np.mean((errors/reference_scales)**2)),
               "persistence_fixed_scale_mse": float(np.mean((persistence_error/reference_scales)**2)),
               "ridge_fixed_scale_mse": float(np.mean(((ridge_pred-actual)/reference_scales)**2)),
               "ridge_support_standardized_mse": float(np.mean(((ridge_pred-actual)/scale)**2))}
        if paired:
            effect_pred = predicted[1::2] - predicted[0::2]
            effect_true = actual[1::2] - actual[0::2]
            query_ids = task["group_ids"][n:]
            if not np.array_equal(query_ids[0::2, 2], query_ids[1::2, 2]):
                raise ValueError("Paired query source IDs differ")
            actions = task["X_action"][0, n:]
            changed = np.abs(actions[1::2] - actions[0::2]) > 1e-6
            graph = np.asarray(task["step_action_graph"]) if "step_action_graph" in task else np.asarray(p["step_action"])[task["target_indices"]]
            graph = graph[:, :changed.shape[1]]
            changed = changed[:, :graph.shape[1]]
            zeros = ~(changed.astype(int) @ graph.T.astype(int) > 0)
            if p.get("graph_precision", "exact") != "exact":
                zeros[:] = False
            nonzero = np.abs(effect_true/reference_scales) > 1e-5
            false_positive = (np.abs(effect_pred/reference_scales) > 1e-5) & zeros
            rec.update(effect_per_target_rmse=np.sqrt(np.mean((effect_pred-effect_true)**2, axis=0)).tolist(),
                       effect_label_semantics=("difference of independent transition draws; not individual counterfactual" if task.get("noise_coupling_id") == "independent_post_source" else "shared-noise simulator individual effect; not conditional causal mean" if p["family"] == "hybrid_line" or task.get("information_track") == "observed_history" else "deterministic/shared-additive-innovation action effect"),
                       effect_support_standardized_mse=float(np.mean(((effect_pred-effect_true)/scale)**2)),
                       effect_fixed_reference_mse=float(np.mean(((effect_pred-effect_true)/reference_scales)**2)),
                       effect_per_target_nrmse=(np.sqrt(np.mean((effect_pred-effect_true)**2, axis=0))/reference_scales).tolist(),
                       proven_zero_false_positives=int(false_positive.sum()),
                       proven_zero_false_positive_rate=float(false_positive.sum()/zeros.sum()) if zeros.any() else None,
                       effect_nonzero_threshold_reference_normalized=1e-5,
                       true_nonzero_effects=int(nonzero.sum()), proven_zero_effects=int(zeros.sum()),
                       nonzero_effect_detection_rate=float(np.mean(np.abs(effect_pred[nonzero]/np.broadcast_to(reference_scales,effect_pred.shape)[nonzero]) > 1e-5)) if nonzero.any() else None,
                       nonzero_effect_sign_accuracy=float(np.mean(np.sign(effect_pred[nonzero]) == np.sign(effect_true[nonzero]))) if nonzero.any() else None,
                       proven_zero_predicted_abs_mean=float(np.mean(np.abs(effect_pred[zeros]))) if zeros.any() else None,
                       proven_zero_predicted_abs_max=float(np.max(np.abs(effect_pred[zeros]))) if zeros.any() else None)
            magnitude = np.linalg.norm(actions[1::2]-actions[0::2], axis=1)
            bins = [(0, .5), (.5, 1), (1, 2), (2, float("inf"))]
            rec["action_amplitude_bins"] = []
            for low, high in bins:
                selected = (magnitude >= low) & (magnitude < high)
                if selected.any():
                    rec["action_amplitude_bins"].append({"low": low, "high": high if np.isfinite(high) else None,
                        "sources": int(selected.sum()), "effect_fixed_reference_mse": float(np.mean(((effect_pred[selected]-effect_true[selected])/reference_scales)**2))})
        records.append(rec)
        if progress is not None:
            progress(len(records), len(tasks), rec)
    model.train(was_training)
    families = {}
    scenarios = {}
    effect_families = {}
    for r in records:
        families.setdefault(r["family"], []).append(r["fixed_scale_mse"])
        scenarios.setdefault(r["scenario"], []).append(r["fixed_scale_mse"])
        if "effect_fixed_reference_mse" in r:
            effect_families.setdefault(r["family"], []).append(r["effect_fixed_reference_mse"])
    return {"worlds": records,
            "family_macro_effect_fixed_reference_mse": {k: float(np.mean(v)) for k, v in effect_families.items()},
            "macro_fixed_scale_mse": float(np.mean([r["fixed_scale_mse"] for r in records])),
            "family_macro_fixed_scale_mse": {k: float(np.mean(v)) for k, v in families.items()},
            "scenario_macro_fixed_scale_mse": {k: float(np.mean(v)) for k, v in scenarios.items()},
            "scenario_world_counts": {k: len(v) for k, v in scenarios.items()},
            "scenario_balanced_fixed_scale_mse": float(np.mean([np.mean(v) for v in scenarios.values()])),
            "macro_support_standardized_mse": float(np.mean([r["support_standardized_mse"] for r in records])),
            "macro_persistence_support_standardized_mse": float(np.mean([r["persistence_support_standardized_mse"] for r in records])),
            "macro_ridge_support_standardized_mse": float(np.mean([r["ridge_support_standardized_mse"] for r in records])),
            "metric_note": "World/target macro average; support standardization is task-specific, not fixed-scale NRMSE"}


def scm_batch(sampler, device):
    for _ in range(3):
        X, y, d, n = sampler.get_batch()
        if torch.isfinite(X).all() and torch.isfinite(y).all() and 0 < n < y.shape[1]:
            return X.to(device), y.to(device), n
    raise FloatingPointError("SCM replay nonfinite after 3 attempts")


@torch.no_grad()
def evaluate_scm(model, sampler, device):
    mode = model.training
    model.eval()
    with isolated_seed(71023):
        X, y, n = scm_batch(sampler, device)
        pred = model(X, y_train=y[:, :n]).mean(dim=-1)
        mse = float(torch.mean((pred-y[:, n:])**2))
    model.train(mode)
    return mse
