"""ICDS truth -> typed CAQ input, bypassing sklearn constant-column filtering."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

ICDS_ROOT = Path(os.environ.get("ICDS_ROOT", "/home/lizitao/project/Industrial_synthetic_data"))
TABICL_ROOT = Path(os.environ.get("TABICL_ROOT", "/home/lizitao/project/tabicl"))
sys.path.insert(0, str(ICDS_ROOT))
sys.path.insert(0, str(TABICL_ROOT / "src"))
sys.path.append(str(TABICL_ROOT / "experiments" / "exp5_CAQ"))
from icds.export import make_task
from icds.registry import scale_for, SCALING_VERSION, SCALES
from icds.storage import load_dataset, load_shard

ABSOLUTE_INPUT_SCALES = {
    "K": 300., "J": 1e9, "m3": 1., "count": 10., "s": 5.,
    "public_scale": 100., "1": 1., "jobs/s": .3,
    # Fixed engineering input scales, not target evaluation references.
    # The registered priors use <=50 N commands, <=24 V / >=.5 Ohm,
    # and >=.05 Nm/A: 50 A and 500 rad/s cover those drive scales.
    "m": 1., "m/s": 5., "N": 50., "A": 50., "rad/s": 500.,
}


def absolute_input_scales(names, units, profile="legacy"):
    if profile not in {"legacy", "industrial_v2"}:
        raise ValueError("Unknown absolute input scale profile")
    fixed = {"K": 300., "J": 1e9, "m3": 1., "count": 10., "s": 5., "public_scale": 100.}
    if profile == "industrial_v2":
        fixed = ABSOLUTE_INPUT_SCALES
        missing = set(units) - set(fixed)
        if missing:
            raise ValueError(f"Unregistered absolute input units: {sorted(missing)}")
    return np.asarray([.1 if name.endswith("_wear") else fixed.get(unit, 1.)
                       for name, unit in zip(names, units)], dtype=float)


def make_episode_flat_task(manifest_path, world_id, context_size=128,
                           query_size=128, seed=0, action_width=8,
                           target_indices=None, query_seed=None,
                           min_history=1, row_split_seed=None, row_split_eval=False):
    """Build an episode-safe task by sampling rows from every branch.

    Context and query use disjoint episode halves, so the same source state can
    never cross the ICL boundary.  Inside each half, rows are sampled uniformly
    from the stored factual and counterfactual branches without pairing or
    branch balancing.  This is the leakage-safe ICDS analogue of Exp5's flat
    row shuffle.
    """
    if isinstance(min_history, bool) or not isinstance(min_history, int) or min_history < 1:
        raise ValueError("Minimum history must be positive integer")
    root, manifest = load_dataset(manifest_path)
    shard = next(s for s in manifest["shards"] if s["world_id"] == world_id)
    world = json.loads((root / shard["world"]).read_text())
    program = json.loads((root / shard["program"]).read_text())
    arrays = load_shard(root, shard)
    ids = arrays["ids"]
    episodes = np.unique(ids[:, 1])
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    if len(episodes) < 2:
        raise ValueError("Need independent context/query episodes")
    split = len(episodes) // 2
    eligible = ids[:, 4] >= min_history - 1
    context_pool = np.flatnonzero(np.isin(ids[:, 1], episodes[:split]) & eligible)
    query_pool = np.flatnonzero(np.isin(ids[:, 1], episodes[split:]) & eligible)
    if (context_size < 1 or query_size < 1 or
            context_size > len(context_pool) or query_size > len(query_pool)):
        raise ValueError("Requested flat task does not fit; rows will not be duplicated")
    ci = rng.choice(context_pool, context_size, replace=False)
    query_rng = rng if query_seed is None else np.random.default_rng(query_seed)
    qi = query_rng.choice(query_pool, query_size, replace=False)
    selected = np.r_[ci, qi]
    if row_split_seed is not None:
        # Fixed per-world split; training queries also come from the 80% pool.
        order = np.random.default_rng(row_split_seed).permutation(len(ids))
        cut = int(.8 * len(order))
        train_pool, test_pool = order[:cut], order[cut:]
        if row_split_eval:
            ci = rng.choice(train_pool, context_size, replace=False)
            qi = query_rng.choice(test_pool, query_size, replace=False)
            selected = np.r_[ci, qi]
        else:
            selected = rng.choice(train_pool, context_size + query_size, replace=False)

    d, a = arrays["state"].shape[1], arrays["action"].shape[1]
    if a > action_width:
        raise ValueError("Action width smaller than physical layout")
    targets = np.arange(d) if target_indices is None else np.asarray(target_indices, dtype=int)
    if targets.ndim != 1 or len(targets) == 0 or np.any(targets < 0) or np.any(targets >= d):
        raise ValueError("Invalid target selection")

    units = program["state_units"]
    public_scales = program.get("state_scales") or [SCALES[u] for u in units]
    kinds = program.get("state_kinds") or ["continuous"] * d
    external_unit = program.get("exogenous_unit") or (
        "K" if program["family"] == "physical_ode" else "1")
    dynamic = np.concatenate(
        [arrays["state"][selected], arrays["exogenous_observed"][selected]], axis=1
    ).astype(float)
    mean = dynamic[:context_size].mean(axis=0)
    floors = np.array(
        [v["dynamic_floor"] for v in public_scales]
        + [scale_for(external_unit, "dynamic_floor")]
    )
    std = np.maximum(dynamic[:context_size].std(axis=0), floors)
    for i, kind in enumerate(kinds):
        if kind == "binary":
            mean[i], std[i] = 0.0, 1.0
    dynamic = (dynamic - mean) / std

    keys = sorted(world["parameter_scales"])
    static_values, masks = [], []
    for key in keys:
        observed = key in world["observed_parameters"]
        static_values.append(
            world["parameters"][key] / world["parameter_scales"][key]
            if observed else 0.0
        )
        masks.append(float(observed))
    action_mask = np.r_[np.ones(a), np.zeros(action_width - a)]
    static = np.tile(
        static_values + [world["dt"]] + masks + action_mask.tolist(),
        (len(selected), 1),
    )
    X = np.c_[dynamic, static].astype(np.float32)

    actions = arrays["action"][selected]
    low, high = np.asarray(world["action_low"]), np.asarray(world["action_high"])
    A = np.zeros((len(selected), action_width), dtype=np.float32)
    A[:, :a] = (2 * (actions - low) / (high - low) - 1).astype(np.float32)

    raw_y = (arrays["next_state"][selected].astype(float)
             - arrays["state"][selected].astype(float))
    for i, kind in enumerate(kinds):
        if kind in {"count", "binary"}:
            raw_y[:, i] = arrays["next_state"][selected, i]
    raw_y = raw_y[:, targets]
    ym = raw_y[:context_size].mean(axis=0)
    target_floors = np.array([public_scales[i]["target_floor"] for i in targets])
    ys = np.maximum(raw_y[:context_size].std(axis=0), target_floors)
    for i, original in enumerate(targets):
        if kinds[original] == "binary":
            ym[i], ys[i] = 0.0, 1.0
    y = ((raw_y - ym) / ys).astype(np.float32)

    return {
        "X_state": X[None], "X_action": A[None],
        "y_train": y[None, :context_size], "y_query": y[None, context_size:],
        "Y": y[None], "raw_y": raw_y,
        "noise_coupling_id": manifest["config"]["fork"]["noise_coupling"],
        "current_state": arrays["observation"][selected].astype(float),
        "current_exogenous": arrays["exogenous_observed"][selected].astype(float),
        "exogenous_unit": external_unit,
        "state_spec": {"names": program["state_names"], "units": units},
        "current_target_state": np.take(
            arrays["state"][selected], targets, axis=1).astype(float),
        "selected_rows": selected.copy(), "context_size": context_size,
        "state_dim": d, "target_indices": targets,
        "target_spec": {
            "state_names": [program["state_names"][i] for i in targets],
            "kinds": [kinds[i] for i in targets],
            "units": [units[i] for i in targets],
            "reference_scales": [public_scales[i]["target_reference"] for i in targets],
        },
        "scaling_version": SCALING_VERSION + ("+row_flat_v1" if row_split_seed is not None else "+episode_flat_all_branches_v1"),
        "sampling_protocol": "row_flat_v1" if row_split_seed is not None else "episode_flat_all_branches_v1",
        "group_ids": ids[selected].copy(), "action_mask": action_mask,
        "target_mask": np.ones(len(targets)),
        "feature_spec": ([f"state_{i}" for i in range(d)]
                         + ["current_exogenous"] + keys + ["dt"]
                         + [f"available_{k}" for k in keys]
                         + [f"action_available_{i}" for i in range(action_width)]),
        "scaler": {"dynamic_mean": mean, "dynamic_scale": std,
                   "target_mean": ym, "target_scale": ys},
    }


def persistence_target(task):
    """Public-current baseline in the task's physical target semantics."""
    baseline = np.zeros_like(task["raw_y"])
    if task.get("target_mode") == "next_observation":
        baseline[:] = task["current_target_state"]
    else:
        for j, kind in enumerate(task["target_spec"]["kinds"]):
            if kind in {"count", "binary"}:
                baseline[:, j] = task["current_target_state"][:, j]
    return baseline


def residual_baseline_scaled(task, scope="all"):
    if scope not in {"all", "next_observation"}:
        raise ValueError("Unknown residual scope")
    if scope == "next_observation" and task.get("target_mode") != "next_observation":
        return np.zeros_like(task["raw_y"])
    return (persistence_target(task) - task["scaler"]["target_mean"]) / task["scaler"]["target_scale"]


def to_target_batch(task, device, target_columns=None, target_selector=False, target_anchor=False, absolute_state=False, persistence_residual=False, persistence_residual_scope="all", input_scale_profile="legacy"):
    """One ICL table per selected scalar target, with its own y conditioning."""
    cols = np.arange(task["Y"].shape[-1]) if target_columns is None else np.asarray(target_columns)
    X = torch.as_tensor(task["X_state"], device=device)
    A = torch.as_tensor(task["X_action"], device=device)
    Y = torch.as_tensor(np.take(task["Y"][0], cols, axis=1).T.copy(), device=device)
    if persistence_residual:
        baseline = residual_baseline_scaled(task, persistence_residual_scope)
        Y = Y - torch.as_tensor(baseline[:, cols].T.copy(), device=device, dtype=Y.dtype)
    n = task["context_size"]
    if not torch.isfinite(X).all() or not torch.isfinite(A).all() or not torch.isfinite(Y).all():
        raise ValueError("Nonfinite industrial task")
    ids = task["group_ids"]
    if len(np.unique(task["selected_rows"])) != len(task["selected_rows"]):
        raise ValueError("Duplicate context/query rows")
    if task.get("sampling_protocol") != "row_flat_v1" and (set(ids[:n, 1]) & set(ids[n:, 1]) or set(ids[:n, 2]) & set(ids[n:, 2])):
        raise ValueError("Context/query episode or source overlap")
    X = X.repeat(len(cols), 1, 1)
    if target_selector:
        original_targets = torch.as_tensor(task["target_indices"][cols], device=device, dtype=torch.long)
        selector = torch.nn.functional.one_hot(original_targets, num_classes=task["state_dim"]).float()
        X = torch.cat([X, selector[:, None, :].expand(-1, X.shape[1], -1)], dim=-1)
    if target_anchor:
        # Absolute origins matter for reset/nonlinear systems. Support centering
        # otherwise hides them; this is the public current observation/state.
        units = task["target_spec"]["units"]
        names = task["target_spec"].get("state_names", [""]*len(units))
        scales = absolute_input_scales(names, units, input_scale_profile)[cols]
        current = np.take(task["current_target_state"], cols, axis=1).T.copy()
        anchor = torch.as_tensor(current, device=device, dtype=torch.float32) / torch.tensor(scales, device=device, dtype=torch.float32)[:,None]
        X = torch.cat([X, anchor[:,:,None]], dim=-1)
    if absolute_state:
        names, units = task["state_spec"]["names"], task["state_spec"]["units"]
        scales = absolute_input_scales(names, units, input_scale_profile)
        observed = torch.as_tensor(task["current_state"],device=device,dtype=torch.float32)/torch.tensor(scales,device=device,dtype=torch.float32)
        X = torch.cat([X,observed[None].expand(len(cols),-1,-1)],dim=-1)
        if "current_exogenous" in task:
            external_unit = task.get("exogenous_unit", "1")
            scale = (absolute_input_scales(["driver"], [external_unit], input_scale_profile)[0]
                     if input_scale_profile == "industrial_v2" else {"K":300.,"jobs/s":.3}.get(external_unit,1.))
            driver = torch.as_tensor(task["current_exogenous"],device=device,dtype=torch.float32)/scale
            X = torch.cat([X,driver[None].expand(len(cols),-1,-1)],dim=-1)
    return X, A.repeat(len(cols), 1, 1), Y[:, :n], Y[:, n:]


def predict_task(model, task, device, chunk_size=2):
    chunks = []
    for start in range(0, task["Y"].shape[-1], chunk_size):
        cols = np.arange(start, min(start + chunk_size, task["Y"].shape[-1]))
        X, A, support, query = to_target_batch(task, device, cols, target_selector=getattr(model, "target_selector", False), target_anchor=getattr(model, "target_anchor", False), absolute_state=getattr(model, "absolute_state", False), persistence_residual=getattr(model, "persistence_residual", False), persistence_residual_scope=getattr(model, "persistence_residual_scope", "all"), input_scale_profile=getattr(model, "input_scale_profile", "legacy"))
        chunks.append(model(X, X_action=A, y_train=support).mean(dim=-1).T)
    result = torch.cat(chunks, dim=1)
    if getattr(model, "persistence_residual", False):
        n = task["context_size"]
        baseline = residual_baseline_scaled(task, getattr(model, "persistence_residual_scope", "all"))[n:]
        result = result + torch.as_tensor(baseline, device=device, dtype=result.dtype)
    return result
