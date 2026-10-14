"""Meta-world split first, episode/source-safe ICL sampling second."""
import json
import numpy as np
from industrial_adapter import make_episode_flat_task, make_task
from icds.storage import load_dataset
from icds.noise import stable_seed
from icds.industrial_templates import scenario_key


class IndustrialTaskSampler:
    def __init__(self, manifest, split="meta_train", context_size=64, query_size=32,
                 targets_per_task=2, seed=42, families=None, target_floor_profile="original", external_context_mode="original", information_track="full_state", history_length=8, history_ratio=0., sensor_noise_multiplier=1., history_families=None, sampling_protocol="factual"):
        self.target_floor_profile = target_floor_profile
        self.external_context_mode = external_context_mode
        self.information_track, self.history_length, self.history_ratio = information_track, history_length, history_ratio
        self.sensor_noise_multiplier = sensor_noise_multiplier
        self.history_families = None if history_families is None else set(history_families)
        self.sampling_protocol = sampling_protocol
        if sampling_protocol not in {"factual", "episode_flat", "row_flat"}:
            raise ValueError("Unknown industrial sampling protocol")
        if self.history_families is not None and not self.history_families <= {'discrete','physical_ode','hybrid_line'}:
            raise ValueError('Unknown history family')
        self.manifest = str(manifest)
        self.root, self.metadata = load_dataset(manifest)
        self.splits = json.loads((self.root / self.metadata["split"]).read_text())
        self.worlds = list(self.splits[split])
        self.context_size, self.query_size = context_size, query_size
        self.targets_per_task, self.seed = targets_per_task, seed
        self.shards = {s["world_id"]: s for s in self.metadata["shards"]}
        if families is not None:
            self.worlds = [w for w in self.worlds if self.shards[w]["family"] in families]
        if not self.worlds:
            raise ValueError("No worlds for requested training prior")
        self.buckets = {}
        self.scenarios = {}
        for wid in self.worlds:
            s = self.shards[wid]
            p = json.loads((self.root / s["program"]).read_text())
            self.scenarios[wid] = scenario_key(p["mechanism"])
            w = json.loads((self.root / s["world"]).read_text())
            key = (p["family"], tuple(p["state_units"]), tuple(sorted(w["parameter_scales"])), len(p["action_names"]))
            self.buckets.setdefault(key, []).append(wid)

    def sample(self, step, paired=False):
        # Permuted round-robin worlds prevents rare family starvation in a short smoke.
        cycle, offset = divmod(step, len(self.worlds))
        rng = np.random.default_rng(stable_seed(self.seed, "train_worlds", cycle))
        wid = int(rng.permutation(self.worlds)[offset])
        use_history = self.information_track == "observed_history" or (self.history_ratio > 0 and np.random.default_rng(stable_seed(self.seed,"history_track",step)).random() < self.history_ratio)
        if self.history_families is not None:
            use_history = use_history and self.shards[wid]['family'] in self.history_families
        if use_history:
            from icds.history import make_history_task, observer_spec
            program=json.loads((self.root/self.shards[wid]["program"]).read_text())
            d=len(observer_spec(program)["names"])
        else:
            d = self.shards[wid]["shapes"]["state"][1]
        rng = np.random.default_rng(stable_seed(self.seed, "targets", step))
        targets = rng.choice(d, min(d, self.targets_per_task), replace=False)
        if self.sampling_protocol in {"episode_flat", "row_flat"}:
            if paired or use_history:
                raise ValueError("Episode-flat sampling does not support paired/history tasks")
            factory = make_episode_flat_task
        else:
            factory = make_history_task if use_history else make_task
        kwargs = {"history_length": self.history_length, "noise_multiplier": self.sensor_noise_multiplier} if use_history else {}
        if self.sampling_protocol in {"episode_flat", "row_flat"}:
            task = factory(self.manifest, wid, self.context_size, self.query_size,
                           seed=stable_seed(self.seed, "task", step),
                           target_indices=targets,
                           row_split_seed=stable_seed(self.seed, "row_split", wid) if self.sampling_protocol == "row_flat" else None)
        else:
            task = factory(self.manifest, wid, self.context_size, self.query_size,
                           seed=stable_seed(self.seed, "task", step), target_indices=targets,
                           query_mode="paired" if paired else "factual", **kwargs)
        return self.prepare(task, wid)

    def prepare(self, task, wid):
        if self.external_context_mode != "original":
            if self.external_context_mode not in {"observed", "hidden"}:
                raise ValueError("Unknown external visibility")
            visible = self.external_context_mode == "observed"
            if not visible:
                task["X_state"][0, :, task.get("exogenous_feature_columns", [task["state_dim"]])] = 0
                task["current_exogenous"] = np.zeros_like(task["current_exogenous"])
            mask = np.full((1, task["X_state"].shape[1], 1), float(visible), dtype=np.float32)
            task["X_state"] = np.concatenate([task["X_state"], mask], axis=-1)
            task["feature_spec"].append("external_context_available")
        if self.target_floor_profile == "original":
            return task
        if self.target_floor_profile != "event_v2":
            raise ValueError("Unknown floor profile")
        program = json.loads((self.root / self.shards[wid]["program"]).read_text())
        scales = task["scaler"]["target_scale"].copy()
        for j, original in enumerate(task["target_indices"]):
            # Public wear range starts at 0..0.1. Maintenance resets are valid.
            # No hidden world parameter or query label sets this fixed floor.
            if program["family"] == "hybrid_line" and task["target_spec"]["state_names"][j].endswith("_wear"):
                scales[j] = max(scales[j], .01)
        task["scaler"]["target_scale"] = scales
        task["Y"] = ((task["raw_y"] - task["scaler"]["target_mean"])/scales).astype(np.float32)[None]
        n = task["context_size"]
        task["y_train"], task["y_query"] = task["Y"][:, :n], task["Y"][:, n:]
        task["scaling_version"] += "+event_floor_v2"
        return task

    def fixed_tasks(self, worlds=None, paired=False, information_track=None):
        from icds.history import make_history_task
        worlds = self.worlds if worlds is None else worlds
        track = information_track or self.information_track
        factory = make_history_task if track == "observed_history" else make_task
        kwargs = {"history_length": self.history_length, "noise_multiplier": self.sensor_noise_multiplier} if track == "observed_history" else {}
        return [(wid, self.prepare(factory(self.manifest, wid, self.context_size, self.query_size,
                               seed=stable_seed(self.seed, "fixed_eval", wid),
                               query_mode="paired" if paired else "factual",
                               query_seed=stable_seed(self.seed, "fixed_query", wid), **kwargs), wid)) for wid in worlds]

    def fixed_row_tasks(self):
        return [(wid, self.prepare(make_episode_flat_task(
            self.manifest, wid, self.context_size, self.query_size,
            seed=stable_seed(self.seed, "row_eval", wid),
            row_split_seed=stable_seed(self.seed, "row_split", wid),
            row_split_eval=True), wid)) for wid in self.worlds]
