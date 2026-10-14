"""Evaluate a frozen industrial checkpoint, with episode-safe support/query."""
import argparse
from pathlib import Path
import json
import torch
from industrial_adapter import ICDS_ROOT
from industrial_task_sampler import IndustrialTaskSampler
from industrial_common import evaluate, evaluate_scm
from train_industrial import build_model
from train_large6_counterfactual import SCMReplaySampler
from icds.storage import write_json
from icds.validate import validate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True, help="best.pt or final.pt from train_industrial")
    p.add_argument("--split", choices=["meta_train", "meta_val", "meta_test"], default="meta_test")
    p.add_argument("--context-sizes", type=int, nargs="+", default=[32, 128])
    p.add_argument("--query-size", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--information-track", choices=["full_state", "observed_history"],
                   help="Evaluate the declared public information view; default is checkpoint track")
    args = p.parse_args()
    if Path(args.out).exists():
        raise FileExistsError("Evaluation output already exists; choose a new path")
    torch.set_num_threads(4)
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_device(torch.device(args.device))
    validate(args.manifest, level="schema")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["base_checkpoint"], args.device, checkpoint["args"].get("model_kind", "caq"))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.target_selector = checkpoint["args"].get("target_selector", False)
    model.target_anchor = checkpoint["args"].get("target_anchor", False)
    model.absolute_state = checkpoint["args"].get("absolute_state", False)
    model.input_scale_profile = checkpoint["args"].get("input_scale_profile", "legacy")
    model.persistence_residual = checkpoint["args"].get("persistence_residual", False)
    model.persistence_residual_scope = checkpoint["args"].get("persistence_residual_scope", "all")
    results = {}
    for context in args.context_sizes:
        sampler = IndustrialTaskSampler(args.manifest, args.split, context, args.query_size, seed=args.seed, target_floor_profile=checkpoint["args"].get("target_floor_profile", "original"), external_context_mode=checkpoint["args"].get("external_context_mode", "original"), information_track=checkpoint["args"].get("information_track", "full_state"), history_length=checkpoint["args"].get("history_length", 8), sensor_noise_multiplier=checkpoint["args"].get("sensor_noise_multiplier",1.))
        tasks = sampler.fixed_tasks(information_track=args.information_track)
        pairs = sampler.fixed_tasks(paired=True, information_track=args.information_track)
        results[str(context)] = {"prediction": evaluate(model, tasks, args.device, sampler),
                                 "paired_effect": evaluate(model, pairs, args.device, sampler, paired=True)}
    scm = SCMReplaySampler(batch_size=1, min_features=5, max_features=12,
                           min_seq_len=96, max_seq_len=97, n_jobs=1)
    report = {"checkpoint": str(Path(args.checkpoint).resolve()), "selected_step": checkpoint["step"],
              "split": args.split, "target_selector": model.target_selector, "few_shot": results, "scm_mse": evaluate_scm(model, scm, args.device),
              "information_track": args.information_track or checkpoint["args"].get("information_track", "full_state"),
              "limitations": ["This report covers internal one-step forecasts and paired simulator labels; recursive/external results have separate reports",
                               "One checkpoint; confidence intervals require multiple training seeds and world-level aggregation",
                               "Partial-observation or hybrid fork effects are descriptive simulator labels, not identified available-input causal means"]}
    write_json(args.out, report)
    print(json.dumps({"out": args.out, "selected_step": checkpoint["step"],
                      "family_mse": {k: v["prediction"]["family_macro_fixed_scale_mse"] for k, v in results.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
