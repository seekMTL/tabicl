"""Industrial comparison on the same one-step evaluation protocol.

The trained CAQ, fine-tuned TabICL, frozen TabICL core, and (optionally) the
official TabICLRegressor zero-shot pipeline are evaluated on identical fixed
tasks (prediction + paired action effects).

frozen TabICL keeps the same public input columns (target selector/anchor/
absolute state) but persistence_residual=False: it was never trained to predict
the residual target, so the baseline must not be added back at eval time.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from industrial_task_sampler import IndustrialTaskSampler
from industrial_common import evaluate, evaluate_scm
from train_industrial import build_model, DEFAULT_CHECKPOINT
from train_large6_counterfactual import SCMReplaySampler
from icds.storage import write_json
from tabicl import TabICLRegressor

MANIFEST = "/home/lizitao/project/Industrial_synthetic_data/data/pilot-m3/manifest.json"
EXP = Path(__file__).resolve().parent
CAQ_CKPT = EXP / "checkpoints/paper-main/caq-A4-seed42-hr0/best.pt"
TABICL_FT_CKPT = EXP / "checkpoints/paper-main/tabicl-A4-seed42-hr0/best.pt"


class OfficialTabICLRegressorAdapter:
    """Expose the official sklearn zero-shot pipeline through the model API.

    Each batch element is one scalar target. The official estimator performs
    its own target scaling, preprocessing, feature shuffling and ensembling;
    no neural-network parameters are updated by ``fit``.
    """

    def __init__(self, checkpoint, device, n_estimators=8):
        self.checkpoint = str(checkpoint)
        self.device = torch.device(device)
        self.n_estimators = n_estimators
        self.training = False
        self.target_selector = True
        self.target_anchor = True
        self.absolute_state = True
        self.persistence_residual = False
        self.persistence_residual_scope = "next_observation"

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def __call__(self, X, X_action=None, y_train=None):
        if y_train is None:
            raise ValueError("y_train is required")
        if X_action is not None:
            X = torch.cat([X, X_action], dim=-1)
        features = X.detach().float().cpu().numpy()
        support = y_train.detach().float().cpu().numpy()
        n = support.shape[1]
        predictions = []
        for i in range(features.shape[0]):
            reg = TabICLRegressor(
                n_estimators=self.n_estimators,
                model_path=self.checkpoint,
                allow_auto_download=False,
                device=str(self.device),
                random_state=42,
            )
            reg.fit(features[i, :n], support[i])
            pred = reg.predict(features[i, n:])
            predictions.append(torch.as_tensor(pred, device=self.device, dtype=X.dtype))
            del reg
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return torch.stack(predictions).unsqueeze(-1)


def set_flags(model, residual=True):
    model.target_selector = True
    model.target_anchor = True
    model.absolute_state = True
    model.persistence_residual = residual
    model.persistence_residual_scope = "next_observation"
    return model


def load(ckpt_path, device):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = build_model(state["base_checkpoint"], device, state["args"].get("model_kind"))
    model.load_state_dict(state["model"], strict=True)
    model.input_scale_profile = state["args"].get("input_scale_profile", "legacy")
    return set_flags(model, residual=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--caq-checkpoint", default=str(CAQ_CKPT))
    p.add_argument("--tabicl-ft-checkpoint", default=str(TABICL_FT_CKPT))
    p.add_argument("--split", choices=["meta_train", "meta_val", "meta_test"], default="meta_val")
    p.add_argument("--context-size", type=int, default=960)
    p.add_argument("--query-size", type=int, default=256)
    p.add_argument("--include-official-zero-shot", action="store_true",
                   help="Also run the official 8-estimator TabICLRegressor pipeline")
    p.add_argument("--official-estimators", type=int, default=8)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.official_estimators < 1:
        p.error("--official-estimators must be positive")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    models = {}
    # Frozen core is deliberately separate from the official sklearn pipeline.
    models["tabicl_frozen_core"] = set_flags(
        build_model(DEFAULT_CHECKPOINT, device, "tabicl"), residual=False).eval()
    models["tabicl_ft"] = load(Path(args.tabicl_ft_checkpoint), device).eval()
    models["caq"] = load(Path(args.caq_checkpoint), device).eval()
    if args.include_official_zero_shot:
        models["tabicl_official_zero_shot"] = OfficialTabICLRegressorAdapter(
            DEFAULT_CHECKPOINT, device, args.official_estimators)

    sampler = IndustrialTaskSampler(MANIFEST, args.split, args.context_size, args.query_size, seed=42,
                                    target_floor_profile="event_v2", external_context_mode="observed",
                                    information_track="full_state", history_length=8, history_ratio=0.)
    tasks = sampler.fixed_tasks()
    pairs = sampler.fixed_tasks(paired=True)
    scm = SCMReplaySampler(batch_size=1, min_features=5, max_features=12,
                           min_seq_len=96, max_seq_len=97, n_jobs=1)
    report, rows = {}, []
    for name, model in models.items():
        pred = evaluate(model, tasks, device, sampler)
        eff = evaluate(model, pairs, device, sampler, paired=True)
        report[name] = {"prediction": pred, "paired_effect": eff,
                        "scm_mse": evaluate_scm(model, scm, device)}
        rows.append((name, pred, eff, report[name]["scm_mse"]))

    out = Path(args.out) if args.out else EXP / "checkpoints/compare-three" / f"{args.split}-hr0.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    print(f"\n== 模型对比 {args.split} (context={args.context_size}, query={args.query_size}) ==")
    print(f"{'model':<15}{'macro_fixed':>11}{'hybrid':>9}{'phys':>9}{'disc':>9}{'effect_ref':>11}{'FP_rate':>9}{'sign_acc':>9}{'scm':>8}")
    for name, pred, eff, scm_mse in rows:
        fam = pred["family_macro_fixed_scale_mse"]
        worlds = eff["worlds"]
        effect_mse = float(np.mean([r["effect_fixed_reference_mse"] for r in worlds]))
        fp = [r["proven_zero_false_positive_rate"] for r in worlds if r.get("proven_zero_false_positive_rate") is not None]
        sign = [r["nonzero_effect_sign_accuracy"] for r in worlds if r.get("nonzero_effect_sign_accuracy") is not None]
        print(f"{name:<15}{pred['macro_fixed_scale_mse']:>11.4f}"
              f"{fam.get('hybrid_line', 0):>9.4f}{fam.get('physical_ode', 0):>9.4f}{fam.get('discrete', 0):>9.4f}"
              f"{effect_mse:>11.4f}"
              f"{np.mean(fp) if fp else 0:>9.4f}"
              f"{np.mean(sign) if sign else 0:>9.4f}{scm_mse:>8.4f}")
    # 参考基线（对所有模型相同：数据驱动）
    w = report["caq"]["prediction"]["worlds"]
    print(f"{'persistence':<15}{np.mean([r['persistence_fixed_scale_mse'] for r in w]):>11.4f}")
    print(f"{'ridge':<15}{np.mean([r['ridge_fixed_scale_mse'] for r in w]):>11.4f}")
    print(f"\n报告已写入 {out}")


if __name__ == "__main__":
    main()
