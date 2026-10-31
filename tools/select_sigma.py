"""Select sigma and validation threshold without touching the final test split.

For each class and candidate sigma, this script trains on train/ and selects
the checkpoint by val I-AUROC. It then selects the best sigma by validation
I-AUROC and writes a validation-only threshold.json.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rsad.data.datasets import TestDataset, mask_collate
from rsad.data.transforms import test_transform
from rsad.metrics import compute_metrics
from rsad.models.rsad_model import RSADModel
from rsad.utils.checkpoint import load_ckpt
from rsad.utils.config import load_config


def val_score(cfg, checkpoint, cls):
    device = torch.device(cfg.device if cfg.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = RSADModel(cfg).to(device)
    load_ckpt(model, checkpoint, device)
    model.eval()
    ds = TestDataset(Path(cfg.val_root), test_transform(cfg.crop), classes=[cls])
    loader = DataLoader(ds, batch_size=cfg.batch, shuffle=False,
                        num_workers=0, collate_fn=mask_collate)
    return compute_metrics(model, loader, device)


def val_threshold(cfg, checkpoint, cls, q):
    device = torch.device(cfg.device if cfg.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = RSADModel(cfg).to(device)
    load_ckpt(model, checkpoint, device)
    model.eval()
    ds = TestDataset(Path(cfg.val_root), test_transform(cfg.crop), classes=[cls])
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=mask_collate)
    scores = []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            probs = model.test_forward(imgs.to(device))
            sad = (1.0 - probs).amax(dim=(1, 2, 3)).cpu().numpy()
            scores.extend(float(s) for s, label in zip(sad, labels) if label == 0)
    scores = np.asarray(scores)
    return float(np.percentile(scores, q * 100)), scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mpdd.yaml")
    ap.add_argument("--classes", nargs="+", default=[
        "bracket_black", "bracket_brown", "bracket_white",
        "connector", "metal_plate", "tubes"])
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.05, 0.1, 0.15])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--q", type=float, default=0.95)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="checkpoints_v3_sigma_selection")
    args = ap.parse_args()

    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for cls in args.classes:
        candidates = []
        for sigma in args.sigmas:
            candidate_out = out / "candidates" / cls / f"sigma_{sigma:g}"
            best = candidate_out / cls / "best.pt"
            done = candidate_out / "SELECTION_TRAIN_DONE"
            if not done.exists():
                cmd = [sys.executable, str(ROOT / "train.py"),
                       "--config", str(ROOT / args.config), "--cls", cls,
                       "--sigma", str(sigma), "--epochs", str(args.epochs),
                       "--batch", str(args.batch), "--device", args.device,
                       "--out", str(candidate_out)]
                env = dict(**__import__("os").environ, RSAD_DISABLE_FINAL_TEST="1")
                subprocess.run(cmd, cwd=ROOT, env=env, check=True)
                done.write_text("train_only; final_test_disabled\n", encoding="ascii")
            cfg = load_config(str(ROOT / args.config))
            cfg.cls, cfg.sigma, cfg.batch, cfg.device = cls, sigma, args.batch, args.device
            i_auc, p_auc = val_score(cfg, best, cls)
            candidates.append({"sigma": sigma, "checkpoint": str(best),
                               "val_i_auroc": i_auc, "val_p_auroc": p_auc})
            print(f"{cls} sigma={sigma:g}: val I={i_auc:.4f}, P={p_auc:.4f}")
        selected = max(candidates, key=lambda x: x["val_i_auroc"])
        threshold, normal_scores = val_threshold(cfg, selected["checkpoint"], cls, args.q)
        final_dir = out / cls
        final_dir.mkdir(parents=True, exist_ok=True)
        (final_dir / "threshold.json").write_text(json.dumps({
            "th": threshold, "q": args.q, "n": int(len(normal_scores)),
            "selection_split": "val", "sigma": selected["sigma"],
        }, indent=2), encoding="utf-8")
        results[cls] = {"selected": selected, "threshold": threshold,
                        "candidates": candidates}
        print(f"{cls}: selected sigma={selected['sigma']:g}, threshold={threshold:.6f}")
    (out / "selection_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
