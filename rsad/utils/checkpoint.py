"""Checkpoint save/load, including optimizer, scheduler and RNG state."""
import random

import numpy as np
import torch
from pathlib import Path


def _rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.set_rng_state requires a CPU ByteTensor.  When the checkpoint was
    # loaded with map_location=cuda, the saved RNG tensor is otherwise moved to
    # the GPU and resume fails before training can continue.
    torch_state = state["torch"]
    if not isinstance(torch_state, torch.Tensor):
        torch_state = torch.tensor(torch_state, dtype=torch.uint8)
    torch.set_rng_state(torch_state.detach().cpu().to(dtype=torch.uint8))
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = state["cuda"]
        if not isinstance(cuda_states, (list, tuple)):
            cuda_states = [cuda_states]
        cuda_states = [
            (s if isinstance(s, torch.Tensor)
             else torch.tensor(s, dtype=torch.uint8))
            .detach().cpu().to(dtype=torch.uint8)
            for s in cuda_states
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def save_ckpt(model, out_dir, name, epoch, best, sigma,
              optimizer=None, scheduler=None, meta=None):
    """Save model and, when supplied, the complete training state."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ck = {
        "aligner": model.aligner.state_dict(),
        "disc": model.disc.state_dict(),
        "sigma": sigma,
        "epoch": epoch,
        "best": best,
        "feature_norm": bool(getattr(model.cfg, "feature_norm", False)),
        "contrast": getattr(model.cfg, "contrast", None),
        "rng_state": _rng_state(),
        "run_meta": dict(meta or {}),
    }
    if optimizer is not None:
        ck["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ck["scheduler"] = scheduler.state_dict()
    path = Path(out_dir) / name
    torch.save(ck, path)
    return path


def load_ckpt(model, path, device, optimizer=None, scheduler=None,
              restore_rng=True):
    ck = torch.load(path, map_location=device)
    # extractor 冻结不保存; 兼容旧 ckpt 携带 extractor 的情况
    if "extractor" in ck:
        model.extractor.load_state_dict(ck["extractor"])
    model.aligner.load_state_dict(ck["aligner"])
    model.disc.load_state_dict(ck["disc"])
    # 结构标志写回 cfg, 保证评估/推理与训练行为一致
    if "feature_norm" in ck:
        model.cfg.feature_norm = bool(ck["feature_norm"])
    if "contrast" in ck:
        model.cfg.contrast = ck["contrast"]
    epoch = int(ck.get("epoch", 0))
    best = float(ck.get("best", 0.0))
    sigma = ck.get("sigma", 0.015)
    if optimizer is not None and "optimizer" in ck:
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and "scheduler" in ck:
        scheduler.load_state_dict(ck["scheduler"])
    if restore_rng and "rng_state" in ck:
        _restore_rng_state(ck["rng_state"])
    return epoch, best, sigma
