import math
from typing import Dict, Optional

import torch


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        if self.best is None:
            self.best = metric
            return False
        improved = metric < self.best if self.mode == "min" else metric > self.best
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def build_optimizer(model_or_params, lr: float, weight_decay: float):
    if hasattr(model_or_params, "parameters"):
        params = model_or_params.parameters()
    else:
        params = model_or_params
    return torch.optim.AdamW(filter(lambda p: p.requires_grad, params), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(path: str, model, optimizer=None, meta: Optional[Dict] = None):
    payload = {"model_state": model.state_dict()}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if meta:
        payload["meta"] = meta
    torch.save(payload, path)


def load_checkpoint(path: str, model, optimizer=None):
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def set_requires_grad(module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def maybe_compile(model, optimization_cfg: dict):
    if not optimization_cfg:
        return model
    if not optimization_cfg.get("compile_model", False):
        return model
    if not hasattr(torch, "compile"):
        return model
    mode = optimization_cfg.get("compile_mode", "default")
    try:
        return torch.compile(model, mode=mode)
    except Exception:
        return model
