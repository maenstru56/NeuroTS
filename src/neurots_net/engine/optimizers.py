from __future__ import annotations

import torch


def build_optimizer(
    parameters,
    name: str,
    learning_rate: float,
    weight_decay: float,
    adam_eps: float,
) -> torch.optim.Optimizer:
    key = str(name).strip().lower()
    if key == "adamw":
        return torch.optim.AdamW(parameters, lr=float(learning_rate), weight_decay=float(weight_decay), eps=float(adam_eps))
    if key == "adam":
        return torch.optim.Adam(parameters, lr=float(learning_rate), weight_decay=float(weight_decay), eps=float(adam_eps))
    if key == "sgd":
        return torch.optim.SGD(parameters, lr=float(learning_rate), weight_decay=float(weight_decay), momentum=0.99, nesterov=True)
    raise ValueError(f"Unsupported optimizer: {name}")
