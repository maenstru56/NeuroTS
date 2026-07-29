from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from neurots_net.config import ExperimentConfig
from neurots_net.losses import build_loss
from neurots_net.models.registry import build_model


def main() -> None:
    cfg = ExperimentConfig()
    cfg.model.base_channels = 4
    cfg.model.block_counts = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    cfg.model.expansion_ratios = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    cfg.model.raw_detail_channels = 2
    cfg.model.deep_supervision = True
    model = build_model(cfg.model)
    x = torch.randn(1, 4, 32, 32, 32)
    y = torch.randint(0, 5, (1, 32, 32, 32))
    prediction = model(x)
    loss = build_loss(cfg.loss, cfg.model.num_classes)(prediction, y)
    loss.backward()
    logits = prediction[0] if isinstance(prediction, tuple) else prediction
    print({"logits_shape": tuple(logits.shape), "loss": float(loss.detach().item())})


if __name__ == "__main__":
    main()
