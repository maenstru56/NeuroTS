import torch

from neurots_net.config import ExperimentConfig
from neurots_net.losses import build_loss
from neurots_net.models.registry import build_model


def test_tiny_forward_backward_and_absent_rare_penalty() -> None:
    cfg = ExperimentConfig()
    cfg.model.base_channels = 4
    cfg.model.model_id = "S"
    cfg.model.block_counts = [1] * 9
    cfg.model.expansion_ratios = [1] * 9
    cfg.model.raw_detail_channels = 2
    cfg.model.deep_supervision = True
    cfg.loss.batch_dice = False
    cfg.loss.absent_rare_fp_weight = 0.1

    model = build_model(cfg.model)
    criterion = build_loss(cfg.loss, cfg.model.num_classes)
    image = torch.randn(1, 4, 32, 32, 32)
    target = torch.zeros((1, 32, 32, 32), dtype=torch.long)
    target[:, 8:24, 8:24, 8:24] = 2

    prediction = model(image)
    loss = criterion(prediction, target)
    loss.backward()

    logits = prediction[0] if isinstance(prediction, tuple) else prediction
    assert logits.shape == (1, 5, 32, 32, 32)
    assert torch.isfinite(loss)
    assert "loss_absent_rare_fp" in criterion.last_components
