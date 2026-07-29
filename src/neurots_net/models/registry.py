from __future__ import annotations

from neurots_net.config import ModelConfig
from neurots_net.models.neurots_net import NeuroTS_Net_v1


def build_model(cfg: ModelConfig) -> NeuroTS_Net_v1:
    name = str(cfg.name).strip().lower()
    if name not in {"neurots_net_v1", "neurots-net-v1", "neurots_net"}:
        raise ValueError(f"Unknown NeuroTS model '{cfg.name}'.")
    return NeuroTS_Net_v1(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        base_channels=cfg.base_channels,
        model_id=cfg.model_id,
        kernel_size=cfg.kernel_size,
        expansion_ratios=cfg.expansion_ratios,
        block_counts=cfg.block_counts,
        normalization=cfg.normalization,
        dropout=cfg.dropout,
        deep_supervision=cfg.deep_supervision,
        residual_blocks=cfg.residual_blocks,
        residual_resampling=cfg.residual_resampling,
        checkpoint_style=cfg.checkpoint_style,
        use_global_response_norm=cfg.use_global_response_norm,
        use_selective_branches=cfg.use_selective_branches,
        branch_dropout_prob=cfg.branch_dropout_prob,
        branch_selection_topk=cfg.branch_selection_topk,
        branch_temperature_start=cfg.branch_temperature_start,
        branch_temperature_end=cfg.branch_temperature_end,
        branch_residual_gain_init=cfg.branch_residual_gain_init,
        use_raw_detail_stream=cfg.use_raw_detail_stream,
        raw_detail_channels=cfg.raw_detail_channels,
        raw_detail_injection_init=cfg.raw_detail_injection_init,
    )
