
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint


NEUROTS_VARIANTS: dict[str, dict[str, int | list[int] | str]] = {
    "S": {"model_id": "S", "expansion": 2, "blocks": [2, 2, 2, 2, 2, 2, 2, 2, 2]},
    "B": {"model_id": "B", "expansion": [2, 3, 4, 4, 4, 4, 4, 3, 2], "blocks": [2, 2, 2, 2, 2, 2, 2, 2, 2]},
    "M": {"model_id": "M", "expansion": [2, 3, 4, 4, 4, 4, 4, 3, 2], "blocks": [3, 4, 4, 4, 4, 4, 4, 4, 3]},
    "L": {"model_id": "L", "expansion": [3, 4, 8, 8, 8, 8, 8, 4, 3], "blocks": [3, 4, 8, 8, 8, 8, 8, 4, 3]},
}


def _normalize_model_id(model_id: str | None) -> str:
    normalized = str(model_id or "L").strip().upper()
    if normalized not in NEUROTS_VARIANTS:
        supported = ", ".join(sorted(NEUROTS_VARIANTS))
        raise ValueError(f"Unknown NeuroTS model_id '{model_id}'. Supported: {supported}.")
    return normalized


def _as_nine_ints(value: int | Sequence[int], name: str) -> list[int]:
    if isinstance(value, int):
        return [int(value)] * 9
    result = [int(item) for item in value]
    if len(result) != 9:
        raise ValueError(f"{name} must have exactly 9 entries.")
    return result


class LayerNorm3d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return x * self.weight.view(shape) + self.bias.view(shape)


class GlobalResponseNorm3d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        work_dtype = torch.float32 if x.dtype in {torch.float16, torch.bfloat16} else x.dtype
        x_work = x.to(dtype=work_dtype)
        gx = torch.linalg.vector_norm(x_work, ord=2, dim=(2, 3, 4), keepdim=True)
        nx = gx / gx.mean(dim=1, keepdim=True).clamp_min(self.eps)
        y = x_work + self.gamma.to(dtype=work_dtype) * (x_work * nx) + self.beta.to(dtype=work_dtype)
        return y.to(dtype=x.dtype)


def _build_norm(channels: int, normalization: str) -> nn.Module:
    key = str(normalization).strip().lower()
    if key in {"group", "groupnorm", "group_norm"}:
        return nn.GroupNorm(num_groups=channels, num_channels=channels)
    if key in {"layer", "layernorm", "layer_norm"}:
        return LayerNorm3d(channels)
    if key in {"instance", "instancenorm", "instance_norm"}:
        return nn.InstanceNorm3d(channels, affine=True)
    if key in {"batch", "batchnorm", "batch_norm"}:
        return nn.BatchNorm3d(channels)
    if key in {"none", "identity"}:
        return nn.Identity()
    raise ValueError("normalization must be one of group, layer, instance, batch, none.")


class _GlobalContextBranch3d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.project = nn.Conv3d(channels, channels, kernel_size=1, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = F.adaptive_avg_pool3d(x, output_size=1)
        return self.project(context).expand_as(x)


def _feature_gradient_magnitude_3d(x: torch.Tensor) -> torch.Tensor:
    dz = F.pad(torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :]), (0, 0, 0, 0, 1, 0))
    dy = F.pad(torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]), (0, 0, 1, 0, 0, 0))
    dx = F.pad(torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]), (1, 0, 0, 0, 0, 0))
    return (dz + dy + dx) / 3.0


class RawImageDetailProjector3d(nn.Module):
    def __init__(self, in_channels: int, detail_channels: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv3d(3 * int(in_channels), int(detail_channels), kernel_size=1),
            nn.GELU(),
            nn.Conv3d(int(detail_channels), int(detail_channels), kernel_size=3, padding=1, groups=int(detail_channels)),
            nn.GELU(),
        )

    @staticmethod
    def _low_pass(x: torch.Tensor) -> torch.Tensor:
        if min(int(item) for item in x.shape[2:]) < 3:
            return x.mean(dim=(2, 3, 4), keepdim=True).expand_as(x)
        return F.avg_pool3d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        high_frequency = torch.abs(image - self._low_pass(image))
        gradient = _feature_gradient_magnitude_3d(image)
        return self.project(torch.cat((image, high_frequency, gradient), dim=1))


class RawDetailInjection3d(nn.Module):
    def __init__(self, detail_channels: int, feature_channels: int, init_gain: float = 1e-3) -> None:
        super().__init__()
        self.detail_to_feature = nn.Conv3d(int(detail_channels), int(feature_channels), kernel_size=1)
        self.detail_gate = nn.Conv3d(int(feature_channels), int(feature_channels), kernel_size=1, groups=int(feature_channels))
        self.alpha = nn.Parameter(torch.tensor(float(init_gain)))
        nn.init.zeros_(self.detail_gate.weight)
        nn.init.zeros_(self.detail_gate.bias)

    def forward(self, features: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        if detail.shape[2:] != features.shape[2:]:
            detail = F.interpolate(detail, size=features.shape[2:], mode="trilinear", align_corners=False)
        delta = self.detail_to_feature(detail)
        gate = torch.sigmoid(self.detail_gate(features))
        return features + self.alpha.to(dtype=features.dtype, device=features.device) * gate * delta.to(dtype=features.dtype)


class NeuroTS_Block_v1(nn.Module):
    branch_libraries: dict[int, tuple[str, ...]] = {
        3: ("conv_5x5x5",),
        4: ("conv_5x5x5", "global_context"),
    }

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 4,
        kernel_size: int = 3,
        residual: bool = True,
        normalization: str = "group",
        use_global_response_norm: bool = True,
        dropout: float = 0.0,
        stage_index: int = 0,
        use_selective_branches: bool = True,
        branch_dropout_prob: float = 0.05,
        branch_selection_topk: int = 2,
        branch_temperature_start: float = 2.0,
        branch_temperature_end: float = 0.5,
        branch_residual_gain_init: float = 0.005,
        detail_channels: int | None = None,
        detail_gain_init: float = 1e-3,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("NeuroTS block kernel_size must be odd.")
        self.stage_index = int(stage_index)
        self.residual = bool(residual)
        self.selector_eps = 1e-6
        self.branch_dropout_prob = float(branch_dropout_prob)
        self.branch_selection_topk = int(branch_selection_topk)
        self.branch_temperature_start = float(branch_temperature_start)
        self.branch_temperature_end = float(branch_temperature_end)
        self.selection_progress = 0.0
        self.sparse_selection_start = 0.3
        self.usage_regularization_weight = 0.0
        self.branch_usage_loss: torch.Tensor | None = None
        self.last_branch_usage: torch.Tensor | None = None
        self.last_branch_entropy: float | None = None
        self.register_buffer("selection_temperature", torch.tensor(float(branch_temperature_start)), persistent=False)

        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
        )
        self.branch_names = self.branch_libraries.get(self.stage_index, tuple()) if use_selective_branches else tuple()
        self.extra_branches = nn.ModuleList([self._make_branch(in_channels, name) for name in self.branch_names])
        if self.branch_names:
            self.branch_selector = nn.Conv1d(2 * len(self.branch_names), len(self.branch_names), kernel_size=1)
            self.detail_selector = nn.Conv3d(in_channels, len(self.branch_names), kernel_size=1)
            self.branch_prior = nn.Parameter(torch.zeros(1, len(self.branch_names), 1, 1, 1, 1))
            self.detail_lambda = nn.Parameter(torch.full((1, len(self.branch_names), 1, 1, 1, 1), 0.1))
            self.branch_residual_gain = nn.Parameter(torch.tensor(float(branch_residual_gain_init)))
            nn.init.zeros_(self.branch_selector.weight)
            nn.init.zeros_(self.branch_selector.bias)
            nn.init.zeros_(self.detail_selector.weight)
            nn.init.zeros_(self.detail_selector.bias)
        else:
            self.branch_selector = None
            self.detail_selector = None
            self.branch_prior = None
            self.detail_lambda = None
            self.branch_residual_gain = None

        if detail_channels is not None and int(detail_channels) > 0:
            self.detail_adapter = nn.Conv3d(int(detail_channels), in_channels, kernel_size=1)
            self.detail_gate = nn.Conv3d(in_channels, in_channels, kernel_size=1, groups=in_channels)
            self.detail_gain = nn.Parameter(torch.tensor(float(detail_gain_init)))
            nn.init.zeros_(self.detail_gate.weight)
            nn.init.zeros_(self.detail_gate.bias)
        else:
            self.detail_adapter = None
            self.detail_gate = None
            self.detail_gain = None

        hidden_channels = int(expansion) * int(in_channels)
        self.norm = _build_norm(in_channels, normalization)
        self.expand = nn.Conv3d(in_channels, hidden_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.response_norm = GlobalResponseNorm3d(hidden_channels) if use_global_response_norm else nn.Identity()
        self.project = nn.Conv3d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout3d(float(dropout)) if float(dropout) > 0 else nn.Identity()

    @staticmethod
    def _make_branch(channels: int, name: str) -> nn.Module:
        if name == "conv_5x5x5":
            return nn.Conv3d(channels, channels, kernel_size=5, padding=2, groups=channels)
        if name == "global_context":
            return _GlobalContextBranch3d(channels)
        raise ValueError(f"Unknown NeuroTS branch '{name}'.")

    def set_selection_progress(self, progress: float) -> None:
        progress = min(1.0, max(0.0, float(progress)))
        self.selection_progress = progress
        temperature = self.branch_temperature_start + (self.branch_temperature_end - self.branch_temperature_start) * progress
        self.selection_temperature.fill_(float(temperature))

    def set_usage_regularization_weight(self, weight: float) -> None:
        self.usage_regularization_weight = max(0.0, float(weight))

    @staticmethod
    def _low_pass(x: torch.Tensor) -> torch.Tensor:
        if min(int(item) for item in x.shape[2:]) < 3:
            return x.mean(dim=(2, 3, 4), keepdim=True).expand_as(x)
        return F.avg_pool3d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)

    def _apply_branch_dropout(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.training or self.branch_dropout_prob <= 0.0 or len(self.branch_names) <= 1:
            return logits
        batch_size = int(logits.shape[0])
        should_drop = torch.rand(batch_size, device=logits.device) < self.branch_dropout_prob
        if not bool(should_drop.any()):
            return logits
        drop_indices = torch.randint(len(self.branch_names), (batch_size,), device=logits.device)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[torch.arange(batch_size, device=logits.device), drop_indices] = should_drop.view(batch_size, 1, 1, 1, 1)
        return logits.masked_fill(mask, torch.finfo(logits.dtype).min)

    def _apply_sparse_topk(self, weights: torch.Tensor) -> torch.Tensor:
        if (
            self.selection_progress < self.sparse_selection_start
            or self.branch_selection_topk <= 0
            or self.branch_selection_topk >= len(self.branch_names)
        ):
            return weights
        _, indices = torch.topk(weights, k=self.branch_selection_topk, dim=1)
        mask = torch.zeros_like(weights, dtype=torch.bool).scatter_(1, indices, True)
        sparse_weights = weights.masked_fill(~mask, 0.0)
        return sparse_weights / sparse_weights.sum(dim=1, keepdim=True).clamp_min(self.selector_eps)

    def _record_branch_usage(self, weights: torch.Tensor) -> None:
        usage = weights.to(dtype=torch.float32).mean(dim=(0, 2, 3, 4, 5))
        usage = usage / usage.sum().clamp_min(self.selector_eps)
        usage_safe = usage.clamp_min(self.selector_eps)
        entropy = -(usage_safe * usage_safe.log()).sum()
        self.last_branch_usage = usage.detach()
        self.last_branch_entropy = float(entropy.detach().item())
        if self.training and self.usage_regularization_weight > 0.0:
            self.branch_usage_loss = -float(self.usage_regularization_weight) * entropy
        else:
            self.branch_usage_loss = None

    def _branch_weights(self, x: torch.Tensor, extra_stack: torch.Tensor) -> torch.Tensor:
        if self.branch_selector is None or self.detail_selector is None:
            raise RuntimeError("Selective branches are not configured for this block.")
        work_dtype = torch.float32 if extra_stack.dtype in {torch.float16, torch.bfloat16} else extra_stack.dtype
        extra_work = extra_stack.to(dtype=work_dtype)
        response = torch.linalg.vector_norm(extra_work, ord=2, dim=(3, 4, 5))
        channel_norm = response / response.mean(dim=2, keepdim=True).clamp_min(self.selector_eps)
        branch_norm = response / response.sum(dim=1, keepdim=True).clamp_min(self.selector_eps)
        descriptor = torch.cat((channel_norm, branch_norm), dim=1)
        global_logits = self.branch_selector(descriptor.to(dtype=self.branch_selector.weight.dtype)).to(dtype=work_dtype)
        global_logits = global_logits.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        detail = torch.abs(x - self._low_pass(x))
        detail_gate = torch.sigmoid(self.detail_selector(detail)).to(dtype=work_dtype).unsqueeze(2)
        logits = (
            global_logits
            + self.detail_lambda.to(dtype=work_dtype, device=x.device) * detail_gate
            + self.branch_prior.to(dtype=work_dtype, device=x.device)
        )
        logits = self._apply_branch_dropout(logits)
        temperature = self.selection_temperature.to(dtype=work_dtype, device=x.device).clamp_min(0.05)
        weights = torch.softmax(logits / temperature, dim=1)
        weights = self._apply_sparse_topk(weights)
        self._record_branch_usage(weights)
        return weights.to(dtype=extra_stack.dtype)

    def _mixed_depthwise(self, x: torch.Tensor, detail: torch.Tensor | None) -> torch.Tensor:
        base = self.depthwise(x)
        if detail is not None and self.detail_adapter is not None and self.detail_gate is not None and self.detail_gain is not None:
            if detail.shape[2:] != base.shape[2:]:
                detail = F.interpolate(detail, size=base.shape[2:], mode="trilinear", align_corners=False)
            detail_delta = self.detail_adapter(detail).to(dtype=base.dtype)
            detail_gate = torch.sigmoid(self.detail_gate(x)).to(dtype=base.dtype)
            base = base + self.detail_gain.to(dtype=base.dtype, device=base.device) * detail_gate * detail_delta
        if not self.branch_names:
            return base
        extra_stack = torch.stack([branch(x) for branch in self.extra_branches], dim=1)
        weights = self._branch_weights(x, extra_stack)
        correction = (extra_stack * weights).sum(dim=1)
        return base + self.branch_residual_gain.to(dtype=base.dtype, device=base.device) * correction

    def forward(self, x: torch.Tensor, detail: torch.Tensor | None = None) -> torch.Tensor:
        y = self._mixed_depthwise(x, detail)
        y = self.norm(y)
        y = self.expand(y)
        y = self.activation(y)
        y = self.response_norm(y)
        y = self.project(y)
        y = self.drop(y)
        if self.residual:
            if x.shape != y.shape:
                raise ValueError("NeuroTS residual block requires matching input and output shapes.")
            y = x + y
        return y


class NeuroTS_Stage_v1(nn.Module):
    def __init__(self, blocks: Sequence[NeuroTS_Block_v1]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, detail: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, detail)
        return x


class NeuroTS_DownBlock_v1(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int,
        kernel_size: int,
        residual: bool,
        normalization: str,
        use_global_response_norm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.residual = bool(residual)
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )
        self.norm = _build_norm(in_channels, normalization)
        hidden_channels = int(expansion) * int(in_channels)
        self.expand = nn.Conv3d(in_channels, hidden_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.response_norm = GlobalResponseNorm3d(hidden_channels) if use_global_response_norm else nn.Identity()
        self.project = nn.Conv3d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout3d(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.residual_projection = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2) if self.residual else nn.Identity()
        self.pool_fuse = nn.Conv3d(out_channels + 2 * in_channels, out_channels, kernel_size=1)
        self.pool_norm = GlobalResponseNorm3d(out_channels) if use_global_response_norm else nn.Identity()
        self.anti_alias_gate = nn.Parameter(torch.zeros(1, out_channels, 1, 1, 1))

    @staticmethod
    def _avg_downsample(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(x, kernel_size=3, stride=2, padding=1, count_include_pad=False)

    @staticmethod
    def _max_downsample(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool3d(x, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x)
        y = self.norm(y)
        y = self.expand(y)
        y = self.activation(y)
        y = self.response_norm(y)
        y = self.project(y)
        y = self.drop(y)
        if self.residual:
            y = y + self.residual_projection(x)
        pooled = torch.cat((y, self._avg_downsample(x), self._max_downsample(x)), dim=1)
        correction = self.pool_norm(self.pool_fuse(pooled))
        return y + self.anti_alias_gate.to(dtype=y.dtype, device=y.device) * correction


class NeuroTS_UpBlock_v1(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int,
        kernel_size: int,
        residual: bool,
        normalization: str,
        use_global_response_norm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.residual = bool(residual)
        self.depthwise = nn.ConvTranspose3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )
        self.norm = _build_norm(in_channels, normalization)
        hidden_channels = int(expansion) * int(in_channels)
        self.expand = nn.Conv3d(in_channels, hidden_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.response_norm = GlobalResponseNorm3d(hidden_channels) if use_global_response_norm else nn.Identity()
        self.project = nn.Conv3d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout3d(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.residual_projection = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=1, stride=2) if self.residual else nn.Identity()

    @staticmethod
    def _pad_transposed_output(x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (1, 0, 1, 0, 1, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._pad_transposed_output(self.depthwise(x))
        y = self.norm(y)
        y = self.expand(y)
        y = self.activation(y)
        y = self.response_norm(y)
        y = self.project(y)
        y = self.drop(y)
        if self.residual:
            y = y + self._pad_transposed_output(self.residual_projection(x))
        return y


class NeuroTS_OutBlock_v1(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.output = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(x)


class NeuroTS_Net_v1(nn.Module):
    raw_detail_stage_indices = frozenset({0, 1})

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 5,
        base_channels: int = 32,
        model_id: str = "L",
        kernel_size: int = 3,
        expansion_ratios: int | Sequence[int] | None = None,
        block_counts: Sequence[int] | None = None,
        normalization: str = "group",
        dropout: float = 0.0,
        deep_supervision: bool = True,
        residual_blocks: bool = True,
        residual_resampling: bool = True,
        checkpoint_style: str | None = "outside_block",
        use_global_response_norm: bool = True,
        use_selective_branches: bool = True,
        branch_dropout_prob: float = 0.05,
        branch_selection_topk: int = 2,
        branch_temperature_start: float = 2.0,
        branch_temperature_end: float = 0.5,
        branch_residual_gain_init: float = 0.005,
        use_raw_detail_stream: bool = True,
        raw_detail_channels: int = 8,
        raw_detail_injection_init: float = 1e-3,
    ) -> None:
        super().__init__()
        if checkpoint_style not in {None, "outside_block"}:
            raise ValueError("checkpoint_style must be None or 'outside_block'.")
        model_id = _normalize_model_id(model_id)
        variant = NEUROTS_VARIANTS[model_id]
        expansion = variant["expansion"] if expansion_ratios is None else expansion_ratios
        blocks = variant["blocks"] if block_counts is None else block_counts
        self.expansion_ratios = _as_nine_ints(expansion, "expansion_ratios")
        self.block_counts = _as_nine_ints(blocks, "block_counts")
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.base_channels = int(base_channels)
        self.deep_supervision = bool(deep_supervision)
        self.outside_block_checkpointing = checkpoint_style == "outside_block"
        self.use_raw_detail_stream = bool(use_raw_detail_stream)
        self.raw_detail_channels = int(raw_detail_channels)
        self.raw_detail_injection_init = float(raw_detail_injection_init)

        n = self.base_channels
        self.stem = nn.Conv3d(self.in_channels, n, kernel_size=1)
        if self.use_raw_detail_stream:
            self.raw_detail_projector_0 = RawImageDetailProjector3d(self.in_channels, self.raw_detail_channels)
            self.raw_detail_projector_1 = RawImageDetailProjector3d(self.in_channels, self.raw_detail_channels)
            self.raw_detail_inject_0 = RawDetailInjection3d(self.raw_detail_channels, n, init_gain=self.raw_detail_injection_init)
            self.raw_detail_inject_1 = RawDetailInjection3d(self.raw_detail_channels, 2 * n, init_gain=self.raw_detail_injection_init)

        stage_kwargs: dict[str, Any] = {
            "kernel_size": kernel_size,
            "normalization": normalization,
            "dropout": dropout,
            "residual": residual_blocks,
            "use_global_response_norm": use_global_response_norm,
            "use_selective_branches": use_selective_branches,
            "branch_dropout_prob": branch_dropout_prob,
            "branch_selection_topk": branch_selection_topk,
            "branch_temperature_start": branch_temperature_start,
            "branch_temperature_end": branch_temperature_end,
            "branch_residual_gain_init": branch_residual_gain_init,
        }
        self.enc_block_0 = self._make_stage(n, n, stage_index=0, detail_channels=self.raw_detail_channels if self.use_raw_detail_stream else None, **stage_kwargs)
        self.down_0 = self._make_down_block(n, 2 * n, stage_index=1, kernel_size=kernel_size, residual=residual_resampling, normalization=normalization, use_global_response_norm=use_global_response_norm, dropout=dropout)
        self.enc_block_1 = self._make_stage(2 * n, 2 * n, stage_index=1, detail_channels=self.raw_detail_channels if self.use_raw_detail_stream else None, **stage_kwargs)
        self.down_1 = self._make_down_block(2 * n, 4 * n, stage_index=2, kernel_size=kernel_size, residual=residual_resampling, normalization=normalization, use_global_response_norm=use_global_response_norm, dropout=dropout)
        self.enc_block_2 = self._make_stage(4 * n, 4 * n, stage_index=2, **stage_kwargs)
        self.down_2 = self._make_down_block(4 * n, 8 * n, stage_index=3, kernel_size=kernel_size, residual=residual_resampling, normalization=normalization, use_global_response_norm=use_global_response_norm, dropout=dropout)
        self.enc_block_3 = self._make_stage(8 * n, 8 * n, stage_index=3, **stage_kwargs)
        self.down_3 = self._make_down_block(8 * n, 16 * n, stage_index=4, kernel_size=kernel_size, residual=residual_resampling, normalization=normalization, use_global_response_norm=use_global_response_norm, dropout=dropout)
        self.bottleneck = self._make_stage(16 * n, 16 * n, stage_index=4, **stage_kwargs)
        self.up_3 = self._make_up_block(16 * n, 8 * n, stage_index=5, kernel_size=kernel_size, residual=residual_resampling, normalization=normalization, use_global_response_norm=use_global_response_norm, dropout=dropout)
        self.dec_block_3 = self._make_stage(8 * n, 8 * n, stage_index=5, **stage_kwargs)
        self.up_2 = self._make_up_block(8 * n, 4 * n, stage_index=6, kernel_size=kernel_size, residual=residual_resampling, normalization=normalization, use_global_response_norm=use_global_response_norm, dropout=dropout)
        self.dec_block_2 = self._make_stage(4 * n, 4 * n, stage_index=6, **stage_kwargs)
        self.up_1 = self._make_up_block(4 * n, 2 * n, stage_index=7, kernel_size=kernel_size, residual=residual_resampling, normalization=normalization, use_global_response_norm=use_global_response_norm, dropout=dropout)
        self.dec_block_1 = self._make_stage(2 * n, 2 * n, stage_index=7, **stage_kwargs)
        self.up_0 = self._make_up_block(2 * n, n, stage_index=8, kernel_size=kernel_size, residual=residual_resampling, normalization=normalization, use_global_response_norm=use_global_response_norm, dropout=dropout)
        self.dec_block_0 = self._make_stage(n, n, stage_index=8, **stage_kwargs)
        self.out_0 = NeuroTS_OutBlock_v1(n, self.num_classes)
        if self.deep_supervision:
            self.out_1 = NeuroTS_OutBlock_v1(2 * n, self.num_classes)
            self.out_2 = NeuroTS_OutBlock_v1(4 * n, self.num_classes)
            self.out_3 = NeuroTS_OutBlock_v1(8 * n, self.num_classes)
            self.out_4 = NeuroTS_OutBlock_v1(16 * n, self.num_classes)

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stage_index: int,
        kernel_size: int,
        normalization: str,
        dropout: float,
        residual: bool,
        use_global_response_norm: bool,
        use_selective_branches: bool,
        branch_dropout_prob: float,
        branch_selection_topk: int,
        branch_temperature_start: float,
        branch_temperature_end: float,
        branch_residual_gain_init: float,
        detail_channels: int | None = None,
    ) -> NeuroTS_Stage_v1:
        return NeuroTS_Stage_v1(
            [
                NeuroTS_Block_v1(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=self.expansion_ratios[stage_index],
                    kernel_size=kernel_size,
                    residual=residual,
                    normalization=normalization,
                    use_global_response_norm=use_global_response_norm,
                    dropout=dropout,
                    stage_index=stage_index,
                    use_selective_branches=use_selective_branches,
                    branch_dropout_prob=branch_dropout_prob,
                    branch_selection_topk=branch_selection_topk,
                    branch_temperature_start=branch_temperature_start,
                    branch_temperature_end=branch_temperature_end,
                    branch_residual_gain_init=branch_residual_gain_init,
                    detail_channels=detail_channels,
                    detail_gain_init=self.raw_detail_injection_init,
                )
                for _ in range(self.block_counts[stage_index])
            ]
        )

    def _make_down_block(self, in_channels: int, out_channels: int, *, stage_index: int, kernel_size: int, residual: bool, normalization: str, use_global_response_norm: bool, dropout: float) -> NeuroTS_DownBlock_v1:
        return NeuroTS_DownBlock_v1(in_channels, out_channels, self.expansion_ratios[stage_index], kernel_size, residual, normalization, use_global_response_norm, dropout)

    def _make_up_block(self, in_channels: int, out_channels: int, *, stage_index: int, kernel_size: int, residual: bool, normalization: str, use_global_response_norm: bool, dropout: float) -> NeuroTS_UpBlock_v1:
        return NeuroTS_UpBlock_v1(in_channels, out_channels, self.expansion_ratios[stage_index], kernel_size, residual, normalization, use_global_response_norm, dropout)

    @staticmethod
    def _match_spatial(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] == reference.shape[2:]:
            return x
        return F.interpolate(x, size=reference.shape[2:], mode="trilinear", align_corners=False)

    def _forward_module(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.outside_block_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint.checkpoint(module, x, use_reentrant=False)
        return module(x)

    def _forward_stage(self, stage: NeuroTS_Stage_v1, x: torch.Tensor, detail: torch.Tensor | None = None) -> torch.Tensor:
        if self.outside_block_checkpointing and self.training and torch.is_grad_enabled():
            for block in stage.blocks:
                x = checkpoint.checkpoint(block, x, detail, use_reentrant=False)
            return x
        return stage(x, detail)

    @staticmethod
    def _blur_downsample_image(image: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(image, kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def _raw_detail_pyramid(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_1 = self._blur_downsample_image(image)
        return self.raw_detail_projector_0(image), self.raw_detail_projector_1(image_1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if x.ndim != 5:
            raise ValueError("NeuroTS_Net_v1 expects input with shape [B,C,D,H,W].")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}.")
        if self.use_raw_detail_stream:
            detail_0, detail_1 = self._raw_detail_pyramid(x)
        else:
            detail_0 = detail_1 = None

        x = self.stem(x)
        x_res_0 = self._forward_stage(self.enc_block_0, x, detail_0)
        if self.use_raw_detail_stream:
            x_res_0 = self.raw_detail_inject_0(x_res_0, detail_0)
        x = self._forward_module(self.down_0, x_res_0)
        x_res_1 = self._forward_stage(self.enc_block_1, x, detail_1)
        if self.use_raw_detail_stream:
            x_res_1 = self.raw_detail_inject_1(x_res_1, detail_1)
        x = self._forward_module(self.down_1, x_res_1)
        x_res_2 = self._forward_stage(self.enc_block_2, x)
        x = self._forward_module(self.down_2, x_res_2)
        x_res_3 = self._forward_stage(self.enc_block_3, x)
        x = self._forward_module(self.down_3, x_res_3)
        x = self._forward_stage(self.bottleneck, x)

        aux_outputs: list[torch.Tensor] = []
        if self.deep_supervision:
            aux_outputs.append(self.out_4(x))
        x = self._match_spatial(self._forward_module(self.up_3, x), x_res_3)
        x = self._forward_stage(self.dec_block_3, x + x_res_3)
        if self.deep_supervision:
            aux_outputs.append(self.out_3(x))
        x = self._match_spatial(self._forward_module(self.up_2, x), x_res_2)
        x = self._forward_stage(self.dec_block_2, x + x_res_2)
        if self.deep_supervision:
            aux_outputs.append(self.out_2(x))
        x = self._match_spatial(self._forward_module(self.up_1, x), x_res_1)
        x = self._forward_stage(self.dec_block_1, x + x_res_1)
        if self.deep_supervision:
            aux_outputs.append(self.out_1(x))
        x = self._match_spatial(self._forward_module(self.up_0, x), x_res_0)
        x = self._forward_stage(self.dec_block_0, x + x_res_0)
        logits = self.out_0(x)
        if self.deep_supervision:
            resized_aux = tuple(
                F.interpolate(aux, size=logits.shape[2:], mode="trilinear", align_corners=False)
                for aux in reversed(aux_outputs)
            )
            return logits, resized_aux
        return logits


def configure_branch_selection(model: nn.Module, progress: float, regularization_weight: float) -> None:
    for module in model.modules():
        if isinstance(module, NeuroTS_Block_v1) and module.branch_names:
            module.set_selection_progress(progress)
            module.set_usage_regularization_weight(regularization_weight)


def branch_usage_regularization_loss(model: nn.Module) -> torch.Tensor | None:
    losses = [
        module.branch_usage_loss
        for module in model.modules()
        if isinstance(module, NeuroTS_Block_v1) and module.branch_usage_loss is not None
    ]
    if not losses:
        return None
    return torch.stack(tuple(losses)).sum()

