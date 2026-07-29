from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurots_net.config import LossConfig


def main_logits(prediction: torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]) -> torch.Tensor:
    if isinstance(prediction, tuple):
        return prediction[0]
    return prediction


class DiceCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        include_background: bool = False,
        batch_dice: bool = True,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        label_smoothing: float = 0.0,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.include_background = bool(include_background)
        self.batch_dice = bool(batch_dice)
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.label_smoothing = float(label_smoothing)
        self.smooth = float(smooth)

    def _dice_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target.clamp_min(0).long(), num_classes=self.num_classes)
        target_one_hot = target_one_hot.movedim(-1, 1).to(dtype=probabilities.dtype)
        if not self.include_background:
            probabilities = probabilities[:, 1:]
            target_one_hot = target_one_hot[:, 1:]
        if self.batch_dice:
            reduce_dims = (0, 2, 3, 4)
        else:
            reduce_dims = (2, 3, 4)
        intersection = torch.sum(probabilities * target_one_hot, dim=reduce_dims)
        denominator = torch.sum(probabilities + target_one_hot, dim=reduce_dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()

    def _single_output_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self._dice_loss(logits, target)
        ce = F.cross_entropy(logits, target.long(), label_smoothing=self.label_smoothing)
        return self.dice_weight * dice + self.ce_weight * ce

    def forward(self, prediction: torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]], target: torch.Tensor) -> torch.Tensor:
        if not isinstance(prediction, tuple):
            return self._single_output_loss(prediction, target)
        logits, aux_outputs = prediction
        losses = [self._single_output_loss(logits, target)]
        weights = [1.0]
        for idx, aux in enumerate(aux_outputs):
            if aux.shape[2:] != target.shape[1:]:
                aux = F.interpolate(aux, size=target.shape[1:], mode="trilinear", align_corners=False)
            losses.append(self._single_output_loss(aux, target))
            weights.append(0.5 ** (idx + 1))
        weight_tensor = torch.as_tensor(weights, dtype=logits.dtype, device=logits.device)
        weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-6)
        return torch.stack(losses).mul(weight_tensor).sum()


class RegionAwareDiceCELoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        ce_weight: float,
        label_dice_weight: float,
        region_weight: float,
        absent_rare_fp_weight: float,
        ce_class_weights: list[float] | None,
        label_dice_class_weights: list[float],
        region_loss_weights: dict[str, float],
        rare_tversky_alpha_fp: float,
        rare_tversky_beta_fn: float,
        et_tversky_alpha_fp: float | None = None,
        et_tversky_beta_fn: float | None = None,
        cc_tversky_alpha_fp: float | None = None,
        cc_tversky_beta_fn: float | None = None,
        ed_tversky_alpha_fp_main: float | None = None,
        ed_tversky_beta_fn_main: float | None = None,
        absent_rare_fp_classes: list[int] | None = None,
        absent_rare_fp_power: float = 2.0,
        deep_supervision_full_rare_outputs: int = 1,
        label_smoothing: float = 0.0,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        if int(num_classes) != 5:
            raise ValueError("RegionAwareDiceCELoss expects five classes: background, ET, NET, CC, ED.")
        self.num_classes = int(num_classes)
        self.ce_weight = float(ce_weight)
        self.label_dice_weight = float(label_dice_weight)
        self.region_weight = float(region_weight)
        self.absent_rare_fp_weight = float(absent_rare_fp_weight)
        self.ce_class_weights = None if ce_class_weights is None else [float(item) for item in ce_class_weights]
        self.label_dice_class_weights = [float(item) for item in label_dice_class_weights]
        if self.ce_class_weights is not None and len(self.ce_class_weights) != self.num_classes:
            raise ValueError(
                "loss.ce_class_weights must use all-class order [BG, ET, NET, CC, ED] "
                f"with length {self.num_classes}; got length {len(self.ce_class_weights)}."
            )
        if len(self.label_dice_class_weights) != self.num_classes - 1:
            raise ValueError(
                "loss.label_dice_class_weights must use foreground order [ET, NET, CC, ED] "
                f"with length {self.num_classes - 1}; got length {len(self.label_dice_class_weights)}."
            )
        self.region_loss_weights = {str(key).upper(): float(value) for key, value in region_loss_weights.items()}
        self.rare_tversky_alpha_fp = float(rare_tversky_alpha_fp)
        self.rare_tversky_beta_fn = float(rare_tversky_beta_fn)
        self.tversky_overrides = {
            "ET": (
                self.rare_tversky_alpha_fp if et_tversky_alpha_fp is None else float(et_tversky_alpha_fp),
                self.rare_tversky_beta_fn if et_tversky_beta_fn is None else float(et_tversky_beta_fn),
            ),
            "CC": (
                self.rare_tversky_alpha_fp if cc_tversky_alpha_fp is None else float(cc_tversky_alpha_fp),
                self.rare_tversky_beta_fn if cc_tversky_beta_fn is None else float(cc_tversky_beta_fn),
            ),
            "ED": (
                self.rare_tversky_alpha_fp if ed_tversky_alpha_fp_main is None else float(ed_tversky_alpha_fp_main),
                self.rare_tversky_beta_fn if ed_tversky_beta_fn_main is None else float(ed_tversky_beta_fn_main),
            ),
        }
        self.absent_rare_fp_classes = [int(item) for item in (absent_rare_fp_classes or [])]
        invalid_absent_classes = [item for item in self.absent_rare_fp_classes if item <= 0 or item >= self.num_classes]
        if invalid_absent_classes:
            raise ValueError(f"absent_rare_fp_classes must be foreground class IDs 1..{self.num_classes - 1}, got {invalid_absent_classes}.")
        self.absent_rare_fp_power = float(absent_rare_fp_power)
        self.deep_supervision_full_rare_outputs = int(deep_supervision_full_rare_outputs)
        self.label_smoothing = float(label_smoothing)
        self.smooth = float(smooth)
        self.last_components: dict[str, float] = {}

    @staticmethod
    def _region_target(target: torch.Tensor, labels: tuple[int, ...]) -> torch.Tensor:
        result = torch.zeros_like(target, dtype=torch.bool)
        for label in labels:
            result |= target.long() == int(label)
        return result.to(dtype=torch.float32)

    def _weighted_ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = None
        if self.ce_class_weights is not None:
            weight = torch.as_tensor(self.ce_class_weights, dtype=torch.float32, device=logits.device)
        return F.cross_entropy(logits.float(), target.long(), weight=weight, label_smoothing=self.label_smoothing)

    def _label_dice(self, probabilities: torch.Tensor, target: torch.Tensor, *, include_rare: bool) -> torch.Tensor:
        target_one_hot = F.one_hot(target.clamp_min(0).long(), num_classes=self.num_classes)
        target_one_hot = target_one_hot.movedim(-1, 1).to(dtype=probabilities.dtype)
        probs_fg = probabilities[:, 1:]
        target_fg = target_one_hot[:, 1:]
        reduce_dims = (2, 3, 4)
        intersection = torch.sum(probs_fg * target_fg, dim=reduce_dims)
        denominator = torch.sum(probs_fg + target_fg, dim=reduce_dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        weights = torch.as_tensor(self.label_dice_class_weights, dtype=probabilities.dtype, device=probabilities.device)
        if not include_rare:
            # Foreground order is ET, NET, CC, ED. Keep only NET for coarse auxiliary outputs.
            weights = weights * torch.as_tensor([0.0, 1.0, 0.0, 0.0], dtype=weights.dtype, device=weights.device)
        if float(weights.sum().detach().cpu()) <= 0.0:
            return probabilities.new_tensor(0.0)
        loss = (1.0 - dice) * weights.view(1, -1)
        return loss.sum() / (weights.sum() * max(1, int(probabilities.shape[0])))

    def _binary_dice(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        reduce_dims = (1, 2, 3)
        intersection = torch.sum(prediction * target, dim=reduce_dims)
        denominator = torch.sum(prediction + target, dim=reduce_dims)
        return 1.0 - ((2.0 * intersection + self.smooth) / (denominator + self.smooth)).mean()

    def _binary_tversky(self, prediction: torch.Tensor, target: torch.Tensor, alpha_fp: float, beta_fn: float) -> torch.Tensor:
        reduce_dims = (1, 2, 3)
        tp = torch.sum(prediction * target, dim=reduce_dims)
        fp = torch.sum(prediction * (1.0 - target), dim=reduce_dims)
        fn = torch.sum((1.0 - prediction) * target, dim=reduce_dims)
        score = (tp + self.smooth) / (tp + float(alpha_fp) * fp + float(beta_fn) * fn + self.smooth)
        return 1.0 - score.mean()

    def _region_loss(self, probabilities: torch.Tensor, target: torch.Tensor, *, include_rare: bool) -> torch.Tensor:
        regions: dict[str, tuple[int, ...]] = {
            "ED": (4,),
            "CC": (3,),
            "ET": (1,),
            "NET": (2,),
            "TC": (1, 2, 3),
            "WT": (1, 2, 3, 4),
        }
        rare_regions = {"ED", "CC", "ET"}
        losses: list[torch.Tensor] = []
        weights: list[float] = []
        for region_name, labels in regions.items():
            if not include_rare and region_name in rare_regions:
                continue
            weight = float(self.region_loss_weights.get(region_name, 0.0))
            if weight <= 0.0:
                continue
            prediction = torch.stack([probabilities[:, int(label)] for label in labels], dim=0).sum(dim=0)
            target_mask = self._region_target(target, labels).to(dtype=prediction.dtype, device=prediction.device)
            if region_name in rare_regions:
                alpha_fp, beta_fn = self.tversky_overrides.get(region_name, (self.rare_tversky_alpha_fp, self.rare_tversky_beta_fn))
                loss = self._binary_tversky(prediction, target_mask, alpha_fp, beta_fn)
            else:
                loss = self._binary_dice(prediction, target_mask)
            losses.append(loss * weight)
            weights.append(weight)
        if not losses:
            return probabilities.new_tensor(0.0)
        return torch.stack(losses).sum() / max(1e-6, float(sum(weights)))

    def _absent_rare_fp_loss(self, probabilities: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.absent_rare_fp_weight <= 0.0 or not self.absent_rare_fp_classes:
            return probabilities.new_tensor(0.0)
        terms: list[torch.Tensor] = []
        for class_id in self.absent_rare_fp_classes:
            class_present = (target.long() == int(class_id)).flatten(1).any(dim=1)
            absent_indices = torch.nonzero(~class_present, as_tuple=False).flatten()
            if absent_indices.numel() == 0:
                continue
            class_probabilities = probabilities.index_select(0, absent_indices)[:, int(class_id)]
            terms.append(torch.pow(class_probabilities.clamp_min(0.0), self.absent_rare_fp_power).mean())
        if not terms:
            return probabilities.new_tensor(0.0)
        return torch.stack(terms).mean()

    def _single_output_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        include_rare: bool,
        include_absent_fp: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        probabilities = torch.softmax(logits.float(), dim=1)
        ce = self._weighted_ce(logits, target)
        label_dice = self._label_dice(probabilities, target, include_rare=include_rare)
        region = self._region_loss(probabilities, target, include_rare=include_rare)
        absent_rare_fp = self._absent_rare_fp_loss(probabilities, target) if include_absent_fp else probabilities.new_tensor(0.0)
        total = (
            self.ce_weight * ce
            + self.label_dice_weight * label_dice
            + self.region_weight * region
            + self.absent_rare_fp_weight * absent_rare_fp
        )
        return total, {
            "loss_ce": ce.detach(),
            "loss_label_dice": label_dice.detach(),
            "loss_region": region.detach(),
            "loss_absent_rare_fp": absent_rare_fp.detach(),
        }

    def forward(self, prediction: torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]], target: torch.Tensor) -> torch.Tensor:
        if not isinstance(prediction, tuple):
            loss, components = self._single_output_loss(prediction, target, include_rare=True, include_absent_fp=True)
            self.last_components = {key: float(value.detach().cpu().item()) for key, value in components.items()}
            return loss
        logits, aux_outputs = prediction
        main_loss, components = self._single_output_loss(logits, target, include_rare=True, include_absent_fp=True)
        self.last_components = {key: float(value.detach().cpu().item()) for key, value in components.items()}
        losses = [main_loss]
        weights = [1.0]
        for idx, aux in enumerate(aux_outputs):
            if aux.shape[2:] != target.shape[1:]:
                aux = F.interpolate(aux, size=target.shape[1:], mode="trilinear", align_corners=False)
            include_rare = idx < max(0, self.deep_supervision_full_rare_outputs)
            aux_loss, _ = self._single_output_loss(aux, target, include_rare=include_rare, include_absent_fp=False)
            losses.append(aux_loss)
            weights.append(0.5 ** (idx + 1))
        weight_tensor = torch.as_tensor(weights, dtype=logits.dtype, device=logits.device)
        weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-6)
        return torch.stack(losses).mul(weight_tensor).sum()

def build_loss(cfg: LossConfig, num_classes: int) -> nn.Module:
    loss_name = str(getattr(cfg, "name", "dice_ce")).strip().lower()
    if loss_name in {"region_aware", "region_aware_dice_ce", "brats_peds_region_aware"}:
        return RegionAwareDiceCELoss(
            num_classes=num_classes,
            ce_weight=cfg.main_ce_weight,
            label_dice_weight=cfg.main_label_dice_weight,
            region_weight=cfg.main_region_weight,
            absent_rare_fp_weight=cfg.absent_rare_fp_weight,
            ce_class_weights=cfg.ce_class_weights,
            label_dice_class_weights=cfg.label_dice_class_weights,
            region_loss_weights=cfg.region_loss_weights,
            rare_tversky_alpha_fp=cfg.rare_tversky_alpha_fp,
            rare_tversky_beta_fn=cfg.rare_tversky_beta_fn,
            et_tversky_alpha_fp=cfg.et_tversky_alpha_fp,
            et_tversky_beta_fn=cfg.et_tversky_beta_fn,
            cc_tversky_alpha_fp=cfg.cc_tversky_alpha_fp,
            cc_tversky_beta_fn=cfg.cc_tversky_beta_fn,
            ed_tversky_alpha_fp_main=cfg.ed_tversky_alpha_fp_main,
            ed_tversky_beta_fn_main=cfg.ed_tversky_beta_fn_main,
            absent_rare_fp_classes=cfg.absent_rare_fp_classes,
            absent_rare_fp_power=cfg.absent_rare_fp_power,
            deep_supervision_full_rare_outputs=cfg.deep_supervision_full_rare_outputs,
            label_smoothing=cfg.label_smoothing,
        )
    if loss_name not in {"dice_ce", "dice_cross_entropy", "soft_dice_ce"}:
        raise ValueError(
            "Unsupported loss.name. Public NeuroTS-Net supports 'region_aware' "
            "and 'dice_ce'."
        )
    return DiceCrossEntropyLoss(
        num_classes=num_classes,
        include_background=cfg.include_background,
        batch_dice=cfg.batch_dice,
        dice_weight=cfg.dice_weight,
        ce_weight=cfg.ce_weight,
        label_smoothing=cfg.label_smoothing,
    )

