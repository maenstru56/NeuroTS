from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_training_plots(metrics_csv: str | Path, output_dir: str | Path) -> None:
    table = pd.read_csv(metrics_csv)
    if table.empty:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    def _plot_lines(columns: list[str], filename: str, ylabel: str) -> None:
        present = [column for column in columns if column in table.columns]
        present = [column for column in present if not pd.to_numeric(table[column], errors="coerce").dropna().empty]
        if not present:
            return
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
        for column in present:
            ax.plot(table["epoch"], pd.to_numeric(table[column], errors="coerce"), label=column)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / filename)
        plt.close(fig)

    for columns, filename, ylabel in [
        (["train_loss", "val_loss"], "loss_curve.png", "loss"),
        (["train_mean_region_dice", "val_mean_region_dice"], "dice_curve.png", "mean region dice"),
        (["train_mean_region_iou", "val_mean_region_iou"], "iou_curve.png", "mean region IoU"),
        (["learning_rate"], "learning_rate.png", "learning rate"),
    ]:
        _plot_lines(columns, filename, ylabel)

    region_columns = [
        "val_dice_et",
        "val_dice_net",
        "val_dice_cc",
        "val_dice_ed",
        "val_dice_tc",
        "val_dice_wt",
    ]
    _plot_lines(region_columns, "validation_dice_by_region_curve.png", "validation Dice")

    heatmap_columns = [column for column in region_columns if column in table.columns]
    if heatmap_columns:
        try:
            import numpy as np

            matrix = np.vstack([pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float) for column in heatmap_columns])
            if np.isfinite(matrix).any():
                fig, ax = plt.subplots(figsize=(max(7.0, 0.25 * len(table)), 4.2), dpi=150)
                image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
                ax.set_yticks(range(len(heatmap_columns)))
                ax.set_yticklabels([column.replace("val_dice_", "").upper() for column in heatmap_columns])
                ax.set_xticks(range(len(table)))
                ax.set_xticklabels([str(int(epoch)) for epoch in table["epoch"]], rotation=45, ha="right", fontsize=8)
                ax.set_xlabel("epoch")
                ax.set_ylabel("target")
                ax.set_title("Validation Dice Evolution")
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                fig.savefig(root / "validation_dice_heatmap.png")
                plt.close(fig)
        except Exception:
            pass
