from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _candidate_patterns(candidate: str, pattern: str | None) -> list[str]:
    if pattern:
        return [pattern]
    key = str(candidate).strip().lower()
    if key in {"voxel", "voxel_dice", "best_voxel_dice"}:
        return ["best_voxel_dice.pt", "fold_*/best_voxel_dice.pt", "folds_all/checkpoints/fold_*_best_voxel_dice.pt"]
    if key in {"rare", "rare_present", "best_rare_present_score"}:
        return ["best_rare_present_score.pt", "fold_*/best_rare_present_score.pt", "folds_all/checkpoints/fold_*_best_rare_present_score.pt"]
    if key in {"mixed", "all"}:
        return [
            "best_voxel_dice.pt",
            "best_rare_present_score.pt",
            "fold_*/best_voxel_dice.pt",
            "fold_*/best_rare_present_score.pt",
            "folds_all/checkpoints/fold_*_best_voxel_dice.pt",
            "folds_all/checkpoints/fold_*_best_rare_present_score.pt",
        ]
    raise ValueError("candidate must be voxel_dice, rare_present, or mixed.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List NeuroTS-Net checkpoints and expose duplicate fold copies.")
    parser.add_argument("--root", required=True, type=str, help="Experiment output directory.")
    parser.add_argument("--candidate", default="voxel_dice", choices=["voxel_dice", "rare_present", "mixed"])
    parser.add_argument("--pattern", default=None, help="Optional glob pattern relative to --root.")
    parser.add_argument("--direct-only", action="store_true", help="Only list direct fold_*/best_*.pt checkpoints.")
    parser.add_argument("--hash", action="store_true", help="Compute SHA256 hashes for duplicate detection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Experiment root does not exist: {root}")
    paths: list[Path] = []
    for glob_pattern in _candidate_patterns(args.candidate, args.pattern):
        paths.extend(sorted(root.glob(glob_pattern)))
    if args.direct_only:
        paths = [path for path in paths if path.parent.name.startswith("fold_") and path.name.startswith("best_")]
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    if not unique:
        raise FileNotFoundError(f"No checkpoints found under {root} for candidate={args.candidate}.")

    print(f"root: {root}")
    print(f"candidate: {args.candidate}")
    print(f"checkpoints: {len(unique)}")
    hashes: dict[str, list[Path]] = {}
    for index, path in enumerate(unique, start=1):
        size_mb = path.stat().st_size / (1024.0 * 1024.0)
        digest = _sha256(path) if args.hash else ""
        if digest:
            hashes.setdefault(digest, []).append(path)
        suffix = f" | sha256={digest[:16]}" if digest else ""
        print(f"{index:02d} | {size_mb:8.1f} MiB | {path}{suffix}")
    if args.hash:
        duplicate_groups = [group for group in hashes.values() if len(group) > 1]
        if duplicate_groups:
            print("\nduplicate-content groups:")
            for group_index, group in enumerate(duplicate_groups, start=1):
                print(f"group {group_index}:")
                for path in group:
                    print(f"  {path}")


if __name__ == "__main__":
    main()
