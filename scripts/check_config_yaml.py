from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurots_net.config import ExperimentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate NeuroTS-Net YAML configs, including duplicate-key rejection.")
    parser.add_argument("configs", nargs="+", help="Config YAML files to validate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for item in args.configs:
        path = Path(item)
        cfg = ExperimentConfig.from_yaml(path)
        print(
            f"OK {path} | experiment={cfg.output.experiment_name} | "
            f"label_mode={cfg.data.label_mode} | loss={cfg.loss.name}"
        )


if __name__ == "__main__":
    main()
