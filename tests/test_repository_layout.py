from importlib import import_module
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATA_MODULES = ("__init__", "augmentations", "brats26", "dataset", "folds", "nifti_io")


def test_data_package_modules_are_available() -> None:
    for name in DATA_MODULES:
        module_name = "neurots_net.data" if name == "__init__" else f"neurots_net.data.{name}"
        module = import_module(module_name)
        assert Path(module.__file__).resolve() == ROOT / "src" / "neurots_net" / "data" / f"{name}.py"


def test_gitignore_keeps_data_sources_and_excludes_datasets(tmp_path: Path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is required to check repository ignore rules.")

    # Test the rules independently of which files are already tracked locally.
    subprocess.run(
        [git, "init", "--quiet", str(tmp_path)], check=True, capture_output=True, text=True
    )
    shutil.copyfile(ROOT / ".gitignore", tmp_path / ".gitignore")
    source_paths = [f"src/neurots_net/data/{name}.py" for name in DATA_MODULES]
    private_paths = [
        "data/patient/metadata.json",
        "data/patient/scan.nii.gz",
        "src/neurots_net/data/__pycache__/brats26.cpython-311.pyc",
    ]
    result = subprocess.run(
        [
            git,
            "-C",
            str(tmp_path),
            "-c",
            f"core.excludesFile={tmp_path / 'no-global-excludes'}",
            "check-ignore",
            "--no-index",
            "--stdin",
            "-z",
        ],
        input="\0".join(source_paths + private_paths) + "\0",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    ignored = set(filter(None, result.stdout.split("\0")))
    assert ignored == set(private_paths), (
        "Source modules must be publishable while patient data and bytecode stay ignored: "
        f"{sorted(ignored)}"
    )
