"""Run smoke tests in a disposable source copy, preserving checked-in figures."""
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
with TemporaryDirectory(prefix="pytorch-smoke-copy-") as directory:
    copy = Path(directory)
    shutil.copytree(ROOT / "examples", copy / "examples", ignore=shutil.ignore_patterns("__pycache__"))
    (copy / "tests").mkdir()
    shutil.copyfile(ROOT / "tests/smoke_examples.py", copy / "tests/smoke_examples.py")
    (copy / "docs/assets/images").mkdir(parents=True)
    subprocess.run([sys.executable, "-B", str(copy / "tests/smoke_examples.py")], cwd=copy, check=True)
print("Smoke tests completed in disposable copy; retained figures preserved")
