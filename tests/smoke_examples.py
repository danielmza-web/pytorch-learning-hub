"""Fast deterministic checks for the four foundation project implementations."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.emnist_model import smoke_test as emnist_smoke_test
from examples.nature_cnn import smoke_test as nature_smoke_test
from examples.regression_demo import run as regression_run
from examples.robust_dataset import smoke_test as dataset_smoke_test


def main() -> None:
    results = {
        "regression": regression_run(),
        "emnist": emnist_smoke_test(),
        "robust_dataset": dataset_smoke_test(),
        "nature_cnn": nature_smoke_test(),
    }
    for name, result in results.items():
        print(f"{name}: {result}")


if __name__ == "__main__":
    main()
