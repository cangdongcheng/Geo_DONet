"""MLP AT/PCA experiment using the shared DeepONet-PCA training/evaluation."""
import sys
from pathlib import Path

# Use the same decoder, loss, sampling, and evaluator as the operator baseline.
# Output paths remain relative to the caller's working directory.
SHARED = Path(__file__).resolve().parents[1] / "Geo_DeepONet_PCA"
sys.path.insert(0, str(SHARED))
from main import main as run


if __name__ == "__main__":
    run(default_architecture="mlp", default_patience=0)
