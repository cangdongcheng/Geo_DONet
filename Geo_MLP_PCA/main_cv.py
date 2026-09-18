"""MLP five-fold CV with the shared fold-specific train-only PCA decoder."""
import sys
from pathlib import Path

SHARED = Path(__file__).resolve().parents[1] / "Geo_DeepONet_PCA"
sys.path.insert(0, str(SHARED))
from main_cv import main as run


if __name__ == "__main__":
    run(default_architecture="mlp")
