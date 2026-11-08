"""Incremental ablation studies on language modelling (Penn Treebank) and on joint intent classification and slot
filling (ATIS)."""
from pathlib import Path

__version__ = "2.0.0"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHECKPOINTS_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
