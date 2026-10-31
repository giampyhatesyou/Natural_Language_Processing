"""Import the modules of one part of the project.

Every part is self-contained and names its modules utils, model and functions (plus conll for the intent/slot
parts), so the modules of the previously loaded part are dropped from sys.modules before loading the next one.
The returned modules stay valid after that.
"""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load_part(relative_path):
    part_dir = str(ROOT / relative_path)
    for name in ("utils", "model", "functions", "conll"):
        sys.modules.pop(name, None)
    sys.path.insert(0, part_dir)
    try:
        return SimpleNamespace(**{name: importlib.import_module(name) for name in ("utils", "model", "functions")})
    finally:
        sys.path.remove(part_dir)
