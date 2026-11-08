"""Results of every experiment as a JSON file: results/<task>/<experiment>.json.

A record holds the configuration and one entry per run (seed, selected epoch, dev and test scores, per-epoch
history); tables and figures of the reports are computed from these files only.
"""
import json
import platform
from datetime import datetime, timezone

import torch

from . import RESULTS_DIR


def environment():
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"python": platform.python_version(), "torch": torch.__version__, "device": device}


def save_results(task, experiment, config, runs, results_dir=RESULTS_DIR):
    record = {"task": task, "experiment": experiment, "date": datetime.now(timezone.utc).date().isoformat(),
              "environment": environment(), "config": config, "runs": runs}
    path = results_dir / task / f"{experiment}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n")
    return path


def load_results(task, results_dir=RESULTS_DIR):
    return {path.stem: json.loads(path.read_text()) for path in sorted((results_dir / task).glob("*.json"))}
