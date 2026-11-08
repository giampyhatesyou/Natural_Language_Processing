"""Convert the logs of the Colab run (October 2026, before the JSON results existed) into results/<task>/*.json.

New trainings write the JSON files themselves; this script was run once on results/logs. It also checks that the
hyperparameters printed in the logs are the ones of the current experiment registries.

    python scripts/logs_to_results.py
"""
import ast
import json
import re

from nlu_ablations import DATA_DIR, RESULTS_DIR
from nlu_ablations.lm.cli import experiment_config as lm_config
from nlu_ablations.slu.cli import experiment_config as slu_config
from nlu_ablations.slu.data import load_data

LOGS = RESULTS_DIR / "logs"
# keys that the first layout of the project named differently
RENAMED = {"batch_size_dev": "batch_size_eval", "batch_size_test": "batch_size_eval"}


def read_lines(name):
    # tqdm rewrites its line with carriage returns: split on both
    return (LOGS / name).read_text(encoding="utf-8").replace("\r", "\n").split("\n")


def seconds(elapsed):
    value = 0
    for part in elapsed.split(":"):
        value = 60 * value + int(part)
    return value


def checked_config(logged, current, name):
    """The current configuration, after checking it against the one printed in the log."""
    for key, value in logged.items():
        if key not in current:
            key = RENAMED.get(key, key)
        if key in current and current[key] != value:
            raise ValueError(f"{name}: {key} is {value} in the log and {current[key]} in the registry")
    return current


def record(task, name, config, runs, log):
    return {"task": task, "experiment": name, "date": "2026-10-31", "environment": None,
            "source": f"converted from results/logs/{log} by scripts/logs_to_results.py", "config": config,
            "runs": runs}


def convert_lm(log):
    records, name, config, dev_ppl, trigger, elapsed = {}, None, None, [], None, 0
    for line in read_lines(log):
        if start := re.match(r"=== Training '([^']+)': (\{.*\})", line):
            name, dev_ppl, trigger = start.group(1), [], None
            config = checked_config(ast.literal_eval(start.group(2)), lm_config(start.group(1)), name)
        elif progress := re.match(r"Dev PPL: ([0-9.]+):.*\| *\d+/\d+ \[([0-9:]+)<", line):
            if not dev_ppl or float(progress.group(1)) != dev_ppl[-1]:
                dev_ppl.append(float(progress.group(1)))
            elapsed = seconds(progress.group(2))
        elif triggered := re.match(r"NT-AvSGD triggered at epoch (\d+)", line):
            trigger = int(triggered.group(1))
        elif done := re.match(rf"'{name}': best dev PPL [0-9.]+ \| test PPL ([0-9.]+)", line):
            best = min(range(len(dev_ppl)), key=dev_ppl.__getitem__)
            run = {"seed": config["seed"], "epochs": len(dev_ppl), "best_epoch": best + 1, "dev_ppl": dev_ppl[best],
                   "test_ppl": float(done.group(1)), "trigger_epoch": trigger, "seconds": elapsed,
                   "history": {"epoch": list(range(1, len(dev_ppl) + 1)), "dev_ppl": dev_ppl}}
            records[name] = record("lm", name, config, [run], log)
    return records


def convert_slu(log, n_test):
    records, name, config, runs, elapsed = {}, None, None, [], None
    for line in read_lines(log):
        if start := re.match(r"=== (?:Training '([^']+)'|Fine-tuning [^:]+): (\{.*\})", line):
            if name is not None:
                records[name] = record("slu", name, config, runs, log)
            name, runs, elapsed = start.group(1) or "bert", [], None
            config = checked_config(ast.literal_eval(start.group(2)), slu_config(name), name)
        elif progress := re.search(r"\| *\d+/\d+ \[([0-9:]+)<", line):
            elapsed = seconds(progress.group(1))  # the progress bar of the current run
        elif run := re.match(r"run (\d+): epochs (\d+) \| dev slot F1 ([0-9.]+) \| test slot F1 ([0-9.]+) \| "
                             r"test intent acc ([0-9.]+)", line):
            # the log rounds to 4 decimals; an accuracy over n_test utterances is recovered exactly
            correct = round(float(run.group(5)) * n_test)
            runs.append({"seed": config["seed"] + int(run.group(1)), "epochs": int(run.group(2)), "best_epoch": None,
                         "dev_slot_f1": float(run.group(3)), "test_slot_f1": float(run.group(4)),
                         "test_intent_acc": correct / n_test,
                         "seconds": elapsed, "history": None})
            elapsed = None
    if name is not None:
        records[name] = record("slu", name, config, runs, log)
    return records


def main():
    n_test = len(load_data(DATA_DIR / "atis" / "test.json"))
    converted = {"lm": {**convert_lm("lm_part_a.log"), **convert_lm("lm_part_b.log")},
                 "slu": {**convert_slu("slu_part_a.log", n_test), **convert_slu("slu_part_b.log", n_test)}}
    for task, records in converted.items():
        (RESULTS_DIR / task).mkdir(parents=True, exist_ok=True)
        for name, content in records.items():
            (RESULTS_DIR / task / f"{name}.json").write_text(json.dumps(content, indent=1) + "\n")
            print(f"{task}/{name}.json: {len(content['runs'])} run(s)")


if __name__ == "__main__":
    main()
