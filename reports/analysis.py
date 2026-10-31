"""Figures and tables of the reports, computed from the training logs in results/logs.

Usage: python reports/analysis.py
  - reports/figures/lm_dev_ppl.pdf, .png   dev perplexity per epoch of the language models
  - reports/tables/nlu_results.tex         ATIS results: mean, sample std and Welch's t-test over the runs
  - prints the same NLU table in Markdown, for the README
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "results", "logs")
FIGURES = os.path.join(ROOT, "reports", "figures")
TABLES = os.path.join(ROOT, "reports", "tables")
ALPHA = 0.05

# Same incremental step -> same colour and line style in both panels (style keeps them apart in grayscale)
STYLES = [("#2a78d6", "-"), ("#eb6834", "--"), ("#1baf7a", "-.")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LM_PANELS = [
    ("(a) Part A", "lm_part_A.log", [("lstm", "LSTM"), ("lstm_dropout", "+ dropout"),
                                     ("lstm_dropout_adamw", "+ AdamW")]),
    ("(b) Part B", "lm_part_B.log", [("wt", "LSTM + weight tying"), ("wt_vd", "+ variational dropout"),
                                     ("wt_vd_ntavsgd", "+ NT-AvSGD")]),
]
# (log, experiment, label) in the order of the incremental protocol
NLU_MODELS = [("nlu_part_A.log", "lstm", "LSTM"), ("nlu_part_A.log", "bilstm", "BiLSTM"),
              ("nlu_part_A.log", "bilstm_dropout", "BiLSTM + dropout"), ("nlu_part_B.log", "bert", "BERT")]


def read_lines(name):
    # tqdm rewrites its line with carriage returns: split on both
    with open(os.path.join(LOGS, name), encoding="utf-8") as f:
        return f.read().replace("\r", "\n").split("\n")


def parse_lm_log(name):
    """Dev PPL of every epoch (from the tqdm descriptions) and NT-AvSGD trigger epoch of each experiment."""
    runs, current = {}, None
    for line in read_lines(name):
        start = re.match(r"=== Training '([^']+)'", line)
        if start:
            current = start.group(1)
            runs[current] = {"dev_ppl": [], "trigger": None}
            continue
        if current is None:
            continue
        progress = re.match(r"Dev PPL: ([0-9.]+):", line)
        if progress:
            ppl = float(progress.group(1))
            if not runs[current]["dev_ppl"] or runs[current]["dev_ppl"][-1] != ppl:
                runs[current]["dev_ppl"].append(ppl)
        trigger = re.match(r"NT-AvSGD triggered at epoch (\d+)", line)
        if trigger:
            runs[current]["trigger"] = int(trigger.group(1))
    return runs


def parse_nlu_log(name):
    """Per-run test scores (in %) and the means of the summary line, which are computed on unrounded values."""
    runs, current = {}, None
    for line in read_lines(name):
        start = re.match(r"=== (?:Training '([^']+)'|Fine-tuning)", line)
        if start:
            current = start.group(1) or "bert"
            runs[current] = {"runs": [], "mean": None}
            continue
        run = re.match(r"run \d+: .*test slot F1 ([0-9.]+) \| test intent acc ([0-9.]+)", line)
        if run:
            runs[current]["runs"].append((100 * float(run.group(1)), 100 * float(run.group(2))))
        summary = re.search(r"over \d+ runs: slot F1 ([0-9.]+) \+- [0-9.]+ \| intent acc ([0-9.]+)", line)
        if summary:
            runs[current]["mean"] = (100 * float(summary.group(1)), 100 * float(summary.group(2)))
    return runs


def lm_figure():
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.35), sharey=True)
    for ax, (title, log, experiments) in zip(axes, LM_PANELS, strict=True):
        runs = parse_lm_log(log)
        for (name, label), (color, style) in zip(experiments, STYLES, strict=True):
            ppl = runs[name]["dev_ppl"]
            ax.plot(range(1, len(ppl) + 1), ppl, color=color, linestyle=style, linewidth=1.3, label=label,
                    solid_capstyle="round", dash_capstyle="round")
            best = min(range(len(ppl)), key=ppl.__getitem__)
            # The selected checkpoint: best dev epoch (averaged weights after the NT-AvSGD trigger)
            ax.plot(best + 1, ppl[best], "o", color=color, markersize=4.5, markeredgecolor="white",
                    markeredgewidth=0.8, zorder=5)
            if best + 1 > 90:  # close to the right edge: label above, on the left
                offset, ha, va = (-5, 6), "right", "bottom"
            elif len(ppl) < 20:  # short run, the curve below is close: label on the right
                offset, ha, va = (6, -1), "left", "top"
            else:
                offset, ha, va = (0, -9), "center", "top"
            ax.annotate(f"{ppl[best]:.1f}", (best + 1, ppl[best]), xytext=offset, textcoords="offset points",
                        ha=ha, va=va, fontsize=7, color=INK)
            if runs[name]["trigger"]:
                trigger = runs[name]["trigger"]
                ax.axvline(trigger, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
                ax.text(trigger - 1.5, 128, f"NT-AvSGD trigger\n(epoch {trigger})", ha="right", va="center",
                        fontsize=7, color=MUTED)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("epoch")
        ax.set_xlim(0, 101)
        ax.set_ylim(85, 190)
        ax.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=7, loc="upper right", handlelength=2.6, labelcolor=INK)
    axes[0].set_ylabel("dev perplexity")
    fig.tight_layout(w_pad=1.5)
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, "lm_dev_ppl.pdf"))
    fig.savefig(os.path.join(FIGURES, "lm_dev_ppl.png"), dpi=220)
    plt.close(fig)


def nlu_table():
    """Mean and sample std over the runs; * marks a significant gain over the previous row (Welch's t-test)."""
    logs = {name: parse_nlu_log(name) for name in {log for log, _, _ in NLU_MODELS}}
    rows, previous = [], None
    for log, key, label in NLU_MODELS:
        scores = np.array(logs[log][key]["runs"])
        means, cells = logs[log][key]["mean"], []
        for metric in range(2):
            p = None
            if previous is not None:
                p = stats.ttest_ind(scores[:, metric], previous[:, metric], equal_var=False).pvalue
            cells.append((means[metric], scores[:, metric].std(ddof=1), p))
        rows.append((label, len(scores), cells))
        previous = scores

    best = [max(row[2][m][0] for row in rows) for m in range(2)]
    latex = ["\\begin{tabular}{l c c}", "  \\toprule", "  Model & Slot F1 & Intent acc. (\\%) \\\\", "  \\midrule"]
    for i, (label, _, cells) in enumerate(rows):
        tex_cells = []
        for metric, (mean, std, p) in enumerate(cells):
            value = f"\\textbf{{{mean:.2f}}}" if mean == best[metric] else f"{mean:.2f}"
            better = i > 0 and mean > rows[i - 1][2][metric][0]
            star = "$^{*}$" if p is not None and p < ALPHA and better else ""
            tex_cells.append(f"{value} $\\pm$ {std:.2f}{star}")
        latex.append(f"  {label} & {tex_cells[0]} & {tex_cells[1]} \\\\")
    latex += ["  \\bottomrule", "\\end{tabular}", ""]
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, "nlu_results.tex"), "w") as f:
        f.write("% Generated by reports/analysis.py from results/logs: do not edit by hand\n" + "\n".join(latex))

    print("| Model | Slot F1 | p | Intent accuracy | p |")
    print("| - | - | - | - | - |")
    for label, _, cells in rows:
        label = "BERT (`bert-base-uncased`)" if label == "BERT" else label
        fmt = [f"{mean:.2f} ± {std:.2f}" for mean, std, _ in cells]
        ps = ["–" if p is None else ("< 0.001" if p < 0.001 else f"{p:.3f}") for _, _, p in cells]
        print(f"| {label} | {fmt[0]} | {ps[0]} | {fmt[1]} | {ps[1]} |")


if __name__ == "__main__":
    lm_figure()
    nlu_table()
