"""Dev perplexity per epoch of the language models, rebuilt from the Colab training logs.

Usage: python reports/make_figures.py   (writes reports/figures/lm_dev_ppl.pdf and .png)
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "results", "logs")
OUT = os.path.join(ROOT, "reports", "figures")

# Same incremental step -> same colour and line style in both panels (style keeps them apart in grayscale)
STYLES = [("#2a78d6", "-"), ("#eb6834", "--"), ("#1baf7a", "-.")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PANELS = [
    ("(a) Part A", "lm_part_A.log", [("lstm", "LSTM"), ("lstm_dropout", "+ dropout"), ("lstm_dropout_adamw", "+ AdamW")]),
    ("(b) Part B", "lm_part_B.log", [("wt", "LSTM + weight tying"), ("wt_vd", "+ variational dropout"),
                                     ("wt_vd_ntavsgd", "+ NT-AvSGD")]),
]


def parse_log(path):
    """Dev PPL of every epoch (from the tqdm descriptions) and NT-AvSGD trigger epoch of each experiment."""
    text = open(path, encoding="utf-8").read().replace("\r", "\n")
    runs, current = {}, None
    for line in text.split("\n"):
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


def main():
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.35), sharey=True)
    for ax, (title, log, experiments) in zip(axes, PANELS):
        runs = parse_log(os.path.join(LOGS, log))
        for (name, label), (color, style) in zip(experiments, STYLES):
            ppl = runs[name]["dev_ppl"]
            epochs = range(1, len(ppl) + 1)
            ax.plot(epochs, ppl, color=color, linestyle=style, linewidth=1.3, label=label,
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
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "lm_dev_ppl.pdf"))
    fig.savefig(os.path.join(OUT, "lm_dev_ppl.png"), dpi=220)


if __name__ == "__main__":
    main()
