"""Figure and tables of the reports, computed from the JSON results in results/.

Usage: python reports/analysis.py
  - reports/figures/lm_dev_ppl.pdf, .png    dev perplexity per epoch of the language models
  - reports/tables/lm_part_a.tex, lm_part_b.tex, slu_results.tex
  - prints the same tables in Markdown, for the README
"""
import math
from pathlib import Path

import matplotlib
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nlu_ablations.common import sample_std  # noqa: E402
from nlu_ablations.results import load_results  # noqa: E402

REPORTS = Path(__file__).resolve().parent
ALPHA = 0.05
LM_PART_A = ["lstm", "lstm_dropout", "lstm_dropout_adamw"]
LM_PART_B = ["wt", "wt_vd", "wt_vd_ntavsgd"]
SLU_MODELS = ["lstm", "bilstm", "bilstm_dropout", "bert"]

# Same incremental step -> same colour and line style in both panels (style keeps them apart in grayscale)
STYLES = [("#2a78d6", "-"), ("#eb6834", "--"), ("#1baf7a", "-.")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def lm_label(name, config):
    return {"lstm": "LSTM", "lstm_dropout": f"+ dropout ({config['emb_dropout']:g})", "lstm_dropout_adamw": "+ AdamW",
            "wt": "LSTM + weight tying", "wt_vd": f"+ variational dropout ({config['emb_dropout']:g})",
            "wt_vd_ntavsgd": f"+ NT-AvSGD ($n={config.get('non_mono')}$)"}[name]


def slu_label(name, markdown=False):
    if name == "bert":
        return "BERT (`bert-base-uncased`)" if markdown else "BERT"
    return {"lstm": "LSTM", "bilstm": "BiLSTM", "bilstm_dropout": "BiLSTM + dropout"}[name]


def selected_run(record, key, lower_is_better):
    """The run kept as checkpoint: the best one on the dev set."""
    return (min if lower_is_better else max)(record["runs"], key=lambda run: run[key])


def spread(values, digits):
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} $\\pm$ {sample_std(values):.{digits}f}"


def optimiser(config):
    lr = config["lr"]
    exponent = math.log10(lr)
    lr_text = f"$10^{{{int(exponent)}}}$" if lr < 0.01 and exponent.is_integer() else f"{lr:g}"
    return ("AdamW" if config.get("optimizer") == "adamw" else "SGD") + f", lr {lr_text}"


def lm_figure(results):
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.35), sharey=True)
    for ax, title, names in zip(axes, ("(a) Part A", "(b) Part B"), (LM_PART_A, LM_PART_B), strict=True):
        for name, (color, style) in zip(names, STYLES, strict=True):
            run = selected_run(results[name], "dev_ppl", lower_is_better=True)
            ppl = run["history"]["dev_ppl"]
            label = lm_label(name, results[name]["config"]).replace("$", "")
            ax.plot(range(1, len(ppl) + 1), ppl, color=color, linestyle=style, linewidth=1.3, label=label,
                    solid_capstyle="round", dash_capstyle="round")
            best = run["best_epoch"]
            # The selected checkpoint: best dev epoch (averaged weights after the NT-AvSGD trigger)
            ax.plot(best, ppl[best - 1], "o", color=color, markersize=4.5, markeredgecolor="white",
                    markeredgewidth=0.8, zorder=5)
            if best > 90:  # close to the right edge: label above, on the left
                offset, ha, va = (-5, 6), "right", "bottom"
            elif len(ppl) < 20:  # short run, the curve below is close: label on the right
                offset, ha, va = (6, -1), "left", "top"
            else:
                offset, ha, va = (0, -9), "center", "top"
            ax.annotate(f"{ppl[best - 1]:.1f}", (best, ppl[best - 1]), xytext=offset, textcoords="offset points",
                        ha=ha, va=va, fontsize=7, color=INK)
            if run["trigger_epoch"]:
                trigger = run["trigger_epoch"]
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
    (REPORTS / "figures").mkdir(exist_ok=True)
    fig.savefig(REPORTS / "figures" / "lm_dev_ppl.pdf", metadata={"CreationDate": None})  # same input, same file
    fig.savefig(REPORTS / "figures" / "lm_dev_ppl.png", dpi=220)
    plt.close(fig)


def lm_tables(results):
    markdown = ["| Part | Model | Dev PPL | Test PPL |", "| - | - | - | - |"]
    for part, names, with_optimiser in (("A", LM_PART_A, True), ("B", LM_PART_B, False)):
        test_means = [np.mean([run["test_ppl"] for run in results[name]["runs"]]) for name in names]
        columns = "l l c c c" if with_optimiser else "l c c c"
        header = "Model & Optimiser & Ep. & Dev & Test" if with_optimiser else "Model & Ep. & Dev & Test"
        latex = [f"\\begin{{tabular}}{{{columns}}}", "  \\toprule", f"  {header} \\\\", "  \\midrule"]
        for name, test_mean in zip(names, test_means, strict=True):
            runs, config = results[name]["runs"], results[name]["config"]
            epoch = round(np.mean([run["best_epoch"] for run in runs]))
            dev, test = spread([r["dev_ppl"] for r in runs], 1), spread([r["test_ppl"] for r in runs], 1)
            is_best = test_mean == min(test_means)
            cells = [lm_label(name, config)] + ([optimiser(config)] if with_optimiser else [])
            cells += [str(epoch), dev, f"\\textbf{{{test}}}" if is_best else test]
            latex.append("  " + " & ".join(cells) + " \\\\")
            md_dev, md_test = dev.replace("$\\pm$", "±"), test.replace("$\\pm$", "±")
            label = lm_label(name, config).replace("$n=", "n = ").replace("$", "")
            markdown.append(f"| {part} | {label} | {md_dev} | {f'**{md_test}**' if is_best else md_test} |")
        latex += ["  \\bottomrule", "\\end{tabular}", ""]
        write_table(f"lm_part_{part.lower()}.tex", latex)
    return markdown


def slu_table(results):
    """Mean and sample std over the runs; * marks a significant gain over the previous row (Welch's t-test)."""
    rows, previous = [], None
    for name in SLU_MODELS:
        scores = np.array([[100 * run["test_slot_f1"], 100 * run["test_intent_acc"]] for run in results[name]["runs"]])
        cells = []
        for metric in range(2):
            p = None if previous is None else stats.ttest_ind(scores[:, metric], previous[:, metric],
                                                              equal_var=False).pvalue
            better = previous is not None and scores[:, metric].mean() > previous[:, metric].mean()
            cells.append((scores[:, metric].mean(), sample_std(scores[:, metric]), p, better))
        rows.append((name, cells))
        previous = scores

    best = [max(cells[metric][0] for _, cells in rows) for metric in range(2)]
    latex = ["\\begin{tabular}{l c c}", "  \\toprule", "  Model & Slot F1 & Intent acc. (\\%) \\\\", "  \\midrule"]
    markdown = ["| Model | Slot F1 | p | Intent accuracy (%) | p |", "| - | - | - | - | - |"]
    for name, cells in rows:
        tex_cells, md_cells = [], []
        for metric, (mean, std, p, better) in enumerate(cells):
            value = f"\\textbf{{{mean:.2f}}}" if mean == best[metric] else f"{mean:.2f}"
            star = "$^{*}$" if p is not None and p < ALPHA and better else ""
            tex_cells.append(f"{value} $\\pm$ {std:.2f}{star}")
            text = f"{mean:.2f} ± {std:.2f}"
            md_cells += [f"**{text}**" if mean == best[metric] else text,
                         "–" if p is None else ("< 0.001" if p < 0.001 else f"{p:.3f}")]
        latex.append(f"  {slu_label(name)} & {tex_cells[0]} & {tex_cells[1]} \\\\")
        markdown.append(f"| {slu_label(name, markdown=True)} | " + " | ".join(md_cells) + " |")
    latex += ["  \\bottomrule", "\\end{tabular}", ""]
    write_table("slu_results.tex", latex)
    return markdown


def write_table(name, lines):
    (REPORTS / "tables").mkdir(exist_ok=True)
    text = "% Generated by reports/analysis.py from results/: do not edit by hand\n" + "\n".join(lines)
    (REPORTS / "tables" / name).write_text(text)


if __name__ == "__main__":
    lm, slu = load_results("lm"), load_results("slu")
    lm_figure(lm)
    print("\n".join(lm_tables(lm)) + "\n")
    print("\n".join(slu_table(slu)))
