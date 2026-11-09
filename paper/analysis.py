"""Figures and tables of the paper and of the README, computed from the JSON records in results/ only.

Usage: python paper/analysis.py
  - paper/figures/effects.pdf, lm_dev_ppl.pdf, and their light and dark PNG versions in assets/
  - paper/tables/lm_part_a.tex, lm_part_b.tex, slu_results.tex, effects.tex
  - prints the README tables in Markdown and the numbers quoted in the text
"""
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from scipy import stats

from nlp import RESULTS_DIR
from nlp.common import sample_std
from nlp.results import load_results

# DejaVu Sans ships with matplotlib, so the figures are the same on every machine; TrueType fonts in the PDFs
matplotlib.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
PAPER = Path(__file__).resolve().parent
ASSETS = PAPER.parent / "assets"
ALPHA = 0.05
LM_PART_A = ["lstm", "lstm_dropout", "lstm_dropout_adamw"]
LM_PART_B = ["wt", "wt_vd", "wt_vd_ntavsgd"]
SLU_MODELS = ["lstm", "bilstm", "bilstm_dropout", "bert"]
# Ablation steps compared with the original study. Weight tying is left out: in both studies its baseline is the
# LSTM of part A, trained with other hyperparameters, so its effect is not a controlled one.
LM_STEPS = [("lstm", "lstm_dropout", "+ dropout"), ("lstm_dropout", "lstm_dropout_adamw", "+ AdamW"),
            ("wt", "wt_vd", "+ variational dropout"), ("wt_vd", "wt_vd_ntavsgd", "+ NT-AvSGD")]
SLU_STEPS = [("lstm", "bilstm", "BiLSTM"), ("bilstm", "bilstm_dropout", "+ dropout"),
             ("bilstm_dropout", "bert", "BERT")]
SLU_METRICS = [("test_slot_f1", "slot F1"), ("test_intent_acc", "intent acc.")]

# Colours of the reference palette of the dataviz method, validated for both themes: the two shades of one hue of
# the dumbbell chart (ordinal check) and the first three categorical slots of the curves (all-pairs check)
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e", "axis": "#c3c2b7", "grid": "#e1e0d9",
              "original": "#86b6ef", "audited": "#1c5cab", "link": "#c3c2b7",
              "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "muted": "#c3c2b7", "axis": "#383835", "grid": "#2c2c2a",
             "original": "#1c5cab", "audited": "#6da7ec", "link": "#52514e",
             "series": ["#3987e5", "#d95926", "#199e70"]},
}
THEMES["paper"] = {**THEMES["light"], "surface": "#ffffff"}  # the PDF figures sit on the white page
LINE_STYLES = ["-", "--", "-."]  # second encoding of the series, for grayscale and colour-vision deficiencies


# ----------------------------------------------------------------------------------------------- effects of each step

def welch(after, before):
    """Mean difference after - before, its 95% confidence interval and the p-value of Welch's t-test."""
    after, before = np.asarray(after, dtype=float), np.asarray(before, dtype=float)
    diff = after.mean() - before.mean()
    if len(after) < 2 or len(before) < 2:
        return diff, None, None
    var_a, var_b = after.var(ddof=1) / len(after), before.var(ddof=1) / len(before)
    df = (var_a + var_b) ** 2 / (var_a ** 2 / (len(after) - 1) + var_b ** 2 / (len(before) - 1))
    half = stats.t.ppf(1 - ALPHA / 2, df) * math.sqrt(var_a + var_b)
    return diff, (diff - half, diff + half), stats.ttest_ind(after, before, equal_var=False).pvalue


def outcome(original, audited, p):
    """How the audit changed the conclusion of a step: the rule of Section 5 of the paper."""
    if p is not None and p >= ALPHA:
        return "not significant"
    if np.sign(original) != np.sign(audited):
        return "sign reversed"
    ratio = original / audited
    if ratio > 2:
        return f"{ratio:.1f}x overstated"
    if ratio < 0.5:
        return f"{1 / ratio:.0f}x understated"
    return "confirmed"


def effects(lm, slu, original):
    """One row per step and metric; positive = the step helps (lower perplexity, higher F1 or accuracy)."""
    rows = []
    for before, after, label in LM_STEPS:
        ppl = {name: [run["test_ppl"] for run in lm[name]["runs"]] for name in (before, after)}
        diff, ci, p = welch(ppl[before], ppl[after])  # perplexity reduction
        reported = original["lm"][before]["test_ppl"] - original["lm"][after]["test_ppl"]
        rows.append({"task": "lm", "step": label, "metric": "PPL", "original": reported, "audited": diff,
                     "ci": ci, "p": p, "outcome": outcome(reported, diff, p)})
    for before, after, label in SLU_STEPS:
        for key, metric in SLU_METRICS:
            scores = {name: [100 * run[key] for run in slu[name]["runs"]] for name in (before, after)}
            diff, ci, p = welch(scores[after], scores[before])
            reported = 100 * (original["slu"][after][key] - original["slu"][before][key])
            rows.append({"task": "slu", "step": label, "metric": metric, "original": reported, "audited": diff,
                         "ci": ci, "p": p, "outcome": outcome(reported, diff, p)})
    return rows


def style_axes(ax, theme):
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["axis"])
    ax.tick_params(colors=theme["muted"], length=0, labelsize=7.5)
    ax.xaxis.label.set_color(theme["ink"])
    ax.grid(True, axis="x", color=theme["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def effects_figure(rows, theme_name, path):
    """Dumbbell chart: for every step, the effect reported by the original study and the one after the audit."""
    theme = THEMES[theme_name]
    panels = [("lm", "(a) Language modelling: reduction of the test perplexity", "perplexity points"),
              ("slu", "(b) Intent classification and slot filling: gain on the test set",
               "points of slot F1 or of intent accuracy (%)")]
    sizes = [sum(row["task"] == task for row in rows) for task, _, _ in panels]
    fig = Figure(figsize=(7.0, 4.1), facecolor=theme["surface"])
    axes = fig.subplots(2, 1, gridspec_kw={"height_ratios": sizes, "hspace": 0.75})
    for ax, (task, title, xlabel) in zip(axes, panels, strict=True):
        style_axes(ax, theme)
        task_rows = [row for row in rows if row["task"] == task]
        ys = np.arange(len(task_rows))[::-1]
        ax.axvline(0, color=theme["axis"], linewidth=1.0, zorder=1)
        for y, row in zip(ys, task_rows, strict=True):
            ax.plot([row["original"], row["audited"]], [y, y], color=theme["link"], linewidth=1.6, zorder=2,
                    solid_capstyle="round")
            if row["ci"] is not None:
                ax.plot(row["ci"], [y, y], color=theme["audited"], linewidth=1.0, zorder=3)
                for end in row["ci"]:
                    ax.plot([end, end], [y - 0.14, y + 0.14], color=theme["audited"], linewidth=1.0, zorder=3)
            ax.scatter(row["original"], y, s=34, facecolor=theme["surface"], edgecolor=theme["original"],
                       linewidth=1.6, zorder=4)
            ax.scatter(row["audited"], y, s=34, color=theme["audited"], edgecolor=theme["surface"], linewidth=1.0,
                       zorder=5)
            ax.annotate(row["outcome"].replace("x ", "\u00d7 "), xy=(1.02, y), xycoords=("axes fraction", "data"),
                        fontsize=7.5, color=theme["muted"], va="center", ha="left", annotation_clip=False)
        labels = [row["step"] if task == "lm" else f"{row['step']}, {row['metric']}" for row in task_rows]
        ax.set_yticks(ys, labels, fontsize=7.5, color=theme["ink"])
        ax.set_ylim(-0.6, len(task_rows) - 0.4)
        ax.set_xlabel(xlabel, fontsize=7.5)
        ax.set_title(title, fontsize=8, color=theme["ink"], loc="left", pad=5)
    handles = [axes[0].scatter([], [], s=34, facecolor=theme["surface"], edgecolor=theme["original"], linewidth=1.6),
               axes[0].scatter([], [], s=34, color=theme["audited"])]
    legend = fig.legend(handles, ["reported by the original implementation",
                                  "after the audit (bars: 95% CI over 5 seeds)"],
                        loc="upper center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.55, 1.0))
    for text in legend.get_texts():
        text.set_color(theme["ink"])
    fig.subplots_adjust(left=0.2, right=0.84, top=0.88, bottom=0.1)
    save(fig, path, theme)


# ---------------------------------------------------------------------------------------------- language modelling

def lm_label(name, config):
    return {"lstm": "LSTM", "lstm_dropout": f"+ dropout ({config['emb_dropout']:g})", "lstm_dropout_adamw": "+ AdamW",
            "wt": "LSTM + weight tying", "wt_vd": f"+ variational dropout ({config['emb_dropout']:g})",
            "wt_vd_ntavsgd": f"+ NT-AvSGD ($n={config.get('non_mono')}$)"}[name]


def selected_run(record, key, lower_is_better):
    """The run kept as checkpoint: the best one on the dev set."""
    return (min if lower_is_better else max)(record["runs"], key=lambda run: run[key])


def lm_figure(results, theme_name, path, parts=("A", "B")):
    """Dev perplexity per epoch of the selected runs, one panel per part (the paper shows part B only)."""
    theme = THEMES[theme_name]
    panels = {"A": ("(a) Part A", LM_PART_A), "B": ("(b) Part B" if len(parts) > 1 else "Part B", LM_PART_B)}
    fig = Figure(figsize=(3.5 * len(parts), 2.35), facecolor=theme["surface"])
    axes = np.atleast_1d(fig.subplots(1, len(parts), sharey=True))
    for ax, (title, names) in zip(axes, [panels[part] for part in parts], strict=True):
        style_axes(ax, theme)
        ax.grid(True, axis="y", color=theme["grid"], linewidth=0.6)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(theme["axis"])
        for name, color, style in zip(names, theme["series"], LINE_STYLES, strict=True):
            run = selected_run(results[name], "dev_ppl", lower_is_better=True)
            ppl = run["history"]["dev_ppl"]
            label = lm_label(name, results[name]["config"]).replace("$", "")
            ax.plot(range(1, len(ppl) + 1), ppl, color=color, linestyle=style, linewidth=1.4, label=label)
            best = run["best_epoch"]
            # The selected checkpoint: best dev epoch (averaged weights after the NT-AvSGD trigger)
            ax.scatter(best, ppl[best - 1], s=26, color=color, edgecolor=theme["surface"], linewidth=1.0, zorder=5)
            if best > 90:  # close to the right edge: label above, on the left
                offset, ha, va = (-5, 6), "right", "bottom"
            elif len(ppl) < 20:  # short run, the curve below is close: label on the right
                offset, ha, va = (6, -1), "left", "top"
            else:
                offset, ha, va = (0, -9), "center", "top"
            ax.annotate(f"{ppl[best - 1]:.1f}", (best, ppl[best - 1]), xytext=offset, textcoords="offset points",
                        ha=ha, va=va, fontsize=7, color=theme["ink"])
            if run["trigger_epoch"]:
                trigger = run["trigger_epoch"]
                ax.axvline(trigger, color=theme["muted"], linewidth=0.8, zorder=1)
                ax.text(trigger - 1.5, 128, f"NT-AvSGD trigger\n(epoch {trigger})", ha="right", va="center",
                        fontsize=7, color=theme["muted"])
        ax.set_title(title, fontsize=8, color=theme["ink"], loc="left")
        ax.set_xlabel("epoch", fontsize=7.5)
        ax.set_xlim(0, 101)
        ax.set_ylim(85, 190)
        legend = ax.legend(frameon=False, fontsize=7, loc="upper right", handlelength=2.6)
        for text in legend.get_texts():
            text.set_color(theme["ink"])
    axes[0].set_ylabel("dev perplexity", fontsize=7.5, color=theme["ink"])
    fig.tight_layout(w_pad=1.5)
    save(fig, path, theme)


def save(fig, path, theme):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".pdf":
        fig.savefig(path, metadata={"CreationDate": None})  # same input, same file
    else:
        fig.savefig(path, dpi=220, facecolor=theme["surface"])


def spread(values, digits):
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} $\\pm$ {sample_std(values):.{digits}f}"


def optimiser(config):
    lr = config["lr"]
    exponent = math.log10(lr)
    lr_text = f"$10^{{{int(exponent)}}}$" if lr < 0.01 and exponent.is_integer() else f"{lr:g}"
    return ("AdamW" if config.get("optimizer") == "adamw" else "SGD") + f", lr {lr_text}"


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


def ntavsgd_decomposition(results):
    """Where the dev gain of NT-AvSGD over variational dropout comes from: longer SGD training or averaging."""
    without = selected_run(results["wt_vd"], "dev_ppl", lower_is_better=True)
    run = selected_run(results["wt_vd_ntavsgd"], "dev_ppl", lower_is_better=True)
    if not run["trigger_epoch"]:
        return None  # averaging never started (e.g. a short test run): nothing to decompose
    before_trigger = run["history"]["dev_ppl"][:run["trigger_epoch"] - 1]
    sgd_best = min(before_trigger)
    return {"vd_dev": without["dev_ppl"], "vd_stop_epoch": without["epochs"], "sgd_best_dev": sgd_best,
            "sgd_best_epoch": before_trigger.index(sgd_best) + 1, "trigger_epoch": run["trigger_epoch"],
            "final_dev": run["dev_ppl"], "from_longer_training": without["dev_ppl"] - sgd_best,
            "from_averaging": sgd_best - run["dev_ppl"]}


# ----------------------------------------------------------------------------------------- intent and slot filling

def slu_label(name, markdown=False):
    if name == "bert":
        return "BERT (`bert-base-uncased`)" if markdown else "BERT"
    return {"lstm": "LSTM", "bilstm": "BiLSTM", "bilstm_dropout": "BiLSTM + dropout"}[name]


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
            md_cells += [f"**{text}**" if mean == best[metric] else text, format_p(p, markdown=True)]
        latex.append(f"  {slu_label(name)} & {tex_cells[0]} & {tex_cells[1]} \\\\")
        markdown.append(f"| {slu_label(name, markdown=True)} | " + " | ".join(md_cells) + " |")
    latex += ["  \\bottomrule", "\\end{tabular}", ""]
    write_table("slu_results.tex", latex)
    return markdown


def selection_effect(results):
    """Rank on the test set of the BERT run that was selected on dev (1 = best of the five)."""
    runs = results["bert"]["runs"]
    selected = selected_run(results["bert"], "dev_slot_f1", lower_is_better=False)
    ranking = sorted(runs, key=lambda run: run["test_slot_f1"], reverse=True)
    return {"selected_test_f1": 100 * selected["test_slot_f1"], "rank": ranking.index(selected) + 1,
            "runs": len(runs), "mean_test_f1": 100 * np.mean([run["test_slot_f1"] for run in runs])}


# ------------------------------------------------------------------------------------------------- effects table

def format_p(p, markdown=False):
    if p is None:
        return "–" if markdown else "--"
    return ("< 0.001" if markdown else "$<$0.001") if p < 0.001 else f"{p:.3f}"


def effects_table(rows):
    latex = ["\\begin{tabular}{l l r r l}", "  \\toprule",
             "  Step & Metric & Original & Audited & Outcome \\\\", "  \\midrule"]
    markdown = ["| Step | Metric | Original | Audited | p | Outcome |", "| - | - | - | - | - | - |"]
    previous_task = None
    for row in rows:
        if previous_task is not None and row["task"] != previous_task:
            latex.append("  \\midrule")
        previous_task = row["task"]
        outcome_tex = row["outcome"].replace("x ", "$\\times$ ")
        audited = f"{row['audited']:+.2f}" if row["p"] is not None else f"{row['audited']:+.1f}"
        original = f"{row['original']:+.2f}" if row["task"] == "slu" else f"{row['original']:+.1f}"
        latex.append(f"  {row['step']} & {row['metric']} & {original} & {audited} & {outcome_tex} \\\\")
        markdown.append(f"| {row['step']} | {row['metric']} | {original} | {audited} | "
                        f"{format_p(row['p'], markdown=True)} | {row['outcome'].replace('x ', '× ')} |")
    latex += ["  \\bottomrule", "\\end{tabular}", ""]
    write_table("effects.tex", latex)
    return markdown


def write_table(name, lines):
    (PAPER / "tables").mkdir(exist_ok=True)
    text = "% Generated by paper/analysis.py from results/: do not edit by hand\n" + "\n".join(lines)
    (PAPER / "tables" / name).write_text(text)


if __name__ == "__main__":
    lm, slu = load_results("lm"), load_results("slu")
    original = json.loads((RESULTS_DIR / "original_study.json").read_text())
    rows = effects(lm, slu, original)
    effects_figure(rows, "paper", PAPER / "figures" / "effects.pdf")
    lm_figure(lm, "paper", PAPER / "figures" / "lm_dev_ppl.pdf", parts=("B",))
    for theme in ("light", "dark"):
        effects_figure(rows, theme, ASSETS / f"effects_{theme}.png")
        lm_figure(lm, theme, ASSETS / f"lm_dev_ppl_{theme}.png")

    print("\n".join(lm_tables(lm)) + "\n")
    print("\n".join(slu_table(slu)) + "\n")
    print("\n".join(effects_table(rows)) + "\n")
    counts = {}
    for row in rows:
        kind = row["outcome"].split(" ", 1)[-1] if "x " in row["outcome"] else row["outcome"]
        counts[kind] = counts.get(kind, 0) + 1
    print("Outcomes of the", len(rows), "steps:", counts)
    for row in rows:
        ci = "" if row["ci"] is None else f", 95% CI [{row['ci'][0]:+.2f}, {row['ci'][1]:+.2f}]"
        p = "" if row["p"] is None else f", p = {row['p']:.4f}"
        print(f"  {row['task']} {row['step']} ({row['metric']}): original {row['original']:+.2f}, "
              f"audited {row['audited']:+.2f}{ci}{p} -> {row['outcome']}")
    decomposition = ntavsgd_decomposition(lm)
    print("NT-AvSGD:", "never triggered" if decomposition is None
          else {key: round(value, 2) for key, value in decomposition.items()})
    print("BERT selected on dev:", {key: round(value, 2) for key, value in selection_effect(slu).items()})
