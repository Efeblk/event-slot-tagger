"""Draws the README figures from the saved results (no model needed).

Reads results/evaluation.json, results/classic.json, results/bert-base-turkish-cased*_seed*.json,
results/timing_idle_gpu.json, results/predictions/, and (when present) results/robustness.json and
results/learning_curve.json; writes results/figures/*.png.

Every figure follows the same rules: a bold title with a one-line subtitle that says what to look for, direct labels
instead of legends where possible, 95% intervals on every estimate, and the same three colours for HMM, CRF and
BERTurk wherever those models are compared.

Usage: python plot_results.py
"""
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from common import spans

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, in fixed order
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential
WIDTH = 9  # inches; every figure has the same width so text has the same size in the README
THREE = [("hmm", "HMM"), ("crf", "CRF"), ("berturk_crf", "BERTurk + CRF layer")]  # one per categorical slot
LABELS = ["PLACE", "DATE", "TYPE", "NAME", "PRICE", "PEOPLE", "EXCLUDE"]
METHODS = [  # (key in evaluation.json, shown name, BERTurk result files and decoder, or None for classic models)
    ("frequent_tag", "Frequent tag", None),
    ("hmm", "HMM", None),
    ("hmm_bio", "HMM + BIO rules", None),
    ("crf_no_suffix", "CRF, no suffix features", None),
    ("crf", "CRF", None),
    ("berturk_softmax", "BERTurk, softmax", ("", "softmax")),
    ("berturk_softmax_bio", "BERTurk, softmax + BIO rules", ("", "softmax_bio")),
    ("berturk_crf", "BERTurk + CRF layer", ("_crf", "crf")),
]
SEEDS = (7, 13, 42)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2, "axes.labelsize": 9.5, "axes.edgecolor": MUTED,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 9,
})


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def bert_runs(suffix):
    return [load(f"bert-base-turkish-cased{suffix}_seed{s}.json") for s in SEEDS]


def dev_f1():
    """Dev F1 as in the README: the classic model, or the BERTurk mean over seeds."""
    classic = load("classic.json")["models"]
    out = {}
    for key, _, bert in METHODS:
        if bert is None:
            out[key] = classic[key]["dev"]["f1"]
        else:
            out[key] = statistics.mean(r["decoders"][bert[1]]["dev"]["f1"] for r in bert_runs(bert[0]))
    return out


def header(ax, title, subtitle, legend=False, panel_titles=False):
    """Title and subtitle above the plot, left-aligned with it; legend=True leaves a row for a legend and
    panel_titles=True one for the titles of small multiples."""
    top = 6 + (20 if legend else 0) + (20 if panel_titles else 0)
    ax.annotate(subtitle, (0, 1), xycoords="axes fraction", xytext=(0, top), textcoords="offset points",
                fontsize=9.5, color=INK_2, va="bottom")
    ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(0, top + 16), textcoords="offset points",
                fontsize=13, fontweight="bold", color=INK, va="bottom")


def legend_row(ax, ncol, **kwargs):
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, borderaxespad=0.1, handletextpad=0.4,
              columnspacing=1.6, **kwargs)


def note(ax, text, below=40):
    """A muted footnote under the plot."""
    ax.annotate(text, (0, 0), xycoords="axes fraction", xytext=(0, -below), textcoords="offset points",
                fontsize=8, color=MUTED, va="top")


def clean_y(ax):
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)


def interval_dot(ax, x, low, high, y, color, size=46, **kwargs):
    ax.hlines(y, low, high, color=color, linewidth=2.2, zorder=2, capstyle="round")
    ax.scatter(x, y, s=size, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3, **kwargs)


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote results/figures/{name}")


def plot_dev_vs_test(evaluation):
    """Dumbbell: dev F1 (same writer as train) against Codex test F1 (another writer), with the 95% interval."""
    dev = dev_f1()
    fig, ax = plt.subplots(figsize=(WIDTH, 4.6))
    rows = list(reversed(METHODS))
    for y, (key, _, bert) in enumerate(rows):
        test = evaluation["seeds" if bert else "models"][key]["codex_test"]
        f1 = test["mean"] if bert else test["f1"]
        ax.plot([f1, dev[key]], [y, y], color=GRID, linewidth=5, zorder=1, solid_capstyle="round")
        interval_dot(ax, f1, test["low"], test["high"], y, SERIES[0], size=60)
        ax.scatter(dev[key], y, s=60, color=SERIES[1], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(test["low"] - 0.006, y, f"{f1:.3f}", ha="right", va="center", fontsize=8.5, color=INK)
        ax.text(dev[key] + 0.008, y, f"−{dev[key] - f1:.3f}", ha="left", va="center", fontsize=8, color=MUTED)
        if y == len(rows) - 1:  # direct labels on the top row instead of a legend
            ax.annotate("Codex test", (f1, y), xytext=(0, 11), textcoords="offset points", ha="center",
                        fontsize=8.5, color=SERIES[0], fontweight="bold")
            ax.annotate("dev", (dev[key], y), xytext=(0, 11), textcoords="offset points", ha="center",
                        fontsize=8.5, color=SERIES[1], fontweight="bold")
    for boundary in (2.5, 4.5):  # BERTurk | CRF | count-based models
        ax.axhline(boundary, color=GRID, linewidth=0.8, zorder=0)
    ax.set_yticks(range(len(rows)), [name for _, name, _ in rows])
    clean_y(ax)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlim(0.55, 1.03)
    ax.set_xlabel("Exact-span micro F1")
    header(ax, "Every model drops on text from a writer it never saw",
           "Dev set (same AI writer as training) vs Codex test (another AI), with the 95% interval and the drop")
    note(ax, "BERTurk: mean of 3 seeds and the interval of that mean. 1,000 bootstrap draws over requests.")
    save(fig, "dev_vs_test.png")


def plot_per_label(evaluation):
    """Dot plot: F1 per label on the Codex test for HMM, CRF and BERTurk + CRF layer, with 95% intervals."""
    gold = [r["tags"] for r in codex_rows()]
    support = {label: sum(s[0] == label for tags in gold for s in spans(tags)) for label in LABELS}
    results = {key: evaluation["models"][key]["codex_test"] for key, _ in THREE}
    order = sorted(LABELS, key=lambda lab: results["berturk_crf"]["labels"][lab] - results["hmm"]["labels"][lab])
    fig, ax = plt.subplots(figsize=(WIDTH, 4.8))
    for y, label in enumerate(order):
        if y % 2:
            ax.axhspan(y - 0.5, y + 0.5, color="#f3f2ee", zorder=0, linewidth=0)
        for k, (key, name) in enumerate(THREE):
            low, high = results[key]["label_intervals"][label]
            yy = y + (1 - k) * 0.22
            interval_dot(ax, results[key]["labels"][label], low, high, yy, SERIES[k], label=name if y == 0 else None)
        ax.text(1.065, y, f"{support[label]}", va="center", ha="right", fontsize=8.5, color=MUTED)
    ax.text(1.065, len(order) - 0.35, "spans", va="bottom", ha="right", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(order)), order)
    clean_y(ax)
    ax.set_ylim(-0.5, len(order) - 0.4)
    ax.set_xlim(0.2, 1.07)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("F1 on the Codex test")
    header(ax, "Context matters most for EXCLUDE, rare words for NAME",
           "Labels sorted by how much BERTurk gains over the HMM; lines are 95% bootstrap intervals", legend=True)
    legend_row(ax, 3)
    note(ax, "BERTurk: seed 7. Right: gold spans of each label in the Codex test.")
    save(fig, "per_label.png")


def plot_differences(evaluation):
    """Forest plot of the paired bootstrap differences on the Codex test."""
    pairs = [
        ("hmm - frequent_tag", "HMM − frequent tag", "neighbouring tags"),
        ("crf - hmm_bio", "CRF − HMM", "word features + context"),
        ("crf - crf_no_suffix", "CRF − CRF without suffixes", "Turkish endings"),
        ("berturk_softmax_bio - crf", "BERTurk (BIO rules) − CRF", "pre-training"),
        ("berturk_softmax_bio - berturk_softmax", "BERTurk: BIO rules − softmax", "IOB2 constraint"),
        ("berturk_crf - berturk_softmax_bio", "BERTurk: CRF layer − BIO rules", "learned transitions"),
    ]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.9))
    rows = list(reversed(pairs))
    for y, (key, _, what) in enumerate(rows):
        d = evaluation["differences"][key]["codex_test"]["seed_mean"]
        real = d["low"] > 0
        interval_dot(ax, d["diff"], d["low"], d["high"], y, SERIES[0] if real else MUTED, size=60)
        numbers = f"{d['diff']:+.3f}  [{d['low']:+.3f}, {d['high']:+.3f}]".replace("-", "−")
        ax.annotate(numbers, (max(d["high"], 0.006), y), xytext=(7, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK if real else MUTED)
        ax.annotate(what, (max(d["high"], 0.006), y), xytext=(7, -11), textcoords="offset points", va="center",
                    fontsize=8, color=INK_2 if real else MUTED, style="italic")
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(rows)), [name for _, name, _ in rows])
    clean_y(ax)
    ax.set_xlim(-0.012, 0.25)
    ax.set_ylim(-0.75, len(rows) - 0.5)
    ax.set_xticks([0, 0.05, 0.1, 0.15])
    ax.set_xlabel("Difference in Codex test F1")
    header(ax, "Which step helps? Blue: a real gain; grey: indistinguishable from 0",
           "Dot: difference; line: 95% paired bootstrap interval (both models scored on the same drawn requests)")
    note(ax, "BERTurk: difference of the means over 3 seeds.")
    save(fig, "differences.png")


def plot_speed(evaluation):
    """Accuracy against speed on the Codex test (log scale), one point per method, with the F1 interval."""
    classic = load("classic.json")["models"]
    timing = load("timing_idle_gpu.json")["runs"]
    points = []
    for key, name, bert in METHODS:
        if key == "hmm_bio":
            continue  # same predictions as the HMM
        if bert is None:
            test = evaluation["models"][key]["codex_test"]
            ms, f1 = classic[key]["codex_test"]["ms_per_request"], test["f1"]
        else:
            test = evaluation["seeds"][key]["codex_test"]
            ms = statistics.mean(timing[f"bert-base-turkish-cased{bert[0]}_seed{s}"]["ms_per_request"][bert[1]]
                                 ["codex_test"] for s in SEEDS)
            f1 = test["mean"]
        points.append((name, ms, f1, test["low"], test["high"], bert is not None))
    fig, ax = plt.subplots(figsize=(WIDTH, 4.4))
    for name, ms, f1, low, high, gpu in points:
        color = SERIES[1] if gpu else SERIES[0]
        ax.vlines(ms, low, high, color=color, linewidth=2.2, zorder=2)
        ax.scatter(ms, f1, s=60, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if not gpu:
            ax.annotate(f"{name}  {f1:.3f}", (ms, f1), xytext=(9, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=INK)
    gpu_points = [p for p in points if p[5]]
    lines = [f"{name.replace('BERTurk, ', '').replace('BERTurk + ', '')}  {f1:.3f}  ({ms:.1f} ms)"
             for name, ms, f1, *_ in gpu_points]
    ax.text(max(p[1] for p in gpu_points) * 1.6, 0.94, "\n".join(["BERTurk"] + lines), va="center", fontsize=8.5,
            color=INK, linespacing=1.5)
    ax.text(0.0045, 0.955, "CPU, one request at a time", color=SERIES[0], fontsize=9, fontweight="bold")
    ax.text(0.0045, 0.938, "GPU (RTX 4070 SUPER), batches of 16", color=SERIES[1], fontsize=9, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.004, 30)
    ax.set_xticks([0.01, 0.1, 1], ["0.01 ms", "0.1 ms", "1 ms"])
    ax.set_ylim(0.58, 0.98)
    ax.set_xlabel("Time per request (log scale)")
    ax.set_ylabel("Codex test F1")
    header(ax, "Every model answers in about 1 ms or less; BERTurk is the most accurate",
           "Vertical lines: 95% bootstrap interval of F1. BERTurk: mean of 3 seeds, timed on an idle GPU")
    save(fig, "speed.png")


def plot_training():
    """Dev F1 per epoch for every BERTurk run; the kept (best) epoch is circled."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.6), sharey=True)
    for ax, (suffix, title) in zip(axes, [("", "BERTurk, softmax"), ("_crf", "BERTurk + CRF layer")]):
        for color, run in zip(SERIES, bert_runs(suffix)):
            epochs = [h["epoch"] for h in run["history"]]
            f1s = [h["dev_f1"] for h in run["history"]]
            ax.plot(epochs, f1s, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                    label=f"seed {run['seed']}")
            best = run["best_epoch"]
            ax.scatter(best, f1s[best - 1], s=150, facecolor="none", edgecolor=color, linewidth=1.5, zorder=4)
        ax.set_title(title, fontsize=10, loc="left", color=INK, fontweight="bold")
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("Epoch")
        ax.axhline(0.98, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    axes[0].set_ylabel("Dev F1")
    axes[1].scatter([], [], s=150, facecolor="none", edgecolor=INK_2, linewidth=1.5, label="kept epoch")
    axes[1].legend(loc="lower right", fontsize=8.5)
    fig.tight_layout(w_pad=2)
    header(axes[0], "BERTurk passes 0.98 dev F1 after 2 epochs",
           "Dev F1 after each epoch for 3 seeds; the circled epoch (best dev F1) is the model that was kept",
           panel_titles=True)
    save(fig, "training.png")


def codex_rows():
    return [json.loads(line) for line in (ROOT / "data" / "clean" / "codex_test.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]


def plot_confusion():
    """Row-normalized word-level confusion (B-/I- merged) on the Codex test: HMM against BERTurk + CRF layer."""
    gold = [r["tags"] for r in codex_rows()]
    classes = ["O"] + LABELS
    runs = [("HMM", "hmm.json"), ("BERTurk + CRF layer (seed 7)", "bert-base-turkish-cased_crf_seed7_crf.json")]
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 4.9), gridspec_kw={"wspace": 0.08})
    for n, (ax, (title, path)) in enumerate(zip(axes, runs)):
        preds = load(f"predictions/{path}")["codex_test"]
        counts = [[0] * len(classes) for _ in classes]
        for g_tags, p_tags in zip(gold, preds):
            for g, p in zip(g_tags, p_tags):
                counts[classes.index(g[2:] if g != "O" else g)][classes.index(p[2:] if p != "O" else p)] += 1
        share = [[c / sum(row) for c in row] for row in counts]
        image = ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for i, row in enumerate(share):
            for j, v in enumerate(row):
                if v >= 0.01:
                    ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=8,
                            color=SURFACE if v > 0.45 else INK, fontweight="bold" if i != j and v >= 0.1 else None)
        ax.add_patch(Rectangle((classes.index("TYPE") - 0.5, classes.index("EXCLUDE") - 0.5), 1, 1, fill=False,
                               edgecolor=SERIES[1], linewidth=2))
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(classes)), classes if n == 0 else [], fontsize=8.5)
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
        ax.set_xlabel("Predicted label")
    axes[0].set_ylabel("Gold label")
    bar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    bar.set_ticks([0, 0.5, 1], labels=["0%", "50%", "100%"])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8)
    header(axes[0], "The HMM tags most excluded words as TYPE; BERTurk keeps them apart",
           "Share of each gold label's words (%) that got each predicted label (rows sum to 100). Orange box: "
           "excluded words tagged as TYPE", panel_titles=True)
    save(fig, "confusion.png")


def plot_robustness():
    """F1 change on each perturbed copy of the Codex test, for three models."""
    if not (RESULTS / "robustness.json").exists():
        return
    rob = load("robustness.json")
    names = {"lowercase": "all lowercase", "no_turkish_letters": "no Turkish letters", "no_apostrophes":
             "no apostrophes", "all_caps": "ALL CAPS", "typos_10": "typos in 10% of words", "typos_25":
             "typos in 25% of words", "messy": "messy: first three + 10% typos"}
    rows = list(reversed(list(names)))
    fig, ax = plt.subplots(figsize=(WIDTH, 4.8))
    ax.axhspan(rows.index("all_caps") - 0.5, rows.index("all_caps") + 0.5, color="#fdf0ea", zorder=0, linewidth=0)
    for k, (key, label) in enumerate(THREE):
        result = rob["models"][key]["perturbed"]
        for y, p in enumerate(rows):
            interval_dot(ax, result[p]["change"], result[p]["change_low"], result[p]["change_high"],
                         y + (1 - k) * 0.24, SERIES[k], label=f"{label} (clean F1 {result['clean']['f1']:.3f})"
                         if y == 0 else None)
    caps = rob["models"]["berturk_crf"]["perturbed"]["all_caps"]
    ax.annotate(f"{caps['change']:+.3f}: BERTurk is a cased model".replace("-", "−"),
                (caps["change_high"], rows.index("all_caps") - 0.24), xytext=(8, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8.5, color=SERIES[2], fontweight="bold")
    ax.axhline(2.5, color=GRID, linewidth=0.8, zorder=0)
    ax.text(-0.235, 6.45, "spelling and case", fontsize=8, color=MUTED, va="top", style="italic")
    ax.text(-0.235, 2.45, "typos", fontsize=8, color=MUTED, va="top", style="italic")
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(rows)), [names[p] for p in rows])
    clean_y(ax)
    ax.set_xlim(-0.24, 0.012)
    ax.set_xlabel("Change in Codex test F1 against the clean text")
    header(ax, "Attacking the input: what each change costs",
           "Dot: change in F1; line: 95% paired bootstrap interval. BERTurk: mean of 3 seeds", legend=True)
    legend_row(ax, 3)
    save(fig, "robustness.png")


def plot_contrast():
    """Small multiples: target span right in the wanted version, in the excluded version, and in both."""
    if not (RESULTS / "robustness.json").exists():
        return
    rob = load("robustness.json")
    groups = [("all", "all 140 pairs"), ("NAME", "NAME   Duman olsun / olmasın"),
              ("PLACE", "PLACE   Kadıköy civarında / dışında"), ("TYPE", "TYPE   konser olsun / olmasın")]
    panels = [("want", "Wanted version right"), ("exclude", "Excluded version right"), ("pair", "Both right")]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.4), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (part, title) in zip(axes, panels):
        for y, (group, _) in enumerate(groups):
            for k, (key, label) in enumerate(THREE):
                c = rob["models"][key]["contrast"][group]
                interval_dot(ax, c[part], c[part + "_low"], c[part + "_high"], y + (1 - k) * 0.22, SERIES[k],
                             size=36, label=label if (y == 0 and part == "want") else None)
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold" if part == "pair" else None,
                     color=INK)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.axhline(0.5, color=GRID, linewidth=0.8, zorder=0)
        clean_y(ax)
    axes[0].set_yticks(range(len(groups)), [shown for _, shown in groups])
    axes[0].set_ylim(-0.5, len(groups) - 0.5)
    axes[1].set_xlabel("Share of pairs")
    header(axes[0], "Contrast set: one word flips the label, and the HMM cannot see it",
           "The HMM gets the wanted label but almost never EXCLUDE, so no pair is right. Lines: 95% intervals",
           legend=True, panel_titles=True)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=2.3, columnspacing=1.6)
    note(axes[0], "Intervals: bootstrap over pairs. BERTurk: mean of 3 seeds. Pairs are built from templates.",
         below=36)
    save(fig, "contrast.png")


def plot_learning_curve():
    """Codex test F1 against the number of training requests (3 random subsets per point)."""
    if not (RESULTS / "learning_curve.json").exists():
        return
    curve = load("learning_curve.json")
    fig, ax = plt.subplots(figsize=(WIDTH, 4.4))
    shown = [("hmm", "HMM"), ("crf", "CRF"), ("berturk_softmax_bio", "BERTurk")]
    for k, (key, label) in enumerate(shown):
        if key not in curve["models"]:
            continue
        points = curve["models"][key]
        xs = [p["requests"] for p in points]
        ax.fill_between(xs, [p["low"] for p in points], [p["high"] for p in points], color=SERIES[k],
                        alpha=0.15, linewidth=0)
        ax.plot(xs, [p["mean"] for p in points], color=SERIES[k], linewidth=2.2, marker="o", markersize=6,
                markeredgecolor=SURFACE)
        ax.text(xs[-1] * 1.08, points[-1]["mean"], f"{label}  {points[-1]['mean']:.3f}", va="center",
                fontsize=9, color=SERIES[k], fontweight="bold")
        ax.text(xs[0] / 1.08, points[0]["mean"], f"{points[0]['mean']:.3f}", va="center", ha="right", fontsize=8.5,
                color=INK_2)
    crf_full = curve["models"]["crf"][-1]["mean"]
    xs = [p["requests"] for p in curve["models"]["crf"]]
    ax.hlines(crf_full, xs[0], xs[-1], color=SERIES[1], linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    if "berturk_softmax_bio" in curve["models"]:
        quarter = curve["models"]["berturk_softmax_bio"][1]
        ax.annotate(f"BERTurk on 25% of the data ({quarter['mean']:.3f})\nbeats the CRF on all of it ({crf_full:.3f})",
                    (quarter["requests"], quarter["mean"]), xytext=(xs[0] * 1.02, 0.955),
                    fontsize=8.5, color=INK, ha="left", va="top",
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    ax.set_xscale("log")
    ax.set_xticks(xs, [f"{x:,}\n{p:.0%}" for x, p in zip(xs, curve["fractions"])])
    ax.minorticks_off()
    ax.set_xlim(xs[0] * 0.78, xs[-1] * 1.9)
    ax.set_xlabel("Training requests (log scale)")
    ax.set_ylabel("Codex test F1")
    header(ax, "How much training data does each model need?",
           "Mean of 3 random subsets; band: 95% bootstrap interval of that mean; dashed: CRF with all the data")
    save(fig, "learning_curve.png")


def main():
    evaluation = load("evaluation.json")
    plot_dev_vs_test(evaluation)
    plot_per_label(evaluation)
    plot_differences(evaluation)
    plot_speed(evaluation)
    plot_training()
    plot_confusion()
    plot_robustness()
    plot_contrast()
    plot_learning_curve()


if __name__ == "__main__":
    main()
