"""Draws the README figures from the saved results (no model needed).

Reads results/evaluation.json, results/classic.json, results/bert-base-turkish-cased*_seed*.json,
results/timing_idle_gpu.json, results/predictions/, and (when present) results/robustness.json and
results/learning_curve.json; writes results/figures/*.png.

Usage: python plot_results.py
"""
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, in fixed order
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential
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
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": MUTED,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False,
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


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote results/figures/{name}")


def plot_dev_vs_test(evaluation):
    """Dumbbell: dev F1 (same writer as train) against Codex test F1 (another writer), with the 95% interval."""
    dev = dev_f1()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    rows = list(reversed(METHODS))
    for y, (key, _, bert) in enumerate(rows):
        test = evaluation["seeds" if bert else "models"][key]["codex_test"]
        f1 = test["mean"] if bert else test["f1"]
        ax.plot([f1, dev[key]], [y, y], color=GRID, linewidth=2, zorder=1)
        ax.plot([test["low"], test["high"]], [y, y], color=SERIES[0], linewidth=2, zorder=2,
                solid_capstyle="round")
        ax.scatter(dev[key], y, s=60, color=SERIES[1], edgecolor=SURFACE, linewidth=2, zorder=3,
                   label="Dev (same AI writer as training)" if y == 0 else None)
        ax.scatter(f1, y, s=60, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3,
                   label="Codex test (different AI writer), 95% interval" if y == 0 else None)
        ax.text(test["low"] - 0.005, y, f"{f1:.3f}", ha="right", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(range(len(rows)), [name for _, name, _ in rows])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.55, 1.0)
    ax.set_xlabel("Exact-span micro F1")
    ax.set_title("Every model drops on text from a writer it never saw", pad=10)
    ax.legend(loc="lower left", fontsize=9, handletextpad=0.3, facecolor=SURFACE, frameon=True, edgecolor=GRID)
    fig.text(0.0, -0.02, "BERTurk: mean of 3 seeds and the interval of that mean. 1,000 bootstrap draws over requests.",
             fontsize=8, color=MUTED)
    save(fig, "dev_vs_test.png")


def plot_per_label(evaluation):
    """Grouped bars: F1 per label on the Codex test for HMM, CRF and BERTurk + CRF layer, with 95% intervals."""
    models = [("hmm", "HMM"), ("crf", "CRF"), ("berturk_crf", "BERTurk + CRF layer")]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.26
    for i, (key, name) in enumerate(models):
        result = evaluation["models"][key]["codex_test"]
        values = [result["labels"][label] for label in LABELS]
        errors = [[v - result["label_intervals"][label][0] for label, v in zip(LABELS, values)],
                  [result["label_intervals"][label][1] - v for label, v in zip(LABELS, values)]]
        xs = [x + (i - 1) * width for x in range(len(LABELS))]
        ax.bar(xs, values, width=width, color=SERIES[i], edgecolor=SURFACE, linewidth=1.5, label=name)
        ax.errorbar(xs, values, yerr=errors, fmt="none", ecolor=INK_2, elinewidth=1, capsize=0)
    fig.text(0.0, -0.02, "Lines: 95% bootstrap intervals over requests. BERTurk: seed 7.", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 on the Codex test")
    ax.set_title("Context matters most for EXCLUDE, rare words for NAME", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9)
    save(fig, "per_label.png")


def plot_differences(evaluation):
    """Forest plot of the paired bootstrap differences on the Codex test."""
    pairs = [
        ("hmm - frequent_tag", "HMM − frequent tag"),
        ("crf - hmm_bio", "CRF − HMM"),
        ("crf - crf_no_suffix", "CRF − CRF without suffixes"),
        ("berturk_softmax_bio - crf", "BERTurk (BIO rules) − CRF"),
        ("berturk_softmax_bio - berturk_softmax", "BERTurk: BIO rules − softmax"),
        ("berturk_crf - berturk_softmax_bio", "BERTurk: CRF layer − BIO rules"),
    ]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    rows = list(reversed(pairs))
    for y, (key, _) in enumerate(rows):
        d = evaluation["differences"][key]["codex_test"]["seed_mean"]
        ax.plot([d["low"], d["high"]], [y, y], color=SERIES[0], linewidth=2, solid_capstyle="round")
        ax.scatter(d["diff"], y, s=60, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(d["high"] + 0.004, y, f"{d['diff']:+.3f}", va="center", fontsize=8.5, color=INK_2)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(rows)), [name for _, name in rows])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.02, 0.19)
    ax.set_xlabel("Difference in Codex test F1 (dot) and 95% paired bootstrap interval")
    ax.set_title("Which step helps? An interval crossing 0 means no measurable gain")
    fig.text(0.0, -0.09, "BERTurk: difference of the means over 3 seeds; every seed scored on the same drawn requests.",
             fontsize=8, color=MUTED)
    save(fig, "differences.png")


def plot_speed(evaluation):
    """Accuracy against speed on the Codex test (log scale), one point per method."""
    classic = load("classic.json")["models"]
    timing = load("timing_idle_gpu.json")["runs"]
    points = []
    for key, name, bert in METHODS:
        if key == "hmm_bio":
            continue  # same predictions as the HMM
        if bert is None:
            ms, f1 = classic[key]["codex_test"]["ms_per_request"], evaluation["models"][key]["codex_test"]["f1"]
        else:
            ms = statistics.mean(timing[f"bert-base-turkish-cased{bert[0]}_seed{s}"]["ms_per_request"][bert[1]]
                                 ["codex_test"] for s in SEEDS)
            f1 = evaluation["seeds"][key]["codex_test"]["mean"]
        points.append((name, ms, f1, bert is not None))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    offsets = {"BERTurk, softmax": (-8, 6, "right"), "BERTurk, softmax + BIO rules": (-8, -12, "right"),
               "BERTurk + CRF layer": (8, 2, "left")}
    for name, ms, f1, gpu in points:
        ax.scatter(ms, f1, s=70, color=SERIES[1] if gpu else SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
        dx, dy, ha = offsets.get(name, (8, -3, "left"))
        ax.annotate(name, (ms, f1), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=8.5, color=INK)
    ax.scatter([], [], s=70, color=SERIES[0], label="CPU")
    ax.scatter([], [], s=70, color=SERIES[1], label="GPU (RTX 4070 SUPER, batches of 16)")
    ax.set_xscale("log")
    ax.set_xlim(0.003, 5)
    ax.set_ylim(0.6, 0.97)
    ax.set_xlabel("Milliseconds per request (log scale)")
    ax.set_ylabel("Codex test F1")
    ax.set_title("Every model answers in about 1 ms or less; BERTurk is the most accurate", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
    save(fig, "speed.png")


def plot_training():
    """Dev F1 per epoch for every BERTurk run: dev F1 passes 0.98 after 2 epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, (suffix, title) in zip(axes, [("", "BERTurk, softmax"), ("_crf", "BERTurk + CRF layer")]):
        for color, run in zip(SERIES, bert_runs(suffix)):
            ax.plot([h["epoch"] for h in run["history"]], [h["dev_f1"] for h in run["history"]], color=color,
                    linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, label=f"seed {run['seed']}")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Dev F1")
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle("BERTurk passes 0.98 dev F1 after 2 epochs", x=0.0, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "training.png")


def plot_confusion():
    """Row-normalized word-level confusion (B-/I- merged) on the Codex test: HMM against BERTurk + CRF layer."""
    from matplotlib.colors import LinearSegmentedColormap
    gold = [json.loads(line)["tags"] for line in (ROOT / "data" / "clean" / "codex_test.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    classes = ["O"] + LABELS
    runs = [("HMM", "hmm.json"), ("BERTurk + CRF layer (seed 7)", "bert-base-turkish-cased_crf_seed7_crf.json")]
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    for ax, (title, path) in zip(axes, runs):
        preds = load(f"predictions/{path}")["codex_test"]
        counts = [[0] * len(classes) for _ in classes]
        for g_tags, p_tags in zip(gold, preds):
            for g, p in zip(g_tags, p_tags):
                counts[classes.index(g[2:] if g != "O" else g)][classes.index(p[2:] if p != "O" else p)] += 1
        share = [[c / sum(row) for c in row] for row in counts]
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for i, row in enumerate(share):
            for j, v in enumerate(row):
                if v >= 0.005:
                    ax.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else "1", ha="center", va="center", fontsize=7.5,
                            color=SURFACE if v > 0.5 else INK)
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes)), classes, fontsize=8)
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Predicted label")
    axes[0].set_ylabel("Gold label (each row sums to 1)")
    fig.suptitle("The HMM tags most excluded words as TYPE; BERTurk keeps them apart", x=0.0, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "confusion.png")


def plot_robustness():
    """F1 change on each perturbed copy of the Codex test, for three models."""
    if not (RESULTS / "robustness.json").exists():
        return
    rob = load("robustness.json")
    names = {"lowercase": "all lowercase", "no_turkish_letters": "no Turkish letters", "no_apostrophes":
             "no apostrophes", "all_caps": "ALL CAPS", "typos_10": "typos in 10% of words", "typos_25":
             "typos in 25% of words", "messy": "messy (first three + 10% typos)"}
    rows = list(reversed(list(names)))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for k, (key, label) in enumerate(THREE):
        result = rob["models"][key]["perturbed"]
        clean = result["clean"]["f1"]
        drops = [result[p]["change"] for p in rows]
        ys = [y + (1 - k) * 0.22 for y in range(len(rows))]
        ax.hlines(ys, [result[p]["change_low"] for p in rows], [result[p]["change_high"] for p in rows],
                  color=SERIES[k], linewidth=2)
        ax.scatter(drops, ys, s=50, color=SERIES[k], edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label=f"{label} (clean {clean:.3f})")
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(rows)), [names[p] for p in rows])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in Codex test F1 against the clean text (dot) and 95% paired bootstrap interval")
    ax.set_title("Attacking the input: how much F1 each change costs", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8.5)
    save(fig, "robustness.png")


def plot_contrast():
    """Share of minimal pairs where the target span is right in both the wanted and the excluded version."""
    if not (RESULTS / "robustness.json").exists():
        return
    rob = load("robustness.json")
    groups = [("TYPE", "TYPE ↔ EXCLUDE\nkonser olsun / olmasın"), ("PLACE", "PLACE ↔ EXCLUDE\nKadıköy civarında / dışında"),
              ("NAME", "NAME ↔ EXCLUDE\nDuman olsun / olmasın"), ("all", "all pairs")]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    width = 0.26
    for k, (key, label) in enumerate(THREE):
        result = rob["models"][key]["contrast"]
        values = [result[g]["pair"] for g, _ in groups]
        xs = [x + (k - 1) * width for x in range(len(groups))]
        ax.bar(xs, values, width=width, color=SERIES[k], edgecolor=SURFACE, linewidth=1.5, label=label)
        ax.errorbar(xs, values, yerr=[[v - result[g]["pair_low"] for (g, _), v in zip(groups, values)],
                                      [result[g]["pair_high"] - v for (g, _), v in zip(groups, values)]],
                    fmt="none", ecolor=INK_2, elinewidth=1, capsize=0)
        for x, v in zip(xs, values):
            if v < 0.02:
                ax.text(x, 0.02, "0", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(groups)), [shown for _, shown in groups], fontsize=8.5)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pairs with both versions right")
    ax.set_title(f"Contrast set ({rob['contrast_pairs']} minimal pairs): does one word flip the label?", pad=28)
    fig.text(0.0, -0.06, "Lines: 95% bootstrap intervals over pairs. BERTurk: mean of 3 seeds.", fontsize=8, color=MUTED)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9)
    save(fig, "contrast.png")


def plot_learning_curve():
    """Codex test F1 against the number of training requests (3 random subsets per point)."""
    if not (RESULTS / "learning_curve.json").exists():
        return
    curve = load("learning_curve.json")
    fig, ax = plt.subplots(figsize=(8, 4))
    shown = [("hmm", "HMM"), ("crf", "CRF"), ("berturk_softmax_bio", "BERTurk, softmax + BIO rules")]
    for k, (key, label) in enumerate(shown):
        if key not in curve["models"]:
            continue
        points = curve["models"][key]
        xs = [p["requests"] for p in points]
        ax.fill_between(xs, [p["low"] for p in points], [p["high"] for p in points], color=SERIES[k],
                        alpha=0.15, linewidth=0)
        ax.plot(xs, [p["mean"] for p in points], color=SERIES[k], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, label=label)
        ax.annotate(f"{points[0]['mean']:.2f}", (xs[0], points[0]["mean"]), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8.5, color=INK_2)
    ax.set_xscale("log")
    ax.set_xticks(xs, [f"{x:,}" for x in xs])
    ax.minorticks_off()
    ax.set_xlim(xs[0] * 0.75, xs[-1] * 1.15)
    ax.set_xlabel("Training requests (log scale; 10%, 25%, 50%, 100%)")
    ax.set_ylabel("Codex test F1")
    ax.set_title("How much training data does each model need?", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9)
    fig.text(0.0, -0.03, "Line: mean of 3 random subsets (seeds 7, 13, 42); band: 95% bootstrap interval of that mean.",
             fontsize=8,
             color=MUTED)
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
