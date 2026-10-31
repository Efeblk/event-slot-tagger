"""Compare every trained tagger from its saved predictions (results/predictions/*.json); no model is rerun.

For each test set (codex_test, biplan_test) and model:
- exact-span micro F1 with a 95% bootstrap interval over requests (1,000 draws), macro F1, per-label F1 and a
  bootstrap interval for each label;
- for BERTurk models trained with several seeds: mean and standard deviation over the seeds, and the bootstrap
  interval of the mean (each draw scores every seed on the same requests);
- slices: requests typed without any Turkish letter vs with them, and PLACE spans written with a suffix;
- IOB2 breaks: predicted I- tags that do not continue a span of their own label;
- error types: wrong boundary, wrong label, missed span, invented span.
Every Codex test request that BERTurk + CRF (lowest seed) gets wrong is listed with its gold and predicted spans in
results/error_examples.json, for reading the errors by hand.
Paired bootstrap intervals for the differences in PAIRS (the claims in the README) use the same draws for both models.
For BERTurk, the run with the lowest seed stands for its group in the
pairs; "seed_mean" repeats each pair with the mean over all seeds of each model. Writes results/evaluation.json.

Usage: python evaluate.py
"""

import json
import re
import statistics
from collections import Counter, defaultdict

from common import ROOT, bootstrap_f1, bootstrap_mean_f1, read_jsonl, span_scores, spans

TESTS = ("codex_test", "biplan_test")
TURKISH = re.compile("[çğıöşüÇĞİÖŞÜ]")
PAIRS = (("crf", "hmm_bio"), ("crf", "crf_no_suffix"), ("hmm_bio", "hmm"), ("hmm", "frequent_tag"),
         ("berturk_softmax_bio", "berturk_softmax"), ("berturk_crf", "berturk_softmax_bio"),
         ("berturk_softmax_bio", "crf"))


def load_predictions():
    """{model: {seed or None: {split: tags}}}. BERTurk files are named bert-base-turkish-cased[_crf]_seed<N>_<decoder>."""
    models = defaultdict(dict)
    for path in sorted((ROOT / "results" / "predictions").glob("*.json")):
        match = re.fullmatch(r"bert-base-turkish-cased(_crf)?_seed(\d+)_(\w+)", path.stem)
        if match:
            name, seed = ("berturk_crf" if match[1] else f"berturk_{match[3]}"), int(match[2])
        else:
            name, seed = path.stem, None
        models[name][seed] = json.loads(path.read_text(encoding="utf-8"))
    return models


def show(tokens, tags):
    return [f"{label}: {' '.join(tokens[s:e])}" for label, s, e in sorted(spans(tags), key=lambda x: x[1])]


def iob2_breaks(pred_list):
    return sum(tag.startswith("I-") and prev[2:] != tag[2:]
               for tags in pred_list for prev, tag in zip(["O"] + tags[:-1], tags))


def error_types(gold_list, pred_list):
    counts = Counter()
    for gold_tags, pred_tags in zip(gold_list, pred_list):
        gold, pred = spans(gold_tags), spans(pred_tags)
        for label, s, e in gold - pred:
            near = [p for p in pred if p[1] < e and p[2] > s]
            if any(p[0] == label for p in near):
                counts["wrong boundary"] += 1
            elif near:
                counts["wrong label"] += 1
            else:
                counts["missed"] += 1
        counts["invented"] += sum(1 for _, s, e in pred - gold if not any(g[1] < e and g[2] > s for g in gold))
    return dict(counts)


def place_suffix_recall(rows, pred_list):
    """Recall of PLACE spans whose last word carries a suffix (an apostrophe or a -de/-da/-te/-ta ending)."""
    found = total = 0
    for r, pred in zip(rows, pred_list):
        for label, s, e in spans(r["tags"]):
            word = r["tokens"][e - 1].lower()
            if label == "PLACE" and ("'" in word or "’" in word or word[-2:] in ("de", "da", "te", "ta")):
                total += 1
                found += (label, s, e) in spans(pred)
    return found / total if total else None


def describe(rows, preds):
    gold = [r["tags"] for r in rows]
    low, high = bootstrap_f1(gold, preds)
    plain = [i for i, r in enumerate(rows) if not TURKISH.search(r["text"])]
    turkish = [i for i, r in enumerate(rows) if TURKISH.search(r["text"])]
    scores = span_scores(gold, preds)
    return {"f1": scores["f1"], "low": low, "high": high, "macro_f1": scores["macro_f1"],
            "precision": scores["precision"], "recall": scores["recall"],
            "request_exact": scores["request_exact"],
            "labels": {k: round(v["f1"], 4) for k, v in scores["labels"].items()},
            "label_intervals": {k: bootstrap_f1(gold, preds, label=k) for k in scores["labels"]},
            "f1_no_turkish_letters": span_scores([gold[i] for i in plain], [preds[i] for i in plain])["f1"],
            "f1_with_turkish_letters": span_scores([gold[i] for i in turkish], [preds[i] for i in turkish])["f1"],
            "place_suffix_recall": place_suffix_recall(rows, preds),
            "iob2_breaks": iob2_breaks(preds), "errors": error_types(gold, preds)}


def main():
    tests = {name: read_jsonl(ROOT / "data" / "clean" / f"{name}.jsonl") for name in TESTS}
    models = load_predictions()
    out = {"requests": {t: len(rows) for t, rows in tests.items()},
           "slice_sizes": {t: sum(not TURKISH.search(r["text"]) for r in rows) for t, rows in tests.items()},
           "models": {}, "seeds": {}, "differences": {}}
    first = {}
    for name, by_seed in models.items():
        seed = min(by_seed, key=lambda s: -1 if s is None else s)
        first[name] = by_seed[seed]
        out["models"][name] = {t: describe(rows, by_seed[seed][t]) for t, rows in tests.items()}
        if len(by_seed) > 1:
            out["seeds"][name] = {}
            for t, rows in tests.items():
                f1s = [span_scores([r["tags"] for r in rows], by_seed[s][t])["f1"] for s in sorted(by_seed)]
                low, high = bootstrap_mean_f1([r["tags"] for r in rows], [by_seed[s][t] for s in sorted(by_seed)])
                out["seeds"][name][t] = {"seeds": sorted(by_seed), "f1": f1s, "mean": statistics.mean(f1s),
                                         "std": statistics.stdev(f1s), "low": low, "high": high}
    for model, base in PAIRS:
        if model in first and base in first:
            out["differences"][f"{model} - {base}"] = {}
            for t, rows in tests.items():
                gold = [r["tags"] for r in rows]
                diff = span_scores(gold, first[model][t])["f1"] - span_scores(gold, first[base][t])["f1"]
                low, high = bootstrap_f1(gold, first[model][t], base_list=first[base][t])
                out["differences"][f"{model} - {base}"][t] = {"diff": diff, "low": low, "high": high}
                runs, base_runs = ([p[t] for _, p in sorted(models[m].items(), key=lambda kv: kv[0] or 0)]
                                   for m in (model, base))
                mean_diff = (statistics.mean(span_scores(gold, p)["f1"] for p in runs)
                             - statistics.mean(span_scores(gold, p)["f1"] for p in base_runs))
                low, high = bootstrap_mean_f1(gold, runs, base_runs)
                out["differences"][f"{model} - {base}"][t]["seed_mean"] = {"diff": mean_diff, "low": low,
                                                                           "high": high}

    if "berturk_crf" in first:
        rows = tests["codex_test"]
        wrong = [{"text": r["text"], "gold": show(r["tokens"], r["tags"]), "predicted": show(r["tokens"], p)}
                 for r, p in zip(rows, first["berturk_crf"]["codex_test"]) if spans(r["tags"]) != spans(p)]
        path = ROOT / "results" / "error_examples.json"
        path.write_text(json.dumps(wrong, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (ROOT / "results" / "evaluation.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                                                      encoding="utf-8")
    print(f"{'model':<22}" + "".join(f"{t:>28}" for t in TESTS) + "   IOB2 breaks")
    for name, by_test in out["models"].items():
        cells = "".join(f"{s['f1']:>12.3f} [{s['low']:.3f}, {s['high']:.3f}]" for s in by_test.values())
        print(f"{name:<22}{cells}   {sum(s['iob2_breaks'] for s in by_test.values())}")
    for name, by_test in out["seeds"].items():
        print(f"{name:<22}" + "".join(f"{'mean ' + format(s['mean'], '.3f') + ' ± ' + format(s['std'], '.3f'):>28}"
                                      for s in by_test.values()) + f"   ({len(next(iter(by_test.values()))['f1'])} seeds)")
    for key, by_test in out["differences"].items():
        print(f"{key:<44}" + "".join(f"{s['diff']:>+8.3f} [{s['low']:+.3f}, {s['high']:+.3f}]" for s in by_test.values()))
    print("wrote results/evaluation.json")


if __name__ == "__main__":
    main()
