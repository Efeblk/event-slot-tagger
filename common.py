"""Shared helpers: tokenizing Turkish requests, span <-> BIO conversion, exact-span scoring and bootstrap intervals."""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
LABELS = ("PLACE", "DATE", "TYPE", "NAME", "PRICE", "PEOPLE", "EXCLUDE")
TAGS = ("O",) + tuple(f"{p}-{label}" for label in LABELS for p in "BI")

# A word keeps an inner apostrophe or hyphen (İstanbul'da, stand-up); a number keeps . : , and a suffix with or
# without an apostrophe (20:00'den, 19.30dan, 1.500). Any other punctuation mark is its own token.
TOKEN = re.compile(r"\d+(?:[.:,]\d+)*(?:['’]?[^\W\d_]+)?|\w+(?:['’-]\w+)*|[^\w\s]")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f.read().split("\n") if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def tokenize(text):
    return [m.group() for m in TOKEN.finditer(text)]


def to_bio(text, span_list):
    """Tokens and BIO tags for a text and its [[label, substring], ...] list.

    Substrings are searched left to right, each after the end of the previous one, and must start and end on token
    boundaries. Raise ValueError on an unknown label, a missing substring or a substring that cuts a token.
    """
    matches = list(TOKEN.finditer(text))
    starts, ends = {m.start() for m in matches}, {m.end() for m in matches}
    tags, pos = ["O"] * len(matches), 0
    for label, substring in span_list:
        if label not in LABELS:
            raise ValueError(f"unknown label {label!r} in {text!r}")
        start = text.find(substring, pos) if substring.strip() else -1
        while start >= 0 and not (start in starts and start + len(substring) in ends):
            start = text.find(substring, start + 1)
        if start < 0:
            raise ValueError(f"{label} span {substring!r} not found on token boundaries in {text!r}")
        end = start + len(substring)
        inside = [i for i, m in enumerate(matches) if m.start() >= start and m.end() <= end]
        tags[inside[0]] = f"B-{label}"
        for i in inside[1:]:
            tags[i] = f"I-{label}"
        pos = end
    return [m.group() for m in matches], tags


def spans(tags):
    """Set of (label, start, end) spans, end exclusive. An I- tag that does not continue its own label starts a new
    span, so every tag sequence can be scored."""
    found, current = set(), None
    for i, tag in enumerate(list(tags) + ["O"]):
        continues = tag.startswith("I-") and current is not None and current[0] == tag[2:]
        if current is not None and not continues:
            found.add((current[0], current[1], i))
            current = None
        if tag != "O" and not continues:
            current = (tag[2:], i)
    return found


def bio_valid(previous, tag):
    """IOB2 rule: I-X may only follow B-X or I-X. START counts as O."""
    return not tag.startswith("I-") or previous[2:] == tag[2:]


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def span_scores(gold_list, pred_list):
    """Micro precision/recall/F1 over exact spans (same label and boundaries), macro F1 over labels, per-label
    scores, token accuracy, and the share of requests whose spans are all right."""
    counts = defaultdict(lambda: [0, 0, 0])
    exact = correct = total = 0
    for gold_tags, pred_tags in zip(gold_list, pred_list):
        gold, pred = spans(gold_tags), spans(pred_tags)
        exact += gold == pred
        correct += sum(g == p for g, p in zip(gold_tags, pred_tags))
        total += len(gold_tags)
        for label, *_ in gold & pred:
            counts[label][0] += 1
        for label, *_ in pred - gold:
            counts[label][1] += 1
        for label, *_ in gold - pred:
            counts[label][2] += 1
    labels = {label: {**prf(*counts[label]), "support": counts[label][0] + counts[label][2]}
              for label in LABELS if label in counts}
    present = [s["f1"] for s in labels.values() if s["support"]]
    return {**prf(*(sum(c[i] for c in counts.values()) for i in range(3))),
            "macro_f1": sum(present) / len(present) if present else 0.0,
            "token_accuracy": correct / total if total else 0.0,
            "request_exact": exact / len(gold_list) if gold_list else 0.0,
            "labels": labels}


def bootstrap_f1(gold_list, pred_list, base_list=None, rounds=1000, seed=0, label=None):
    """95% bootstrap interval of micro span F1 over requests drawn with replacement. With base_list, the interval of
    F1(pred) - F1(base) on the same draws (a paired comparison). With label, only spans of that label count.
    Returns (low, high)."""
    return bootstrap_mean_f1(gold_list, [pred_list], None if base_list is None else [base_list], rounds, seed, label)


def bootstrap_mean_f1(gold_list, runs, base_runs=None, rounds=1000, seed=0, label=None):
    """Like bootstrap_f1 for a model trained several times (one prediction list per seed): each draw scores the mean
    F1 of the runs, so the interval is that of the mean over seeds. Every run uses the same drawn requests."""
    def per_request(preds):
        rows = []
        for a, b in zip(gold_list, preds):
            g, p = spans(a), spans(b)
            if label is not None:
                g, p = {x for x in g if x[0] == label}, {x for x in p if x[0] == label}
            rows.append((len(g & p), len(p - g), len(g - p)))
        return np.array(rows, dtype=float)

    def f1(counts):
        tp, fp, fn = counts.sum(axis=1).T
        denominator = 2 * tp + fp + fn
        return np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)

    picks = np.random.default_rng(seed).integers(0, len(gold_list), size=(rounds, len(gold_list)))
    scores = np.mean([f1(per_request(preds)[picks]) for preds in runs], axis=0)
    if base_runs is not None:
        scores = scores - np.mean([f1(per_request(preds)[picks]) for preds in base_runs], axis=0)
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_mean(values, rounds=1000, seed=0):
    """95% bootstrap interval of the mean of values [items] or [runs, items] (then the mean over runs and items, with
    the same drawn items for every run). Returns (low, high)."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    picks = np.random.default_rng(seed).integers(0, values.shape[1], size=(rounds, values.shape[1]))
    low, high = np.percentile(values[:, picks].mean(axis=(0, 2)), [2.5, 97.5])
    return float(low), float(high)
