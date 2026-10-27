"""Train the count-based and feature-based taggers on data/clean/train.jsonl and score them.

Models: the most-frequent-tag baseline, the HMM (plain and with IOB2 rules in Viterbi), and the CRF (with and without
suffix features, to measure what Turkish case endings add). The CRF penalties c1, c2 are chosen on the dev set from a
small grid; the test sets are scored once with the chosen setting.

Writes results/classic.json (scores, chosen settings, speed) and results/predictions/<model>.json (predicted tags for
dev and both test sets, used by evaluate.py).

Usage: python train_classic.py
"""

import json
import time

from common import ROOT, read_jsonl, span_scores
from crf import CRF
from hmm import HMM, FrequentTag

SPLITS = ("dev", "codex_test", "biplan_test")
GRID = [(c1, c2) for c1 in (0.0, 0.05, 0.2) for c2 in (0.01, 0.1, 0.5)]


def score(model, rows):
    """Predicted tags, span scores and milliseconds per request for one split."""
    start = time.perf_counter()
    preds = [model.predict(r["tokens"]) for r in rows]
    ms = (time.perf_counter() - start) * 1000 / len(rows)
    return preds, {**span_scores([r["tags"] for r in rows], preds), "ms_per_request": ms}


def main():
    data = {name: read_jsonl(ROOT / "data" / "clean" / f"{name}.jsonl") for name in ("train",) + SPLITS}
    train = data["train"]

    grid = []
    for c1, c2 in GRID:
        dev_f1 = score(CRF(c1, c2).fit(train), data["dev"])[1]["f1"]
        grid.append({"c1": c1, "c2": c2, "dev_f1": dev_f1})
        print(f"CRF c1={c1} c2={c2}: dev F1 {dev_f1:.3f}")
    chosen = max(grid, key=lambda g: g["dev_f1"])

    models = {
        "frequent_tag": FrequentTag(),
        "hmm": HMM(),
        "hmm_bio": HMM(constrained=True),
        "crf": CRF(chosen["c1"], chosen["c2"]),
        "crf_no_suffix": CRF(chosen["c1"], chosen["c2"], suffixes=False),
    }
    out = {"crf_grid": grid, "crf_chosen": chosen, "models": {}}
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(train)
        out["models"][name] = {"train_seconds": time.perf_counter() - start}
        preds = {}
        for split in SPLITS:
            preds[split], out["models"][name][split] = score(model, data[split])
        path = ROOT / "results" / "predictions" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(preds) + "\n", encoding="utf-8")
        print(f"{name:<15}" + "".join(f"  {split} F1 {out['models'][name][split]['f1']:.3f}" for split in SPLITS))
    top = sorted(models["crf"].transitions().items(), key=lambda kv: kv[1])
    out["crf_transitions"] = {"lowest": dict(top[:10]), "highest": dict(top[-10:])}

    (ROOT / "results" / "classic.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote results/classic.json")


if __name__ == "__main__":
    main()
